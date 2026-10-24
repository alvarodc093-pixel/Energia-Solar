"""Predicción de solar fotovoltaica en España para el día siguiente.

Carga el modelo M1 (models/modelo_solar.pkl, generado con `python entrenar_modelo.py`)
y predice el día siguiente al último dato de REE con la meteo prevista por Open-Meteo,
que se puede cambiar a mano.

Para abrirla:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import fila_siguiente, preparar

BASE_DIR = Path(__file__).resolve().parent
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"
RESULTS = BASE_DIR / "data" / "resultados_modelos.csv"
MODEL_PKL = BASE_DIR / "models" / "modelo_solar.pkl"
PREVISION = ("https://api.open-meteo.com/v1/forecast?latitude=40.4168&longitude=-3.7038"
             "&daily=shortwave_radiation_sum,temperature_2m_mean&timezone=Europe%2FMadrid"
             "&past_days=14&forecast_days=3")

st.set_page_config(page_title="Solar España — previsión", page_icon=":material/wb_sunny:", layout="wide")


@st.cache_data(ttl="1h")
def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN, parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)


@st.cache_data(ttl="1h")
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS, index_col=0).sort_values("MAE")


@st.cache_resource
def load_model() -> dict:
    import warnings

    import sklearn
    try:
        with warnings.catch_warnings():  # aviso de versión: se gestiona justo debajo
            warnings.simplefilter("ignore")
            bundle = joblib.load(MODEL_PKL)
        if bundle.get("sklearn") == sklearn.__version__:
            return bundle
    except Exception:
        pass
    # Con otra versión de scikit-learn (p. ej. la demo web) el .pkl no es fiable:
    # se reentrena igual que en entrenar_modelo.py, tarda menos de un segundo.
    from entrenar_modelo import entrenar
    return entrenar(load_clean())


@st.cache_data(ttl="3h")
def meteo_prevista() -> dict:
    """Radiación y temperatura previstas en Madrid por día ({fecha: (rad, temp)}), o {} si falla."""
    try:
        import requests
        datos = requests.get(PREVISION, timeout=10).json()
    except Exception:
        try:  # en la demo web (Pyodide) requests no tiene red
            from pyodide.http import open_url
            datos = json.load(open_url(PREVISION))
        except Exception:
            return {}
    d = datos.get("daily", {})
    return {f: (r, t) for f, r, t in zip(d.get("time", []), d.get("shortwave_radiation_sum", []),
                                         d.get("temperature_2m_mean", [])) if r is not None and t is not None}


try:
    df = load_clean()
    results = load_results()
    bundle = load_model()
except FileNotFoundError as exc:
    st.error(f"Falta un archivo: {exc}. Corre `python datos.py` y `python entrenar_modelo.py` antes.")
    st.stop()

model, m1 = bundle["model"], bundle["metricas"]
dfm = preparar(df)
test = dfm[dfm.fecha >= pd.Timestamp(bundle["test_desde"])].copy()
test["pred_M1"] = model.predict(test[bundle["features"]])
ultimo = df.iloc[-1]
objetivo = (ultimo.fecha + pd.Timedelta(days=1)).date()

st.title("Cuánta solar fotovoltaica producirá España", anchor=False)
st.caption(f"Modelo lineal con meteo (M1), evaluado en el último año de datos "
           f"({bundle['test_desde']} a {bundle['test_hasta']}).")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Último dato real ({ultimo.fecha.date()})", f"{ultimo.solar_gwh:.1f} GWh", border=True)
c2.metric("Error medio en test (MAE)", f"{m1['MAE']:.1f} GWh", border=True)
c3.metric("Días con error < 20 %", f"{m1['p20']:.0f} %", border=True)
c4.metric("Datos", f"{len(df):,} días".replace(",", "."), border=True)

pred_tab, comp_tab, data_tab = st.tabs([":material/bolt: Predicción", ":material/bar_chart: Modelos",
                                        ":material/table_chart: Datos"])

with pred_tab:
    st.subheader(f"Predicción para el {objetivo:%d/%m/%Y}")
    prevista = meteo_prevista().get(str(objetivo))
    if prevista:
        st.caption("La meteo viene de la previsión de Open-Meteo para Madrid; puedes cambiarla.")
    else:
        st.caption("No hay previsión de Open-Meteo para ese día: introduce la meteo a mano.")
    rad0, temp0 = prevista or (18.0, 18.0)
    col_a, col_b = st.columns(2)
    radiacion = col_a.number_input("Radiación (MJ/m²)", 0.0, 35.0, float(round(rad0, 2)))
    temp = col_b.number_input("Temperatura media en Madrid (°C)", -10.0, 45.0, float(round(temp0, 1)))

    yhat = float(model.predict(fila_siguiente(df, radiacion, temp))[0])
    rmse = m1["RMSE"]
    st.metric("Producción estimada", f"{yhat:.1f} GWh", border=True)
    st.caption(f"Con el error típico del modelo, entre {yhat - rmse:.1f} y {yhat + rmse:.1f} GWh. "
               "Los retardos salen de los últimos 30 días reales de REE.")
    st.line_chart(df.tail(60), x="fecha", y="solar_gwh", x_label="Fecha", y_label="Solar (GWh)")

with comp_tab:
    st.subheader("Comparativa de modelos")
    st.caption("Evaluación del notebook (`proyecto_solar.ipynb`): test del 28/09/2025 al 27/09/2026.")
    st.dataframe(results, hide_index=False)
    tabla = results.reset_index(names="Modelo")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Error medio (GWh, cuanto menos mejor)**")
        st.bar_chart(tabla, x="Modelo", y="MAE", horizontal=True)
    with right, st.container(border=True):
        st.markdown("**Días con error < 20 % (cuanto más mejor)**")
        st.bar_chart(tabla, x="Modelo", y="p20", horizontal=True)
    st.markdown("**Real frente a la predicción de M1 en el último año**")
    st.line_chart(test.rename(columns={"solar_gwh": "Real", "pred_M1": "M1"}), x="fecha", y=["Real", "M1"],
                  x_label="Fecha", y_label="GWh")
    st.caption("Con meteo, el modelo lineal gana casi 3 puntos de acierto. Los árboles se quedan cortos "
               "porque en 2026 hay más potencia instalada que en su entrenamiento.")

with data_tab:
    st.subheader("Últimos datos")
    st.dataframe(df.tail(15).iloc[::-1], hide_index=True)
    st.markdown("**Radiación frente a producción (último año; el color es la temperatura)**")
    st.scatter_chart(df.tail(365), x="radiacion_MJm2", y="solar_gwh", color="temp_media_Madrid",
                     x_label="Radiación diaria (MJ/m²)", y_label="Solar (GWh)")
    st.markdown("**Media diaria por mes**")
    st.bar_chart(df.groupby(df.fecha.dt.month).solar_gwh.mean().rename_axis("Mes").reset_index(),
                 x="Mes", y="solar_gwh", y_label="GWh al día")

st.divider()
st.caption("Datos: [REE](https://www.ree.es/es/datos/apidatos) (generación solar fotovoltaica diaria) y "
           "[Open-Meteo](https://open-meteo.com) (radiación y temperatura en Madrid como referencia).")
