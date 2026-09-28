"""Predicción de solar en España.

Esta app carga los datos limpios, entrena el modelo lineal M1
(igual que en el notebook) y deja probar qué pasaría mañana
según la meteo que pongas.

Para abrirla:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"
RESULTS = BASE_DIR / "data" / "resultados_modelos.csv"

FEAT_FULL = [
    "radiacion_MJm2",
    "temp_media_Madrid",
    "dow",
    "month",
    "dayofmonth",
    "dayofyear",
    "year",
    "weekend",
    "dayofyear_sin",
    "dayofyear_cos",
    "lag1",
    "lag2",
    "lag3",
    "lag7",
    "roll7",
    "roll30",
]

st.set_page_config(page_title="Solar España — previsión", layout="wide")


@st.cache_data(ttl="1h")
def load_clean() -> pd.DataFrame:
    """Carga el csv limpio y añade lags y cosas de fecha (sin copiar el futuro)."""
    df = pd.read_csv(CLEAN, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df.dayofyear / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df.dayofyear / 365.25)
    for lag in [1, 2, 3, 7]:
        df[f"lag{lag}"] = df.solar_gwh.shift(lag)
    df["roll7"] = df.solar_gwh.shift(1).rolling(7).mean()
    df["roll30"] = df.solar_gwh.shift(1).rolling(30).mean()
    return df.dropna().reset_index(drop=True)


@st.cache_data(ttl="1h")
def load_results() -> pd.DataFrame:
    """Carga la tabla con cómo lo hizo cada modelo."""
    res = pd.read_csv(RESULTS, index_col=0)
    return res.sort_values("MAE")


@st.cache_resource
def load_model() -> tuple[LinearRegression, pd.DataFrame, pd.DataFrame]:
    """Entrena el M1 separando por fecha (test = últimos 365 días)."""
    dfm = load_clean()
    cutoff = dfm.fecha.max() - pd.Timedelta(days=364)
    train = dfm[dfm.fecha < cutoff]
    test = dfm[dfm.fecha >= cutoff].copy()
    model = LinearRegression().fit(train[FEAT_FULL], train.solar_gwh)
    test["pred_M1"] = model.predict(test[FEAT_FULL])
    return model, train, test


st.title("Cuánta solar haremos mañana en España")
st.caption("Uso un modelo lineal con meteo (M1), probado en el último año")

try:
    dfm = load_clean()
    results = load_results()
    model, train, test = load_model()
except FileNotFoundError as exc:
    st.error(f"Falta un archivo de datos: {exc}. Corre `python datos.py` antes.")
    st.stop()

last = dfm.iloc[-1]
m1 = results.loc["M1_lineal_con_meteo"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Mañana según M1", f"{test.iloc[-1]['pred_M1']:.1f} GWh", border=True)
with c2:
    st.metric("Error medio en test", f"{m1['MAE']:.2f} GWh", border=True)
with c3:
    st.metric("Acierto", f"{m1['Acierto']:.1f} %", border=True)
with c4:
    st.metric(
        f"Último dato ({last.fecha.date()})",
        f"{last.solar_gwh:.1f} GWh",
        border=True,
    )

pred_tab, comp_tab, data_tab = st.tabs(
    ["Predecir mañana", "Modelos", "Ver datos"]
)

with pred_tab:
    st.subheader("Prueba con la meteo de mañana")
    next_day = (dfm.fecha.max() + pd.Timedelta(days=1)).date()
    with st.form("predict", border=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fecha = st.date_input(
                "Qué día quieres predecir",
                value=next_day,
                min_value=dfm.fecha.max().date(),
            )
        with col_b:
            radiacion = st.number_input(
                "Radiación prevista (MJ/m²)", min_value=0.0, max_value=35.0, value=18.0
            )
        with col_c:
            temp = st.number_input(
                "Temperatura en Madrid (°C)",
                min_value=-10.0,
                max_value=45.0,
                value=18.0,
            )
        st.form_submit_button("Predecir", icon=":material/bolt:")

    hist = dfm.set_index("fecha").solar_gwh
    row = {
        "radiacion_MJm2": radiacion,
        "temp_media_Madrid": temp,
        "dow": pd.Timestamp(fecha).dayofweek,
        "month": pd.Timestamp(fecha).month,
        "dayofmonth": pd.Timestamp(fecha).day,
        "dayofyear": pd.Timestamp(fecha).dayofyear,
        "year": pd.Timestamp(fecha).year,
        "weekend": int(pd.Timestamp(fecha).dayofweek >= 5),
        "lag1": hist.iloc[-1],
        "lag2": hist.iloc[-2],
        "lag3": hist.iloc[-3],
        "lag7": hist.iloc[-7],
        "roll7": hist.iloc[-7:].mean(),
        "roll30": hist.iloc[-30:].mean(),
    }
    row["dayofyear_sin"] = float(np.sin(2 * np.pi * row["dayofyear"] / 365.25))
    row["dayofyear_cos"] = float(np.cos(2 * np.pi * row["dayofyear"] / 365.25))
    x = pd.DataFrame([row])[FEAT_FULL]
    yhat = float(model.predict(x)[0])
    rmse = float(results.loc["M1_lineal_con_meteo", "RMSE"])

    st.metric(
        f"Lo que saldría el {fecha}",
        f"{yhat:.1f} GWh",
        delta=f"± error típico {rmse:.1f} GWh",
        border=True,
    )
    st.caption(f"Más o menos estaría entre {yhat - rmse:.1f} y {yhat + rmse:.1f} GWh.")
    st.caption(
        "Los lags salen de los últimos 30 días reales; "
        "la meteo tiene que ser la prevista, no la de después."
    )

    recent = dfm.tail(60)[["fecha", "solar_gwh"]].rename(
        columns={"fecha": "Fecha", "solar_gwh": "Real"}
    )
    st.line_chart(recent, x="Fecha", y="Real")

with comp_tab:
    st.subheader("Cómo lo hizo cada modelo (365 días de prueba)")
    st.dataframe(results, hide_index=False)
    mae_df = results.reset_index(names="Modelo")[["Modelo", "MAE"]]
    acc_df = results.reset_index(names="Modelo")[["Modelo", "Acierto"]]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Error medio en test (GWh, cuanto menos mejor)**")
            st.bar_chart(mae_df, x="Modelo", y="MAE", horizontal=True)
    with right:
        with st.container(border=True):
            st.markdown("**Acierto % = 100·(1−MAPE), cuanto más mejor**")
            st.bar_chart(acc_df, x="Modelo", y="Acierto", horizontal=True)
    st.markdown("**Real contra lo que decía el M1**")
    cmp_chart = test[["fecha", "solar_gwh", "pred_M1"]].rename(
        columns={"fecha": "Fecha", "solar_gwh": "Real", "pred_M1": "M1 previsto"}
    )
    st.line_chart(cmp_chart, x="Fecha", y=["Real", "M1 previsto"])
    st.caption(
        "Con meteo el lineal gana casi 3 puntos de acierto; "
        "los árboles se quedan cortos con lo nuevo de 2026."
    )

with data_tab:
    st.subheader("Últimos datos")
    st.dataframe(dfm.tail(15).iloc[::-1], hide_index=True)
    st.markdown("**Radiación contra solar (el color es la temperatura)**")
    st.scatter_chart(
        dfm.tail(365),
        x="radiacion_MJm2",
        y="solar_gwh",
        color="temp_media_Madrid",
        x_label="Radiación diaria (MJ/m²)",
        y_label="Solar (GWh)",
    )
    st.markdown("**Media de cada mes**")
    monthly = dfm.groupby("month").solar_gwh.mean().reset_index(name="GWh media")
    monthly = monthly.rename(columns={"month": "Mes"})
    st.bar_chart(monthly, x="Mes", y="GWh media", y_label="GWh media diaria")

st.divider()
st.caption(
    "Datos de REE (solar fotovoltaica) + "
    "Open-Meteo (radiación y temperatura en Madrid como referencia). "
    "El paso a paso está en `proyecto_solar.ipynb`."
)
