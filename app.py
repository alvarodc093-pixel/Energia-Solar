"""Energía solar en España — forecasting diario (M1 lineal con meteo)."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"
RESULTS = BASE_DIR / "data" / "resultados_modelos.csv"

FEAT_FULL = [
    "radiacion_MJm2", "temp_media_Madrid", "dow", "month", "dayofmonth",
    "dayofyear", "year", "weekend", "dayofyear_sin", "dayofyear_cos",
    "lag1", "lag2", "lag3", "lag7", "roll7", "roll30",
]

st.set_page_config(page_title="Solar España — previsión", layout="wide")


@st.cache_data(ttl="1h")
def load_clean() -> pd.DataFrame:
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
    res = pd.read_csv(RESULTS, index_col=0)
    return res.sort_values("MAE")


@st.cache_resource
def load_model(_df: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, pd.DataFrame]:
    cutoff = _df.fecha.max() - pd.Timedelta(days=364)
    train = _df[_df.fecha < cutoff]
    test = _df[_df.fecha >= cutoff]
    model = LinearRegression().fit(train[FEAT_FULL], train.solar_gwh)
    pred = model.predict(test[FEAT_FULL])
    test = test.copy()
    test["pred_M1"] = pred
    return model, train, test


st.title("Energía solar en España — previsión diaria")
st.caption("Modelo M1: regresión lineal con meteorología · test últimos 365 días")

try:
    dfm = load_clean()
    results = load_results()
    model, train, test = load_model(dfm)
except FileNotFoundError as exc:
    st.error(f"Falta un fichero de datos: {exc}. Ejecuta `python datos.py` primero.")
    st.stop()

last = dfm.iloc[-1]
m1 = results.loc["M1_lineal_con_meteo"]

with st.container(horizontal=True):
    st.metric("Predicción M1 mañana", f"{test.iloc[-1]['pred_M1']:.1f} GWh", border=True)
    st.metric("MAE en test (M1)", f"{m1['MAE']:.2f} GWh", border=True)
    st.metric("Acierto M1", f"{m1['Acierto']:.1f} %", border=True)
    st.metric(
        f"Último dato real ({last.fecha.date()})",
        f"{last.solar_gwh:.1f} GWh",
        border=True,
    )

pred_tab, comp_tab, data_tab = st.tabs(
    ["Predicción mañana", "Comparativa de modelos", "Exploración"]
)

with pred_tab:
    st.subheader("Predecir la producción de mañana")
    next_day = (dfm.fecha.max() + pd.Timedelta(days=1)).date()
    with st.form("predict", border=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fecha = st.date_input("Fecha a predecir", value=next_day)
        with col_b:
            radiacion = st.number_input(
                "Radiación prevista (MJ/m²)", min_value=0.0, max_value=35.0, value=18.0
            )
        with col_c:
            temp = st.number_input(
                "Temperatura media Madrid (°C)",
                min_value=-10.0,
                max_value=45.0,
                value=18.0,
            )
        submitted = st.form_submit_button("Predecir", icon=":material/bolt:")

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

    if submitted or True:
        st.metric(
            f"Producción prevista {fecha}",
            f"{yhat:.1f} GWh",
            delta=f"±RMSE {rmse:.1f} GWh",
            border=True,
        )
        st.caption(f"Intervalo orientativo: {yhat - rmse:.1f} – {yhat + rmse:.1f} GWh.")
        st.caption(
            "Lags calculados desde los últimos 30 días observados; "
            "la meteo debe ser la prevista, no la observada."
        )

    recent = dfm.tail(60)[["fecha", "solar_gwh"]].rename(
        columns={"fecha": "Fecha", "solar_gwh": "Real"}
    )
    st.line_chart(recent, x="Fecha", y="Real")

with comp_tab:
    st.subheader("Comparativa en test (365 días)")
    st.dataframe(results, hide_index=False)
    mae_df = results.reset_index(names="Modelo")[["Modelo", "MAE"]]
    acc_df = results.reset_index(names="Modelo")[["Modelo", "Acierto"]]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**MAE en test (GWh, menor es mejor)**")
            st.bar_chart(mae_df, x="Modelo", y="MAE", horizontal=True)
    with right:
        with st.container(border=True):
            st.markdown("**Acierto % = 100·(1−MAPE), mayor es mejor**")
            st.bar_chart(acc_df, x="Modelo", y="Acierto", horizontal=True)
    st.markdown("**Real vs M1 en test**")
    cmp_chart = test[["fecha", "solar_gwh", "pred_M1"]].rename(
        columns={"fecha": "Fecha", "solar_gwh": "Real", "pred_M1": "M1 previsto"}
    )
    st.line_chart(cmp_chart, x="Fecha", y=["Real", "M1 previsto"])
    st.caption(
        "La meteo aporta +2.7 pp de acierto en el lineal; "
        "los árboles no extrapolan la capacidad nueva de 2026."
    )

with data_tab:
    st.subheader("Datos recientes")
    st.dataframe(dfm.tail(15).iloc[::-1], hide_index=True)
    st.markdown("**Radiación vs solar (color = temperatura)**")
    st.scatter_chart(
        dfm.tail(365),
        x="radiacion_MJm2",
        y="solar_gwh",
        color="temp_media_Madrid",
        x_label="Radiación diaria (MJ/m²)",
        y_label="Solar (GWh)",
    )
    st.markdown("**Media mensual histórica**")
    monthly = dfm.groupby("month").solar_gwh.mean().reset_index(name="GWh media")
    monthly = monthly.rename(columns={"month": "Mes"})
    st.bar_chart(monthly, x="Mes", y="GWh media", y_label="GWh media diaria")
