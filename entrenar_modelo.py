"""Entrena el modelo M1 (lineal con meteo) y lo guarda en un .pkl.

Flujo clasico pedido en clase:
    1. python datos.py            -> deja data/solar_espana_clean.csv
    2. python entrenar_modelo.py  -> entrena M1 y hace joblib.dump a models/modelo_solar.pkl
    3. streamlit run app.py       -> la app hace joblib.load del .pkl (no reentrena)

El .pkl guarda un dict con: el modelo, la lista de features y el RMSE en test.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"
OUT = BASE_DIR / "models" / "modelo_solar.pkl"

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


def main() -> None:
    if not CLEAN.exists():
        raise FileNotFoundError(
            f"No se encuentra {CLEAN}. Ejecuta primero: python datos.py"
        )

    df = pd.read_csv(CLEAN, parse_dates=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df.dayofyear / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df.dayofyear / 365.25)
    for lag in [1, 2, 3, 7]:
        df[f"lag{lag}"] = df.solar_gwh.shift(lag)
    df["roll7"] = df.solar_gwh.shift(1).rolling(7).mean()
    df["roll30"] = df.solar_gwh.shift(1).rolling(30).mean()
    dfm = df.dropna().reset_index(drop=True)

    cutoff = dfm.fecha.max() - pd.Timedelta(days=364)
    train = dfm[dfm.fecha < cutoff]
    test = dfm[dfm.fecha >= cutoff]
    print(f"train: {train.fecha.min().date()} -> {train.fecha.max().date()} ({len(train)})")
    print(f"test:  {test.fecha.min().date()} -> {test.fecha.max().date()} ({len(test)})")

    model = LinearRegression().fit(train[FEAT_FULL], train.solar_gwh)
    pred = model.predict(test[FEAT_FULL])
    rmse = float(np.sqrt(mean_squared_error(test.solar_gwh, pred)))
    print(f"RMSE en test: {rmse:.2f} GWh")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEAT_FULL, "rmse": rmse}, OUT)
    print(f"Guardado {OUT}")


if __name__ == "__main__":
    main()
