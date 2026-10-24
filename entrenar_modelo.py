"""Entrena el modelo M1 (lineal con meteo) y lo guarda en models/modelo_solar.pkl.

Flujo:
    1. python descargar_datos.py  -> data/solar_espana.csv (REE + Open-Meteo)
    2. python datos.py            -> data/solar_espana_clean.csv
    3. python entrenar_modelo.py  -> models/modelo_solar.pkl
    4. streamlit run app.py

El .pkl guarda un dict con el modelo, la lista de features, el periodo de test
y sus métricas, para que la app muestre el error del modelo que está usando.
"""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

from features import FEAT_FULL, metricas, preparar, separar

BASE_DIR = Path(__file__).resolve().parent
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"
OUT = BASE_DIR / "models" / "modelo_solar.pkl"


def entrenar(df: pd.DataFrame) -> dict:
    """Entrena M1 con todo salvo el último año y lo evalúa en ese año."""
    dfm = preparar(df)
    train, test = separar(dfm)
    model = LinearRegression().fit(train[FEAT_FULL], train.solar_gwh)
    return {
        "model": model,
        "features": FEAT_FULL,
        "sklearn": sklearn.__version__,
        "test_desde": str(test.fecha.min().date()),
        "test_hasta": str(test.fecha.max().date()),
        "metricas": metricas(test.solar_gwh, model.predict(test[FEAT_FULL])),
    }


def main() -> None:
    if not CLEAN.exists():
        raise FileNotFoundError(f"No se encuentra {CLEAN}. Ejecuta primero: python datos.py")
    bundle = entrenar(pd.read_csv(CLEAN, parse_dates=["fecha"]))
    m = bundle["metricas"]
    print(f"Test {bundle['test_desde']} -> {bundle['test_hasta']}: "
          f"MAE {m['MAE']:.2f} GWh | RMSE {m['RMSE']:.2f} | acierto {m['Acierto']:.1f} %")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT)
    print(f"Guardado {OUT}")


if __name__ == "__main__":
    main()
