"""Variables del modelo, compartidas por el entrenamiento y la app.

Todas las variables que miran la propia serie usan datos hasta el día anterior
(shift), así el modelo nunca ve el valor que tiene que predecir.
"""
import numpy as np
import pandas as pd

METEO = ["radiacion_MJm2", "temp_media_Madrid"]
CALENDARIO = ["dow", "month", "dayofmonth", "dayofyear", "year", "weekend", "dayofyear_sin", "dayofyear_cos"]
RETARDOS = ["lag1", "lag2", "lag3", "lag7", "roll7", "roll30"]
FEAT_FULL = METEO + CALENDARIO + RETARDOS  # mismo orden que en el notebook


def calendario(fechas: pd.Series) -> pd.DataFrame:
    """Variables de calendario a partir de una serie de fechas."""
    f = pd.to_datetime(fechas)
    out = pd.DataFrame({
        "dow": f.dt.dayofweek,
        "month": f.dt.month,
        "dayofmonth": f.dt.day,
        "dayofyear": f.dt.dayofyear,
        "year": f.dt.year,
    }, index=fechas.index)
    out["weekend"] = (out.dow >= 5).astype(int)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out.dayofyear / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out.dayofyear / 365.25)
    return out


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Añade calendario y retardos a la serie diaria y quita las filas sin historia."""
    df = df.sort_values("fecha").reset_index(drop=True)
    df = df.drop(columns=[c for c in CALENDARIO if c in df.columns])
    df = pd.concat([df, calendario(df["fecha"])], axis=1)
    for lag in [1, 2, 3, 7]:
        df[f"lag{lag}"] = df.solar_gwh.shift(lag)
    df["roll7"] = df.solar_gwh.shift(1).rolling(7).mean()
    df["roll30"] = df.solar_gwh.shift(1).rolling(30).mean()
    return df.dropna(subset=RETARDOS).reset_index(drop=True)


def fila_siguiente(historia: pd.DataFrame, radiacion: float, temp: float) -> pd.DataFrame:
    """Variables para el día siguiente al último de la historia, con la meteo indicada."""
    serie = historia.sort_values("fecha").solar_gwh.to_numpy()
    fecha = pd.Series([historia.fecha.max() + pd.Timedelta(days=1)])
    fila = calendario(fecha).iloc[0].to_dict()
    fila.update({
        "radiacion_MJm2": radiacion,
        "temp_media_Madrid": temp,
        "lag1": serie[-1], "lag2": serie[-2], "lag3": serie[-3], "lag7": serie[-7],
        "roll7": serie[-7:].mean(), "roll30": serie[-30:].mean(),
    })
    return pd.DataFrame([fila])[FEAT_FULL]


def separar(dfm: pd.DataFrame, dias_test: int = 365):
    """Partición temporal: los últimos `dias_test` días son el test."""
    cutoff = dfm.fecha.max() - pd.Timedelta(days=dias_test - 1)
    return dfm[dfm.fecha < cutoff], dfm[dfm.fecha >= cutoff]


def metricas(real, pred) -> dict:
    """Las mismas métricas que el notebook."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    err = np.abs((real - pred) / real)
    return {
        "MAE": float(np.mean(np.abs(real - pred))),
        "RMSE": float(np.sqrt(np.mean((real - pred) ** 2))),
        "MAPE": float(err.mean()),
        "Acierto": float(100 * (1 - err.mean())),
        "p10": float((err <= 0.10).mean() * 100),
        "p20": float((err <= 0.20).mean() * 100),
    }
