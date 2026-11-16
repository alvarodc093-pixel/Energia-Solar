"""Variables del modelo, compartidas por el entrenamiento y la app.

Todas las variables que miran la propia serie usan datos hasta el día anterior
(shift), así el modelo nunca ve el valor que tiene que predecir.
"""
import numpy as np
import pandas as pd

INICIO_SERIE = pd.Timestamp("2023-01-01")

METEO = ["radiacion_MJm2", "temp_media_Madrid"]
# Sin month/dayofmonth/dayofyear/year: en un modelo lineal son casi colineales y
# provocan saltos en cada cambio de mes y de año. La estacionalidad va en seno y
# coseno, y el crecimiento de potencia en una tendencia continua.
CALENDARIO = ["dow", "weekend", "dayofyear_sin", "dayofyear_cos", "tendencia_anios"]
RETARDOS = ["lag1", "lag2", "lag3", "lag7", "roll7", "roll30"]
# Radiación del día por los GWh por MJ/m² de los 30 días anteriores: recoge que
# la misma radiación da más energía cuanta más potencia hay instalada.
INTERACCION = ["rad_x_ratio30"]
FEAT_FULL = METEO + CALENDARIO + RETARDOS + INTERACCION


def calendario(fechas: pd.Series) -> pd.DataFrame:
    """Variables de calendario a partir de una serie de fechas."""
    f = pd.to_datetime(fechas)
    doy = f.dt.dayofyear
    out = pd.DataFrame({"dow": f.dt.dayofweek}, index=fechas.index)
    out["weekend"] = (out.dow >= 5).astype(int)
    out["dayofyear_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["tendencia_anios"] = (f - INICIO_SERIE).dt.days / 365.25
    return out


def comprobar_dias_seguidos(fechas: pd.Series) -> None:
    """Los retardos se calculan por filas: si falta un día, lag1 dejaría de ser ayer."""
    saltos = pd.to_datetime(fechas).sort_values().diff().dropna()
    malos = saltos[saltos != pd.Timedelta(days=1)]
    if len(malos):
        raise ValueError(f"La serie no tiene un día por fila: {len(malos)} saltos, "
                         f"el primero antes de {fechas.loc[malos.index[0]]}")


def _ratio_produccion_radiacion(df: pd.DataFrame) -> pd.Series:
    return df.solar_gwh / df.radiacion_MJm2.where(df.radiacion_MJm2 > 0)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Añade calendario y retardos a la serie diaria y quita las filas sin historia."""
    df = df.sort_values("fecha").reset_index(drop=True)
    comprobar_dias_seguidos(df["fecha"])
    df = df.drop(columns=[c for c in CALENDARIO if c in df.columns])
    df = pd.concat([df, calendario(df["fecha"])], axis=1)
    for lag in [1, 2, 3, 7]:
        df[f"lag{lag}"] = df.solar_gwh.shift(lag)
    df["roll7"] = df.solar_gwh.shift(1).rolling(7).mean()
    df["roll30"] = df.solar_gwh.shift(1).rolling(30).mean()
    ratio30 = _ratio_produccion_radiacion(df).shift(1).rolling(30).mean()
    df["rad_x_ratio30"] = df.radiacion_MJm2 * ratio30
    return df.dropna(subset=RETARDOS + INTERACCION).reset_index(drop=True)


def fila_siguiente(historia: pd.DataFrame, radiacion: float, temp: float) -> pd.DataFrame:
    """Variables para el día siguiente al último de la historia, con la meteo indicada."""
    historia = historia.sort_values("fecha")
    comprobar_dias_seguidos(historia["fecha"].tail(30))
    serie = historia.solar_gwh.to_numpy()
    ratio30 = _ratio_produccion_radiacion(historia).tail(30).mean()
    fecha = pd.Series([historia.fecha.max() + pd.Timedelta(days=1)])
    fila = calendario(fecha).iloc[0].to_dict()
    fila.update({
        "radiacion_MJm2": radiacion,
        "temp_media_Madrid": temp,
        "lag1": serie[-1], "lag2": serie[-2], "lag3": serie[-3], "lag7": serie[-7],
        "roll7": serie[-7:].mean(), "roll30": serie[-30:].mean(),
        "rad_x_ratio30": radiacion * ratio30,
    })
    return pd.DataFrame([fila])[FEAT_FULL]


def separar(dfm: pd.DataFrame, dias_test: int = 365):
    """Partición temporal: los últimos `dias_test` días son el test."""
    cutoff = dfm.fecha.max() - pd.Timedelta(days=dias_test - 1)
    return dfm[dfm.fecha < cutoff], dfm[dfm.fecha >= cutoff]


def metricas(real, pred) -> dict:
    """MAE, RMSE, MAPE, WAPE y % de días con error relativo <= 10 % y <= 20 %."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    err = np.abs((real - pred) / real)
    return {
        "MAE": float(np.mean(np.abs(real - pred))),
        "RMSE": float(np.sqrt(np.mean((real - pred) ** 2))),
        "MAPE": float(err.mean()),
        "WAPE": float(np.abs(real - pred).sum() / real.sum() * 100),
        "Acierto": float(100 * (1 - err.mean())),
        "p10": float((err <= 0.10).mean() * 100),
        "p20": float((err <= 0.20).mean() * 100),
    }
