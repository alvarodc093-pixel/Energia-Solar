"""Valida data/solar_espana.csv y genera data/solar_espana_clean.csv.

Uso:
    python datos.py

Falla sin escribir nada si hay nulos, fechas repetidas, días que faltan o
valores imposibles, para que el workflow semanal no reentrene con datos malos.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "data" / "solar_espana.csv"
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"
COLUMNAS = ["fecha", "solar_gwh", "radiacion_MJm2", "temp_media_Madrid"]


def validar(df: pd.DataFrame) -> None:
    """Lanza ValueError con la lista de problemas si la serie no es válida."""
    faltan_columnas = set(COLUMNAS) - set(df.columns)
    if faltan_columnas:
        raise ValueError(f"Faltan columnas: {sorted(faltan_columnas)}")
    fechas = pd.to_datetime(df.fecha)
    esperadas = len(pd.date_range(fechas.min(), fechas.max()))
    problemas = {
        "nulos": int(df[COLUMNAS].isna().sum().sum()),
        "fechas duplicadas": int(fechas.duplicated().sum()),
        "días que faltan": esperadas - fechas.nunique(),
        "solar <= 0": int((df.solar_gwh <= 0).sum()),
        "radiación fuera de 0-40": int((~df.radiacion_MJm2.between(0, 40) & df.radiacion_MJm2.notna()).sum()),
        "temperatura fuera de -20-50": int((~df.temp_media_Madrid.between(-20, 50)
                                            & df.temp_media_Madrid.notna()).sum()),
    }
    malos = {k: v for k, v in problemas.items() if v}
    if malos:
        raise ValueError(f"Datos no válidos: {malos}")


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"No se encuentra {RAW}. Ejecuta primero: python descargar_datos.py")

    df = pd.read_csv(RAW, parse_dates=["fecha"])
    try:
        validar(df)
    except ValueError as exc:
        raise SystemExit(f"{exc}. No se guarda {CLEAN.name}.") from exc

    df = df.sort_values("fecha").reset_index(drop=True)
    df["dow"] = df.fecha.dt.dayofweek
    df["month"] = df.fecha.dt.month
    df["dayofmonth"] = df.fecha.dt.day
    df["dayofyear"] = df.fecha.dt.dayofyear
    df["year"] = df.fecha.dt.year
    df["weekend"] = (df.dow >= 5).astype(int)

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    print(f"{len(df)} días válidos, del {df.fecha.min().date()} al {df.fecha.max().date()} -> {CLEAN}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
