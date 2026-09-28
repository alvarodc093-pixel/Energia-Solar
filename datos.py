"""Preparación local de datos solares (sin internet).

Uso:
    python datos.py

Lee data/solar_espana.csv, verifica calidad, añade features de
calendario y guarda data/solar_espana_clean.csv.

Este fichero existe para que `python datos.py` funcione: el script de
descarga desde APIs es `descargar_datos.py` (requiere internet).
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW = BASE_DIR / "data" / "solar_espana.csv"
CLEAN = BASE_DIR / "data" / "solar_espana_clean.csv"


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(
            f"No se encuentra {RAW}. "
            "Ejecuta primero: python descargar_datos.py (necesita internet)"
        )

    df = pd.read_csv(RAW, parse_dates=["fecha"])
    print(f"Filas antes: {len(df)}")
    print("Nulos totales:", int(df.isna().sum().sum()))
    print("Duplicados fecha:", int(df.duplicated("fecha").sum()))
    esperadas = len(pd.date_range(df.fecha.min(), df.fecha.max()))
    print("Fechas faltantes:", esperadas - len(df))
    print(
        "solar<=0:",
        int((df.solar_gwh <= 0).sum()),
        "| radiacion<0:",
        int((df.radiacion_MJm2 < 0).sum()),
    )

    df = df.sort_values("fecha").reset_index(drop=True)
    df["dow"] = df.fecha.dt.dayofweek
    df["month"] = df.fecha.dt.month
    df["dayofmonth"] = df.fecha.dt.day
    df["dayofyear"] = df.fecha.dt.dayofyear
    df["year"] = df.fecha.dt.year
    df["weekend"] = (df.dow >= 5).astype(int)

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    print(f"Filas después: {len(df)} -> {CLEAN}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
