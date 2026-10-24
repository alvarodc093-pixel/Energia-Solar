"""Descarga datos reales: REE solar fotovoltaica + Open-Meteo radiación Madrid.

Uso:
    python descargar_datos.py                 # desde 2023-01-01 hasta el último día disponible
    python descargar_datos.py --hasta 2026-09-27
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "data" / "solar_espana.csv"
INICIO = date(2023, 1, 1)


def fetch_ree(start: str, end: str) -> pd.DataFrame:
    url = "https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
    params = {"start_date": f"{start}T00:00", "end_date": f"{end}T23:59", "time_trunc": "day"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    for item in r.json().get("included", []):
        title = item.get("attributes", {}).get("title", "")
        if "fotovoltaica" in title.lower():
            df = pd.DataFrame(item["attributes"]["values"])
            df["fecha"] = (pd.to_datetime(df["datetime"], utc=True)
                           .dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize())
            df["solar_gwh"] = df["value"] / 1000.0  # MWh -> GWh
            return df[["fecha", "solar_gwh"]]
    raise ValueError("No se encontró Solar fotovoltaica en la respuesta de REE")


def fetch_meteo(start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.4168, "longitude": -3.7038,
        "start_date": start, "end_date": end,
        "daily": "shortwave_radiation_sum,temperature_2m_mean",
        "timezone": "Europe/Madrid",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()["daily"]
    return pd.DataFrame({"fecha": pd.to_datetime(j["time"]),
                         "radiacion_MJm2": j["shortwave_radiation_sum"],
                         "temp_media_Madrid": j["temperature_2m_mean"]})


def tramos_anuales(inicio: date, fin: date):
    """REE limita el tamaño de cada consulta: se pide año a año."""
    for anio in range(inicio.year, fin.year + 1):
        desde = max(inicio, date(anio, 1, 1))
        hasta = min(fin, date(anio, 12, 31))
        yield desde.isoformat(), hasta.isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hasta", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    args = parser.parse_args()

    try:
        solar = pd.concat([fetch_ree(s, e) for s, e in tramos_anuales(INICIO, args.hasta)], ignore_index=True)
        met = fetch_meteo(INICIO.isoformat(), args.hasta.isoformat())
    except requests.RequestException as exc:
        raise SystemExit(f"Error descargando datos: {exc}") from exc

    # El archivo de Open-Meteo va unos días por detrás: los días sin meteo se quedan fuera.
    df = (pd.merge(solar, met, on="fecha", how="inner")
          .dropna().drop_duplicates("fecha").sort_values("fecha"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Guardado {OUT}: {len(df)} días, del {df.fecha.min().date()} al {df.fecha.max().date()}")


if __name__ == "__main__":
    main()
