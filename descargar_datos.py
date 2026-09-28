"""Descarga datos reales: REE solar fotovoltaica + Open-Meteo radiación Madrid."""
import requests
import pandas as pd

OUT = "data/solar_espana.csv"

def fetch_ree(start, end):
    url = "https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
    params = {"start_date": f"{start}T00:00", "end_date": f"{end}T23:59", "time_trunc": "day"}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    j = r.json()
    # buscar Solar fotovoltaica
    for item in j.get("included", []):
        title = item.get("attributes", {}).get("title", "")
        if "fotovoltaica" in title.lower():
            vals = item["attributes"]["values"]
            df = pd.DataFrame(vals)
            df["fecha"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize()
            df = df.rename(columns={"value": "solar_mwh"})
            # MWh -> GWh
            df["solar_gwh"] = df["solar_mwh"] / 1000.0
            return df[["fecha", "solar_gwh"]]
    raise ValueError("No se encontró Solar fotovoltaica en respuesta REE")

def fetch_meteo(start, end):
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
    df = pd.DataFrame({"fecha": pd.to_datetime(j["time"]),
                       "radiacion_MJm2": j["shortwave_radiation_sum"],
                       "temp_media_Madrid": j["temperature_2m_mean"]})
    return df

# Por tramos anuales para no saturar la API de REE
tramos = [("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
          ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-09-27")]
dfs = [fetch_ree(s, e) for s, e in tramos]
solar = pd.concat(dfs, ignore_index=True)

met = fetch_meteo("2023-01-01", "2026-09-27")
df = pd.merge(solar, met, on="fecha", how="inner").sort_values("fecha")
df.to_csv(OUT, index=False)
print(f"Guardado {OUT}: {df.shape}")
print(df.head(3).to_string())
print(df.tail(3).to_string())
print(df.describe().to_string())
