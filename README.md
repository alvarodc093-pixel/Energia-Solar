# Energía solar en España — Forecasting diario

**Pregunta de negocio:** ¿cuánta energía solar fotovoltaica producirá España **mañana** (GWh/día) para planificar la red (Red Eléctrica)?

Proyecto de serie temporal con datos reales: **REE** (generación solar fotovoltaica diaria) + **Open-Meteo** (radiación y temperatura en Madrid como proxy nacional). Incluye limpieza, EDA, split temporal, baselines, 5 modelos comparados con/sin meteorología, conclusiones y app interactiva.

## Estructura

```text
.
├── proyecto_solar.ipynb        # Proyecto completo y ejecutado (pasos 1–6, gráficas inline)
├── app.py                      # App Streamlit: predicción de mañana + comparativa + exploración
├── datos.py                    # Limpieza local: raw -> clean (sin internet)
├── descargar_datos.py          # Descarga REE + Open-Meteo -> data/solar_espana.csv (requiere internet)
├── requirements.txt            # Dependencias pineadas
└── data/
    ├── solar_espana.csv        # 1366 días (2023-01-01 a 2026-09-27): solar + radiación + temp
    ├── solar_espana_clean.csv  # Dataset limpio + features de calendario
    └── resultados_modelos.csv  # Comparativa de modelos en test (la genera el notebook)
```

## Datos

- `data/solar_espana.csv`: 1366 filas, sin nulos/duplicados/huecos. Columnas: `fecha`, `solar_gwh`, `radiacion_MJm2`, `temp_media_Madrid`.
- `data/solar_espana_clean.csv`: + calendario (`dow`, `month`, `dayofmonth`, `dayofyear`, `year`, `weekend`).
- Features de modelado (notebook y app, sin fuga): calendario + `dayofyear_sin/cos` + meteo del día + `lag1/2/3/7`, `roll7`, `roll30` (con `shift(1)`).
- Split temporal: train 2023-01-31 → 2025-09-27 (971 días), test últimos 365 días (2025-09-28 → 2026-09-27, incluye 2026 no visto en train para medir extrapolación real).

Fuentes: [REE estructura de generación](https://apidatos.ree.es/es/datos/generacion/estructura-generacion) y [Open-Meteo archive](https://archive-api.open-meteo.com/v1/archive) (Madrid 40.41, −3.70).

## Instalación y uso

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux/Mac:
# .venv/bin/python -m pip install -r requirements.txt
```

Regenerar el dataset limpio (sin internet):

```bash
.venv\Scripts\python.exe datos.py
```

Re-ejecutar el notebook de punta a punta:

```bash
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute proyecto_solar.ipynb --output proyecto_solar.ipynb --allow-errors
```

Solo descargar datos frescos desde las APIs (requiere internet):

```bash
.venv\Scripts\python.exe descargar_datos.py
.venv\Scripts\python.exe datos.py
```

Lanzar la app:

```bash
.venv\Scripts\streamlit.exe run app.py
```

## Notebook (`proyecto_solar.ipynb`, ejecutado)

1. **Pregunta de negocio** — forecasting diario, `y = solar_gwh`.
2. **Limpieza** — verificación (0 nulos, 0 duplicados, 0 huecos, 0 imposibles; extremos coherentes) → no se elimina ninguna fila.
3. **EDA** — serie temporal + media móvil 30d, radiación vs solar por año (corr 0.85), boxplot mensual, temp vs solar + media anual, día de semana (~nulo), histogramas, heatmap de correlación, año×mes, ratio solar/radiación (efecto capacidad), autocorrelación (lag-1: 0.92).
4. **Split temporal + baselines** — B0 naive (ayer), B1 media móvil 7d, B2 media mensual histórica.
5. **Modelos** — M1 lineal con meteo, M2 lineal sin meteo, M3 RandomForest, M4 HistGradientBoosting, M5 RF sin meteo. Métricas: MAE, RMSE, MAPE, R², % acierto = 100·(1−MAPE), % días con error ≤10 % / ≤20 %.
6. **Conclusiones** — ver abajo. Guarda `data/resultados_modelos.csv`.

## Resultados (test: últimos 365 días)

| Modelo | MAE | R² | % acierto |
|---|---|---|---|
| M1 lineal con meteo | 16.54 | 0.919 | 84.5 |
| M2 lineal sin meteo | 19.51 | 0.882 | 81.8 |
| B1 media móvil 7d | 22.03 | 0.849 | 80.2 |
| B0 naive (ayer) | 22.20 | 0.839 | 80.2 |
| M3 RandomForest | 23.93 | 0.818 | 84.3 |
| M4 HistGradientBoosting | 24.55 | 0.807 | 83.6 |
| M5 RF sin meteo | 24.98 | 0.812 | 80.5 |
| B2 media mensual hist. | 48.70 | 0.336 | 68.9 |

% acierto = 100·(1−MAPE). Detalle completo (RMSE, p10, p20) en `data/resultados_modelos.csv`.

## Conclusiones

1. **Mejor modelo: M1 lineal con meteo** (MAE 16.54 GWh, R² 0.919, 84.5 % acierto; 57.8 % de días con error ≤10 % y 82.5 % con error ≤20 %). Supera al mejor baseline en ~4.3 pp.
2. **La meteorología es decisiva:** el lineal pasa de 81.8 % a 84.5 % con radiación+temperatura (MAE 19.51 → 16.54); el RF de 80.5 % a 84.3 %.
3. **Los árboles no extrapolan la tendencia:** RF/HGB pierden contra el lineal (e incluso contra la media móvil en MAE) porque 2026 trae más potencia instalada no vista en train; el lineal extrapola vía `year` + relación radiación→solar.
4. **Respuesta de negocio:** predecir mañana con M1 (calendario + meteo prevista + últimos 7–30 días); 4 de cada 5 días el error es ≤20 %, útil para reserva de red.
5. **Limitaciones:** se asume meteo prevista perfecta, Madrid como proxy nacional y necesidad de reentrenar cada año por nueva capacidad. Siguiente paso: reentreno mensual, intervalos por cuantiles y evaluar con meteo prevista real.

## Reproducibilidad

- Notebook ejecutado de principio a fin (celdas con salidas y gráficas inline).
- `python datos.py` verificado: 1366 filas, 0 nulos/duplicados/huecos.
- App verificada: reentrena M1 con el mismo split y reproduce MAE 16.54 en test.
- Dependencias pineadas en `requirements.txt` (probado con Python 3.14, Streamlit 1.64).
