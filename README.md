# Energía solar en España — Forecasting diario

**Pregunta de negocio:** ¿cuánta energía solar fotovoltaica producirá España mañana (GWh/día) para planificar la red?

## Datos
- `data/solar_espana.csv`: 1366 días (2023-01-01 a 2026-09-27). REE (solar fotovoltaica) + Open-Meteo Madrid (radiación, temperatura).
- `data/solar_espana_clean.csv`: dataset limpio + features de calendario.
- `data/resultados_modelos.csv`: comparativa de modelos en test.

## Notebook
`proyecto_solar.ipynb` (todo el proyecto, con gráficas inline):
1. Pregunta de negocio · 2. Limpieza · 3. EDA · 4. Split temporal + baselines · 5. Modelos · 6. Conclusiones.

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

% acierto = 100·(1−MAPE). La meteo aporta ~+2.7 pp; los árboles no extrapolan la tendencia de capacidad de 2026.

## Uso
```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
# abrir proyecto_solar.ipynb y ejecutar, o:
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute proyecto_solar.ipynb --output proyecto_solar.ipynb --allow-errors
```
