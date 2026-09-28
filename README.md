# Energía solar en España - predecir lo de mañana

La idea es sencilla: adivinar cuánta solar fotovoltaica va a producir España mañana (en GWh por día). Esto le sirve a Red Eléctrica para organizar la red.

Usé datos reales: la generación de REE + la meteo de Open-Meteo (radiación y temperatura de Madrid, que uso como referencia para toda España). Al final comparé varios modelos con y sin meteo y monté una app para probar.

## Qué hay en cada archivo

```text
.
├── proyecto_solar.ipynb        # todo el proyecto paso a paso, con gráficas
├── app.py                      # app en Streamlit para predecir mañana
├── datos.py                    # deja los datos limpios sin usar internet
├── descargar_datos.py          # baja los datos de REE y Open-Meteo (aquí sí hace falta internet)
├── requirements.txt            # lo que hay que instalar
└── data/
    ├── solar_espana.csv        # 1366 días del 2023-01-01 al 2026-09-27
    ├── solar_espana_clean.csv  # lo mismo pero con columnas de fecha añadidas
    └── resultados_modelos.csv  # tabla con cómo lo hizo cada modelo (la crea el notebook)
```

## Los datos

- `solar_espana.csv`: 1366 filas, no tenía nulos ni duplicados ni días sueltos. Columnas: `fecha`, `solar_gwh`, `radiacion_MJm2`, `temp_media_Madrid`.
- `solar_espana_clean.csv`: le añadí `dow`, `month`, `dayofmonth`, `dayofyear`, `year`, `weekend` a partir de la fecha.
- Para predecir usé: calendario + `dayofyear_sin/cos` + meteo del día + `lag1/2/3/7`, `roll7`, `roll30` (siempre con `shift(1)` para no copiar el futuro).
- Separé train y test por fecha: train del 2023-01-31 al 2025-09-27 (971 días), test los últimos 365 días (2025-09-28 al 2026-09-27). El test tiene 2026, que el modelo no había visto, así se ve si aguanta de verdad.

Fuentes: [REE](https://apidatos.ree.es/es/datos/generacion/estructura-generacion) y [Open-Meteo](https://archive-api.open-meteo.com/v1/archive) (Madrid 40.41, -3.70).

## Cómo probarlo

```bash
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dejar los datos limpios:

```bash
.venv\Scripts\python.exe datos.py
```

Volver a correr el notebook entero:

```bash
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute proyecto_solar.ipynb --output proyecto_solar.ipynb --allow-errors
```

Bajar datos nuevos (hace falta internet):

```bash
.venv\Scripts\python.exe descargar_datos.py
.venv\Scripts\python.exe datos.py
```

Abrir la app:

```bash
.venv\Scripts\streamlit.exe run app.py
```

## Qué hice en el notebook

1. Planteé la pregunta: predecir `solar_gwh` de mañana.
2. Revisé los datos: 0 nulos, 0 duplicados, 0 huecos, 0 valores raros. Los extremos cuadraban (días nublados de invierno y veranos de 2026 con más placas), así que no borré nada.
3. Miré los datos con calma: serie en el tiempo + media de 30 días, radiación contra solar por año (corr 0.85), cajas por mes, temp contra solar, día de la semana (casi no afecta), histogramas, correlaciones, tabla año x mes, ratio solar/radiación (se ve que hay más capacidad cada año), y autocorrelación (lag-1: 0.92).
4. Separé train/test e hice 3 baselines tontos: B0 lo de ayer, B1 media de 7 días, B2 media del mes de años anteriores.
5. Probé 5 modelos: M1 lineal con meteo, M2 lineal sin meteo, M3 RandomForest, M4 HistGradientBoosting, M5 RF sin meteo. Medí MAE, RMSE, MAPE, R2, % acierto = 100*(1-MAPE) y % de días con error pequeño.
6. Saqué conclusiones y guardé `resultados_modelos.csv`.

## Cómo quedó cada modelo (en test)

| Modelo | MAE | R² | % acierto |
|---|---|---|---|
| M1 lineal con meteo | 16.54 | 0.919 | 84.5 |
| M2 lineal sin meteo | 19.51 | 0.882 | 81.8 |
| B1 media móvil 7d | 22.03 | 0.849 | 80.2 |
| B0 lo de ayer | 22.20 | 0.839 | 80.2 |
| M3 RandomForest | 23.93 | 0.818 | 84.3 |
| M4 HistGradientBoosting | 24.55 | 0.807 | 83.6 |
| M5 RF sin meteo | 24.98 | 0.812 | 80.5 |
| B2 media mensual | 48.70 | 0.336 | 68.9 |

El detalle (RMSE, p10, p20) está en `data/resultados_modelos.csv`.

## Lo que saqué en claro

1. El mejor fue el M1, el lineal con meteo (MAE 16.54, acierto 84.5%. Más de la mitad de los días clava con menos de 10% de error y 4 de cada 5 con menos de 20%).
2. La meteo importa mucho: sin ella el lineal baja de 84.5% a 81.8% y el RF de 84.3% a 80.5%.
3. Los árboles se quedaron atrás porque 2026 tiene más potencia instalada que no habían visto. El lineal sí tira para arriba gracias a `year` y a la relación radiación-solar.
4. Para negocio: con el M1 y la meteo prevista se puede planificar la reserva bastante bien.
5. Fallos que asumo: uso la meteo observada como si fuera la prevista perfecta, Madrid no es toda España, y hay que reentrenar cada año porque se ponen más placas. Lo siguiente sería reentrenar cada mes y dar intervalos en vez de un solo número.

## Para repetirlo

- El notebook está ya corrido de arriba a abajo con salidas y gráficas.
- `python datos.py` deja 1366 filas, 0 nulos/duplicados/huecos.
- La app reentrena el M1 igual que el notebook y da el mismo MAE 16.54.
- Probado con Python 3.14 y lo de `requirements.txt`.
