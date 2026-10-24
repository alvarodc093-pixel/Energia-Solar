from datetime import date

import numpy as np
import pandas as pd
import pytest

from descargar_datos import tramos_anuales
from features import FEAT_FULL, fila_siguiente, metricas, preparar, separar


@pytest.fixture
def serie():
    fechas = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "fecha": fechas,
        "solar_gwh": np.arange(60, dtype=float) + 100,
        "radiacion_MJm2": 15.0,
        "temp_media_Madrid": 20.0,
    })


def test_los_retardos_solo_miran_al_pasado(serie):
    dfm = preparar(serie)
    fila = dfm.iloc[5]
    previos = serie[serie.fecha < fila.fecha].solar_gwh
    assert fila.lag1 == previos.iloc[-1]
    assert fila.lag7 == previos.iloc[-7]
    assert fila.roll7 == pytest.approx(previos.iloc[-7:].mean())
    assert fila.roll30 == pytest.approx(previos.iloc[-30:].mean())


def test_preparar_quita_los_dias_sin_historia(serie):
    dfm = preparar(serie)
    assert len(dfm) == 60 - 30
    assert not dfm[FEAT_FULL].isna().any().any()


def test_fila_siguiente_coincide_con_preparar(serie):
    """La fila que usa la app para predecir es la misma que tendría ese día en el histórico."""
    historia = serie.iloc[:-1]
    esperada = preparar(serie).iloc[-1]
    fila = fila_siguiente(historia, 15.0, 20.0).iloc[0]
    for col in FEAT_FULL:
        assert fila[col] == pytest.approx(esperada[col]), col


def test_separar_por_fecha(serie):
    train, test = separar(preparar(serie), dias_test=10)
    assert len(test) == 10
    assert train.fecha.max() < test.fecha.min()


def test_metricas():
    m = metricas([100, 200], [90, 220])
    assert m["MAE"] == pytest.approx(15)
    assert m["Acierto"] == pytest.approx(90)
    assert m["p10"] == pytest.approx(100)


def test_tramos_anuales():
    tramos = list(tramos_anuales(date(2023, 6, 1), date(2025, 2, 3)))
    assert tramos == [("2023-06-01", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-02-03")]
