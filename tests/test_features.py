from datetime import date

import numpy as np
import pandas as pd
import pytest

from descargar_datos import tramos_anuales
from datos import validar
from features import FEAT_FULL, calendario, fila_siguiente, metricas, preparar, separar


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


def test_preparar_falla_si_falta_un_dia(serie):
    with pytest.raises(ValueError, match="un día por fila"):
        preparar(serie.drop(index=20))


def test_el_calendario_no_salta_en_anio_nuevo():
    fechas = pd.Series(pd.to_datetime(["2026-12-31", "2027-01-01"]))
    cal = calendario(fechas)
    un_dia = 2 * np.pi / 365.25  # lo máximo que cambian seno y coseno en un día normal
    assert cal.tendencia_anios.diff().iloc[1] == pytest.approx(1 / 365.25)
    assert abs(cal.dayofyear_sin.diff().iloc[1]) < 2 * un_dia
    assert abs(cal.dayofyear_cos.diff().iloc[1]) < 2 * un_dia


def test_separar_por_fecha(serie):
    train, test = separar(preparar(serie), dias_test=10)
    assert len(test) == 10
    assert train.fecha.max() < test.fecha.min()


def test_metricas():
    m = metricas([100, 200], [90, 220])
    assert m["MAE"] == pytest.approx(15)
    assert m["Acierto"] == pytest.approx(90)
    assert m["p10"] == pytest.approx(100)
    assert m["WAPE"] == pytest.approx(10)


def test_validar_acepta_datos_buenos(serie):
    validar(serie)


@pytest.mark.parametrize("estropear, problema", [
    (lambda d: d.drop(index=10), "días que faltan"),
    (lambda d: pd.concat([d, d.iloc[[5]]]), "fechas duplicadas"),
    (lambda d: d.assign(solar_gwh=d.solar_gwh.where(d.index != 3)), "nulos"),
    (lambda d: d.assign(solar_gwh=d.solar_gwh.where(d.index != 3, -1.0)), "solar <= 0"),
    (lambda d: d.assign(radiacion_MJm2=d.radiacion_MJm2.where(d.index != 3, 50.0)), "radiación fuera de 0-40"),
])
def test_validar_rechaza_datos_malos(serie, estropear, problema):
    with pytest.raises(ValueError, match=problema):
        validar(estropear(serie))


def test_tramos_anuales():
    tramos = list(tramos_anuales(date(2023, 6, 1), date(2025, 2, 3)))
    assert tramos == [("2023-06-01", "2023-12-31"), ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-02-03")]
