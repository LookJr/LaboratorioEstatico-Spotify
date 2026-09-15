"""
test_minhastats.py
===================

Testes automatizados que validam cada função de `core/minhastats.py`
comparando o resultado com a referência de NumPy/SciPy/statistics.

Tolerância numérica documentada: 1e-9 (relativa e absoluta) para a maioria
das medidas. Diferenças nessa ordem de grandeza vêm apenas de erro de
arredondamento de ponto flutuante (float64), não de erro de fórmula.

Como rodar:
    pip install -r requirements.txt
    pytest tests/ -v
"""

import math
import random
import statistics
import sys
import os

import numpy as np
import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import minhastats as ms  # noqa: E402

TOL = 1e-9


@pytest.fixture(scope="module")
def dados_x():
    random.seed(42)
    return [random.uniform(0, 100) for _ in range(500)]


@pytest.fixture(scope="module")
def dados_y(dados_x):
    random.seed(7)
    return [xi * 2.5 + random.gauss(0, 15) for xi in dados_x]


@pytest.fixture
def dados_com_moda():
    return [1, 2, 2, 3, 3, 3, 4]


@pytest.fixture
def dados_com_outliers(dados_x):
    return dados_x + [10_000, -5_000]


# ---------------------------------------------------------------------------
# Tendência central
# ---------------------------------------------------------------------------

def test_media(dados_x):
    assert math.isclose(ms.media(dados_x), float(np.mean(dados_x)), rel_tol=TOL)


def test_mediana_impar(dados_x):
    impar = dados_x[:-1] if len(dados_x) % 2 == 0 else dados_x
    assert math.isclose(ms.mediana(impar), float(np.median(impar)), rel_tol=TOL)


def test_mediana_par(dados_x):
    par = dados_x if len(dados_x) % 2 == 0 else dados_x[:-1]
    assert math.isclose(ms.mediana(par), float(np.median(par)), rel_tol=TOL)


def test_moda(dados_com_moda):
    assert ms.moda(dados_com_moda) == statistics.multimode(dados_com_moda) or \
        set(ms.moda(dados_com_moda)) == {statistics.mode(dados_com_moda)}


def test_moda_sem_repeticao():
    assert ms.moda([1, 2, 3, 4]) == []


# ---------------------------------------------------------------------------
# Dispersão
# ---------------------------------------------------------------------------

def test_amplitude(dados_x):
    assert math.isclose(ms.amplitude(dados_x), float(np.ptp(dados_x)), rel_tol=TOL)


def test_variancia_amostral(dados_x):
    assert math.isclose(
        ms.variancia(dados_x, amostral=True), float(np.var(dados_x, ddof=1)), rel_tol=TOL
    )


def test_variancia_populacional(dados_x):
    assert math.isclose(
        ms.variancia(dados_x, amostral=False), float(np.var(dados_x, ddof=0)), rel_tol=TOL
    )


def test_desvio_padrao_amostral(dados_x):
    assert math.isclose(
        ms.desvio_padrao(dados_x, amostral=True), float(np.std(dados_x, ddof=1)), rel_tol=TOL
    )


def test_desvio_padrao_populacional(dados_x):
    assert math.isclose(
        ms.desvio_padrao(dados_x, amostral=False), float(np.std(dados_x, ddof=0)), rel_tol=TOL
    )


def test_coeficiente_variacao(dados_x):
    esperado = (np.std(dados_x, ddof=1) / np.mean(dados_x)) * 100
    assert math.isclose(ms.coeficiente_variacao(dados_x), float(esperado), rel_tol=TOL)


# ---------------------------------------------------------------------------
# Quartis / percentis / outliers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [1, 10, 25, 50, 75, 90, 99, 33.3, 90.5])
def test_percentil(dados_x, p):
    assert math.isclose(ms.percentil(dados_x, p), float(np.percentile(dados_x, p)), rel_tol=TOL)


def test_quartis(dados_x):
    q1, q2, q3 = ms.quartis(dados_x)
    assert math.isclose(q1, float(np.percentile(dados_x, 25)), rel_tol=TOL)
    assert math.isclose(q2, float(np.percentile(dados_x, 50)), rel_tol=TOL)
    assert math.isclose(q3, float(np.percentile(dados_x, 75)), rel_tol=TOL)


def test_deteccao_outliers_iqr(dados_com_outliers):
    outliers = ms.detectar_outliers_iqr(dados_com_outliers)
    assert 10_000 in outliers
    assert -5_000 in outliers


# ---------------------------------------------------------------------------
# Covariância e correlação
# ---------------------------------------------------------------------------

def test_covariancia_amostral(dados_x, dados_y):
    esperado = float(np.cov(dados_x, dados_y, ddof=1)[0][1])
    assert math.isclose(ms.covariancia(dados_x, dados_y, amostral=True), esperado, rel_tol=TOL)


def test_covariancia_populacional(dados_x, dados_y):
    esperado = float(np.cov(dados_x, dados_y, ddof=0)[0][1])
    assert math.isclose(ms.covariancia(dados_x, dados_y, amostral=False), esperado, rel_tol=TOL)


def test_correlacao_pearson(dados_x, dados_y):
    esperado, _ = scipy_stats.pearsonr(dados_x, dados_y)
    assert math.isclose(ms.correlacao_pearson(dados_x, dados_y), float(esperado), rel_tol=TOL)


# ---------------------------------------------------------------------------
# Regressão linear
# ---------------------------------------------------------------------------

def test_regressao_linear(dados_x, dados_y):
    lin = scipy_stats.linregress(dados_x, dados_y)
    b0, b1 = ms.regressao_linear(dados_x, dados_y)
    assert math.isclose(b1, float(lin.slope), rel_tol=TOL)
    assert math.isclose(b0, float(lin.intercept), rel_tol=TOL)


def test_r_quadrado(dados_x, dados_y):
    lin = scipy_stats.linregress(dados_x, dados_y)
    b0, b1 = ms.regressao_linear(dados_x, dados_y)
    r2 = ms.r_quadrado(dados_x, dados_y, b0, b1)
    assert math.isclose(r2, float(lin.rvalue) ** 2, rel_tol=TOL)


def test_prever():
    b0, b1 = 1.0, 2.0
    assert ms.prever(b0, b1, 3) == 7.0


# ---------------------------------------------------------------------------
# Casos de borda / validações de entrada
# ---------------------------------------------------------------------------

def test_media_lista_vazia_levanta_erro():
    with pytest.raises(ValueError):
        ms.media([])


def test_correlacao_variavel_constante_levanta_erro():
    with pytest.raises(ValueError):
        ms.correlacao_pearson([1, 1, 1], [1, 2, 3])


def test_regressao_x_constante_levanta_erro():
    with pytest.raises(ValueError):
        ms.regressao_linear([5, 5, 5], [1, 2, 3])
