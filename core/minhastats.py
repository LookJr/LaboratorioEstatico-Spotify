"""
minhastats.py
=============

Biblioteca estatística implementada do zero (sem usar funções prontas de
estatística de terceiros) para o Laboratório Estatístico Interativo.

Regra de ouro do projeto: NumPy/Pandas só podem ser usados para carregar e
manipular dados e para VALIDAR os resultados (ver tests/test_minhastats.py).
Todas as medidas exibidas ao usuário na aplicação (app/app.py) vêm das
funções implementadas aqui, usando apenas Python puro (listas, laços,
`math.sqrt`, `math.floor`, etc.).

Convenções:
- Todas as funções aceitam listas/sequências de números (int ou float).
- Funções que têm variante amostral/populacional recebem o parâmetro
  `amostral: bool = True` (amostral é o padrão, pois é o caso mais comum
  quando trabalhamos com uma amostra de um dataset maior).
- Nenhuma função aqui depende de numpy, pandas, scipy ou do módulo
  `statistics` da biblioteca padrão.
"""

from __future__ import annotations

import math
from typing import Sequence, List, Tuple, Union

Numero = Union[int, float]


# ---------------------------------------------------------------------------
# Funções auxiliares internas
# ---------------------------------------------------------------------------

def _validar_nao_vazio(dados: Sequence[Numero], nome: str = "dados") -> None:
    if dados is None or len(dados) == 0:
        raise ValueError(f"'{nome}' não pode ser vazio.")


def _validar_mesmo_tamanho(x: Sequence[Numero], y: Sequence[Numero]) -> None:
    if len(x) != len(y):
        raise ValueError("As duas sequências precisam ter o mesmo tamanho.")
    if len(x) < 2:
        raise ValueError("São necessários pelo menos 2 pares (x, y).")


# ---------------------------------------------------------------------------
# Módulo 1 — Medidas de tendência central
# ---------------------------------------------------------------------------

def media(dados: Sequence[Numero]) -> float:
    """Média aritmética: soma dos valores dividida pela quantidade.

    Fórmula: x̄ = (Σxᵢ) / n
    """
    _validar_nao_vazio(dados)
    soma = 0.0
    n = 0
    for valor in dados:
        soma += valor
        n += 1
    return soma / n


def mediana(dados: Sequence[Numero]) -> float:
    """Mediana: valor central da lista ordenada.

    Se n for ímpar, é o elemento do meio. Se n for par, é a média dos dois
    elementos centrais.
    """
    _validar_nao_vazio(dados)
    ordenado = sorted(dados)
    n = len(ordenado)
    meio = n // 2
    if n % 2 == 1:
        return float(ordenado[meio])
    return (ordenado[meio - 1] + ordenado[meio]) / 2.0


def moda(dados: Sequence[Numero]) -> List[Numero]:
    """Moda: valor(es) mais frequente(s).

    Retorna uma lista porque a moda pode ser multimodal (empate entre dois
    ou mais valores). Se nenhum valor se repete, retorna a lista vazia.
    """
    _validar_nao_vazio(dados)
    frequencias = {}
    for valor in dados:
        frequencias[valor] = frequencias.get(valor, 0) + 1

    freq_maxima = max(frequencias.values())
    if freq_maxima == 1:
        return []  # nenhum valor se repete -> sem moda
    modas = [valor for valor, freq in frequencias.items() if freq == freq_maxima]
    return sorted(modas)


# ---------------------------------------------------------------------------
# Módulo 1 — Medidas de dispersão
# ---------------------------------------------------------------------------

def amplitude(dados: Sequence[Numero]) -> float:
    """Amplitude: diferença entre o maior e o menor valor."""
    _validar_nao_vazio(dados)
    return float(max(dados) - min(dados))


def variancia(dados: Sequence[Numero], amostral: bool = True) -> float:
    """Variância.

    Amostral:    s² = Σ(xᵢ - x̄)² / (n - 1)
    Populacional: σ² = Σ(xᵢ - x̄)² / n
    """
    _validar_nao_vazio(dados)
    n = len(dados)
    if amostral and n < 2:
        raise ValueError("Variância amostral requer pelo menos 2 valores.")
    m = media(dados)
    soma_quadrados = sum((x - m) ** 2 for x in dados)
    denominador = (n - 1) if amostral else n
    return soma_quadrados / denominador


def desvio_padrao(dados: Sequence[Numero], amostral: bool = True) -> float:
    """Desvio padrão = raiz quadrada da variância."""
    return math.sqrt(variancia(dados, amostral=amostral))


def coeficiente_variacao(dados: Sequence[Numero], amostral: bool = True) -> float:
    """Coeficiente de variação (%): CV = (desvio padrão / média) * 100.

    Útil para comparar a dispersão relativa de variáveis com escalas
    diferentes (ex.: comparar a dispersão de 'duração da música' com a de
    'popularidade').
    """
    m = media(dados)
    if m == 0:
        raise ValueError("Coeficiente de variação indefinido quando a média é zero.")
    dp = desvio_padrao(dados, amostral=amostral)
    return (dp / abs(m)) * 100.0


# ---------------------------------------------------------------------------
# Módulo 1 — Quartis e percentis
# ---------------------------------------------------------------------------

def percentil(dados: Sequence[Numero], p: float) -> float:
    """Percentil p (0 <= p <= 100) usando interpolação linear.

    Este é o mesmo método usado por padrão em `numpy.percentile`
    (interpolação linear sobre a posição (n-1)*p/100 na lista ordenada),
    o que facilita a validação cruzada no arquivo de testes.
    """
    _validar_nao_vazio(dados)
    if not (0 <= p <= 100):
        raise ValueError("p deve estar entre 0 e 100.")

    ordenado = sorted(dados)
    n = len(ordenado)
    if n == 1:
        return float(ordenado[0])

    posicao = (n - 1) * (p / 100.0)
    indice_inferior = math.floor(posicao)
    indice_superior = math.ceil(posicao)

    if indice_inferior == indice_superior:
        return float(ordenado[indice_inferior])

    fracao = posicao - indice_inferior
    valor = ordenado[indice_inferior] + fracao * (
        ordenado[indice_superior] - ordenado[indice_inferior]
    )
    return float(valor)


def quartis(dados: Sequence[Numero]) -> Tuple[float, float, float]:
    """Retorna (Q1, Q2, Q3) — primeiro quartil, mediana e terceiro quartil."""
    q1 = percentil(dados, 25)
    q2 = percentil(dados, 50)
    q3 = percentil(dados, 75)
    return q1, q2, q3


def limites_outliers_iqr(dados: Sequence[Numero]) -> Tuple[float, float]:
    """Limites inferior/superior para detecção de outliers pela regra do IQR.

    IQR = Q3 - Q1
    Limite inferior = Q1 - 1.5 * IQR
    Limite superior = Q3 + 1.5 * IQR
    """
    q1, _, q3 = quartis(dados)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return limite_inferior, limite_superior


def detectar_outliers_iqr(dados: Sequence[Numero]) -> List[Numero]:
    """Retorna a lista de valores considerados outliers pela regra do IQR."""
    limite_inferior, limite_superior = limites_outliers_iqr(dados)
    return [x for x in dados if x < limite_inferior or x > limite_superior]


# ---------------------------------------------------------------------------
# Módulo 1 — Covariância e correlação
# ---------------------------------------------------------------------------

def covariancia(x: Sequence[Numero], y: Sequence[Numero], amostral: bool = True) -> float:
    """Covariância entre duas variáveis.

    Amostral:     cov(x,y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1)
    Populacional: cov(x,y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / n
    """
    _validar_mesmo_tamanho(x, y)
    n = len(x)
    mx = media(x)
    my = media(y)
    soma = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denominador = (n - 1) if amostral else n
    return soma / denominador


def correlacao_pearson(x: Sequence[Numero], y: Sequence[Numero]) -> float:
    """Coeficiente de correlação de Pearson.

    r = cov(x,y) / (desvio_padrao(x) * desvio_padrao(y))

    Como o (n-1) do numerador e do denominador se cancelam, o resultado é
    idêntico usando a versão amostral ou populacional — aqui usamos a
    amostral por convenção.
    """
    cov = covariancia(x, y, amostral=True)
    dpx = desvio_padrao(x, amostral=True)
    dpy = desvio_padrao(y, amostral=True)
    if dpx == 0 or dpy == 0:
        raise ValueError("Correlação indefinida quando uma das variáveis é constante.")
    return cov / (dpx * dpy)


# ---------------------------------------------------------------------------
# Módulo 5 — Regressão linear simples (mínimos quadrados)
# ---------------------------------------------------------------------------

def regressao_linear(x: Sequence[Numero], y: Sequence[Numero]) -> Tuple[float, float]:
    """Regressão linear simples pelo método dos mínimos quadrados.

    Ajusta y = b0 + b1*x minimizando a soma dos quadrados dos resíduos.

    b1 = cov(x, y) / var(x)
    b0 = ȳ - b1 * x̄

    Retorna (b0, b1) = (intercepto, inclinação).
    """
    _validar_mesmo_tamanho(x, y)
    var_x = variancia(x, amostral=True)
    if var_x == 0:
        raise ValueError("Regressão indefinida quando x é constante (variância zero).")
    cov_xy = covariancia(x, y, amostral=True)
    b1 = cov_xy / var_x
    b0 = media(y) - b1 * media(x)
    return b0, b1


def prever(b0: float, b1: float, x_novo: Numero) -> float:
    """Aplica a reta ajustada (b0 + b1*x) a um novo valor de x."""
    return b0 + b1 * x_novo


def r_quadrado(x: Sequence[Numero], y: Sequence[Numero], b0: float, b1: float) -> float:
    """Coeficiente de determinação R².

    R² = 1 - (SQ_residual / SQ_total)
    SQ_residual = Σ(yᵢ - ŷᵢ)²
    SQ_total    = Σ(yᵢ - ȳ)²
    """
    _validar_mesmo_tamanho(x, y)
    my = media(y)
    sq_total = sum((yi - my) ** 2 for yi in y)
    if sq_total == 0:
        raise ValueError("R² indefinido quando y é constante (SQ total zero).")
    sq_residual = sum((yi - prever(b0, b1, xi)) ** 2 for xi, yi in zip(x, y))
    return 1.0 - (sq_residual / sq_total)


# ---------------------------------------------------------------------------
# Módulo 2 — Tabela de frequências
# ---------------------------------------------------------------------------

def tabela_frequencias_categorica(dados: Sequence) -> List[Tuple[object, int, float]]:
    """Tabela de frequências para variável categórica.

    Retorna lista de tuplas (categoria, frequência absoluta, frequência
    relativa em %), ordenada da mais para a menos frequente.
    """
    _validar_nao_vazio(dados)
    n = len(dados)
    contagem = {}
    for valor in dados:
        contagem[valor] = contagem.get(valor, 0) + 1
    tabela = [(cat, freq, (freq / n) * 100.0) for cat, freq in contagem.items()]
    tabela.sort(key=lambda linha: linha[1], reverse=True)
    return tabela


def tabela_frequencias_por_classes(
    dados: Sequence[Numero], n_classes: int = 10
) -> List[Tuple[float, float, int, float]]:
    """Tabela de frequências em classes para variável contínua (regra de
    Sturges simplificada — o número de classes é um parâmetro para permitir
    controle interativo na aplicação).

    Retorna lista de tuplas (limite_inferior, limite_superior, frequência
    absoluta, frequência relativa em %).
    """
    _validar_nao_vazio(dados)
    if n_classes < 1:
        raise ValueError("n_classes deve ser >= 1.")

    n = len(dados)
    minimo = min(dados)
    maximo = max(dados)
    largura = (maximo - minimo) / n_classes if maximo > minimo else 1.0

    limites = [minimo + i * largura for i in range(n_classes + 1)]
    contagens = [0] * n_classes

    for valor in dados:
        if valor == maximo:
            indice_classe = n_classes - 1  # inclui o valor máximo na última classe
        else:
            indice_classe = int((valor - minimo) / largura)
            indice_classe = min(max(indice_classe, 0), n_classes - 1)
        contagens[indice_classe] += 1

    tabela = []
    for i in range(n_classes):
        freq = contagens[i]
        tabela.append((limites[i], limites[i + 1], freq, (freq / n) * 100.0))
    return tabela


# ---------------------------------------------------------------------------
# Módulo 2 — Assimetria (para a interpretação textual automática)
# ---------------------------------------------------------------------------

def assimetria_pearson(dados: Sequence[Numero]) -> float:
    """Coeficiente de assimetria (skewness) de Pearson (2º coeficiente):

    As = 3 * (média - mediana) / desvio_padrao

    Interpretação prática usada no relatório automático:
    - As > 0.15  -> assimetria positiva (cauda à direita)
    - As < -0.15 -> assimetria negativa (cauda à esquerda)
    - caso contrário -> aproximadamente simétrica
    """
    m = media(dados)
    med = mediana(dados)
    dp = desvio_padrao(dados, amostral=True)
    if dp == 0:
        return 0.0
    return 3 * (m - med) / dp


def interpretar_assimetria(coef: float) -> str:
    if coef > 0.15:
        return "distribuição com assimetria positiva (cauda mais longa à direita)"
    if coef < -0.15:
        return "distribuição com assimetria negativa (cauda mais longa à esquerda)"
    return "distribuição aproximadamente simétrica"


# ---------------------------------------------------------------------------
# Módulo 4 — Distribuições teóricas (funções de densidade/probabilidade)
# ---------------------------------------------------------------------------
# Implementadas manualmente (sem scipy.stats) para manter a "regra de ouro"
# também na parte de distribuições: só usamos bibliotecas prontas para
# CARREGAR dados ou VALIDAR, nunca para gerar o número final mostrado ao
# usuário.

def pdf_normal(x: Numero, mu: float, sigma: float) -> float:
    """Densidade da Normal(mu, sigma) no ponto x.

    f(x) = (1 / (sigma * sqrt(2π))) * exp(-(x-mu)² / (2*sigma²))
    """
    if sigma <= 0:
        raise ValueError("sigma deve ser positivo.")
    coef = 1.0 / (sigma * math.sqrt(2 * math.pi))
    expoente = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coef * math.exp(expoente)


def pdf_uniforme(x: Numero, a: float, b: float) -> float:
    """Densidade da Uniforme contínua no intervalo [a, b]."""
    if b <= a:
        raise ValueError("b deve ser maior que a.")
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def pdf_exponencial(x: Numero, lambd: float) -> float:
    """Densidade da Exponencial(lambda) no ponto x (x >= 0).

    f(x) = lambda * exp(-lambda * x)
    """
    if lambd <= 0:
        raise ValueError("lambda deve ser positivo.")
    if x < 0:
        return 0.0
    return lambd * math.exp(-lambd * x)


def _combinacao(n: int, k: int) -> int:
    """Combinação C(n, k) = n! / (k! (n-k)!), calculada de forma estável."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    resultado = 1
    for i in range(k):
        resultado = resultado * (n - i) // (i + 1)
    return resultado


def pmf_binomial(k: int, n: int, p: float) -> float:
    """Probabilidade P(X = k) da Binomial(n, p).

    P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    """
    if not (0 <= p <= 1):
        raise ValueError("p deve estar entre 0 e 1.")
    return _combinacao(n, k) * (p ** k) * ((1 - p) ** (n - k))


def pmf_poisson(k: int, lambd: float) -> float:
    """Probabilidade P(X = k) da Poisson(lambda).

    P(X=k) = (lambda^k * e^-lambda) / k!
    """
    if lambd < 0:
        raise ValueError("lambda deve ser >= 0.")
    return (lambd ** k) * math.exp(-lambd) / math.factorial(k)
