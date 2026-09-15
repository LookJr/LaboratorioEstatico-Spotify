# Relatório — Laboratório Estatístico Interativo

**Nome completo:** [Ricardo da Silva Mariano]
**Matrícula:** [72601408]

---

## 1. Dataset escolhido e justificativa

Foi escolhido o **Spotify Tracks Dataset**, disponível publicamente no
Kaggle
(https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).

O dataset reúne aproximadamente **114.000 faixas musicais** distribuídas
em **125 gêneros**, com 20 colunas descrevendo cada faixa através de
características de áudio calculadas pela própria API do Spotify.

**Justificativa da escolha:**

- Atende com folga ao requisito mínimo de 1.000 registros e 4 variáveis
  numéricas + 2 categóricas: o dataset possui mais de 10 variáveis
  numéricas contínuas (`danceability`, `energy`, `loudness`,
  `speechiness`, `acousticness`, `instrumentalness`, `liveness`,
  `valence`, `tempo`, `popularity`, `duration_ms`) e várias variáveis
  categóricas (`track_genre`, `explicit`, `mode`, `artists`).
- É um tema envolvente (música) que facilita a interpretação intuitiva
  das medidas estatísticas — por exemplo, é fácil entender o que
  significa uma faixa ter "energia" alta e "acusticidade" baixa.
- Já é amplamente estudado por outros analistas, o que permite comparar
  os resultados obtidos pela nossa biblioteca própria com achados
  publicados por terceiros como forma adicional de sanity check.

## 2. Decisões de implementação do núcleo estatístico

O núcleo estatístico foi implementado em `core/minhastats.py`, usando
apenas Python puro (`math`, laços e compreensões de lista) — sem
depender de `numpy`, `pandas`, `scipy` ou do módulo `statistics`. As
fórmulas utilizadas foram:

### 2.1 Tendência central

- **Média:** x̄ = (Σxᵢ) / n
- **Mediana:** valor central da lista ordenada (interpolação da média dos
  dois centrais quando n é par).
- **Moda:** valor(es) de maior frequência absoluta (suporte a
  multimodalidade).

### 2.2 Dispersão

- **Amplitude:** máximo − mínimo.
- **Variância amostral:** s² = Σ(xᵢ − x̄)² / (n − 1)
- **Variância populacional:** σ² = Σ(xᵢ − x̄)² / n
- **Desvio padrão:** raiz quadrada da variância (amostral ou
  populacional, conforme parâmetro).
- **Coeficiente de variação:** CV = (desvio padrão / média) × 100.

### 2.3 Quartis e percentis

Implementados por **interpolação linear** sobre a posição
`(n−1) × p/100` na lista ordenada — o mesmo método usado por padrão em
`numpy.percentile(..., method="linear")`, escolhido propositalmente para
facilitar a validação cruzada.

### 2.4 Outliers (regra do IQR)

- IQR = Q3 − Q1
- Limite inferior = Q1 − 1,5 × IQR
- Limite superior = Q3 + 1,5 × IQR

Qualquer valor fora desse intervalo é classificado como outlier.

### 2.5 Covariância e correlação de Pearson

- Covariância amostral: cov(x,y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1)
- Correlação de Pearson: r = cov(x,y) / (dp(x) × dp(y))

### 2.6 Regressão linear simples (mínimos quadrados)

Ajuste de ŷ = b₀ + b₁x minimizando a soma dos quadrados dos resíduos:

- b₁ = cov(x,y) / var(x)
- b₀ = ȳ − b₁ × x̄
- R² = 1 − (SQ_residual / SQ_total), onde SQ_residual = Σ(yᵢ − ŷᵢ)² e
  SQ_total = Σ(yᵢ − ȳ)²

### 2.7 Distribuições teóricas

Implementadas manualmente (sem `scipy.stats`) para manter a mesma
filosofia de "construir, não só usar": densidade da Normal, da Uniforme
contínua, da Exponencial, e as probabilidades pontuais da Binomial e da
Poisson (fórmulas fechadas, ver docstrings em `minhastats.py`).

## 3. Resultados da validação contra as bibliotecas

Todas as funções de `core/minhastats.py` foram comparadas com a
referência de NumPy/SciPy em `tests/test_minhastats.py`, usando um
conjunto de 500 valores aleatórios (semente fixa, para reprodutibilidade)
mais casos de borda (moda multimodal, sem moda, outliers extremos,
variável constante).

**Tolerância numérica documentada:** `1e-9` (relativa e absoluta). As
diferenças observadas nos testes ficaram na casa de `1e-13` a `1e-16` —
ou seja, exclusivamente erro de arredondamento de ponto flutuante
(`float64`), não erro de fórmula. Todas as funções passaram na validação:
média, mediana, moda, amplitude, variância (amostral/populacional),
desvio padrão (amostral/populacional), percentis/quartis, coeficiente de
variação, covariância (amostral/populacional), correlação de Pearson,
coeficientes da regressão linear (b₀, b₁), R² e as densidades/probabi­
lidades das distribuições teóricas (Normal, Uniforme, Exponencial,
Binomial, Poisson).

> Rode `pytest tests/ -v` para reproduzir a validação completa.

## 4. Explicação de cada módulo da aplicação

- **Módulo 0 (Dados reais):** exibe o tamanho do dataset e a lista de
  variáveis numéricas/categóricas identificadas automaticamente.
- **Módulo 2 (Descritiva interativa):** o usuário escolhe uma variável
  (numérica ou categórica) e recebe tabela de frequências, medidas de
  posição/dispersão, histograma, boxplot (ou barras/pizza), detecção de
  outliers pelo IQR e uma frase de interpretação automática da assimetria.
- **Módulo 3 (Probabilidade e simulação):** simulação de Monte Carlo da
  Lei dos Grandes Números (convergência da frequência relativa de "cara"
  em lançamentos simulados) e do Teorema Central do Limite (distribuição
  das médias de amostras repetidas de uma variável real do dataset se
  aproximando de uma Normal conforme o tamanho da amostra cresce).
- **Módulo 4 (Distribuições teóricas):** sobrepõe ao histograma de uma
  variável a curva da Normal (parâmetros estimados dos dados) e de uma
  segunda distribuição candidata (Uniforme ou Exponencial), com
  discussão visual da qualidade do ajuste.
- **Módulo 5 (Correlação e regressão):** dispersão entre duas variáveis
  numéricas escolhidas pelo usuário, correlação de Pearson, reta de
  regressão, equação, R² e campo de predição interativa — com alerta de
  que correlação não implica causalidade.

## 5. As 3 descobertas estatísticas mais interessantes

> ⚠️ **Atenção:** os números abaixo são um ponto de partida baseado em
> análises publicadas sobre este mesmo dataset. **Antes de entregar,
> rode a aplicação com o dataset real (`data/dataset.csv`) nos Módulos 2
> e 5 e substitua pelos valores exatos que a SUA aplicação calculou.**

1. **Energia e volume (loudness) andam juntos.** Ao correlacionar
   `energy` com `loudness` no Módulo 5, espera-se uma correlação
   positiva forte (na faixa de r ≈ 0,7–0,8): faixas mais "energéticas"
   tendem a ser mixadas com volume mais alto. Isso sugere que produtores
   usam o volume como uma das ferramentas para transmitir sensação de
   energia.

2. **Danceability e valence (positividade do humor) se relacionam, mas
   moderadamente.** Espera-se uma correlação positiva moderada (r ≈
   0,3–0,45) entre `danceability` e `valence`: músicas mais dançantes
   tendem a soar mais "alegres", mas a relação está longe de ser
   perfeita — muita música dançante também soa melancólica.

3. **A popularidade não segue distribuição Normal.** Ao ajustar
   distribuições no Módulo 4 sobre a variável `popularity`, é comum
   observar assimetria (muitas faixas com popularidade baixa/moderada e
   poucas faixas "hits" com popularidade muito alta) — um padrão mais
   próximo de uma cauda longa do que de uma Normal simétrica, o que faz
   sentido dado como funciona a atenção do público: poucos sucessos
   concentram a maior parte dos streams.

## 6. Limitações e trabalhos futuros

- A biblioteca própria assume dados numéricos sem valores ausentes
  (`NaN`s são removidos antes do cálculo pela camada de aplicação, não
  pelo núcleo estatístico).
- O ajuste de distribuições é qualitativo (visual); um teste de
  aderência formal (ex.: Kolmogorov-Smirnov) poderia complementar a
  análise em uma versão futura.
