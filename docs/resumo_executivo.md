# Resumo Executivo

**Nome completo:** [Ricardo da Silva Mariano]
**Matrícula:** [72601408]
**Grupo:** Individual

## Dataset escolhido

Spotify Tracks Dataset (Kaggle) — ~114.000 faixas musicais, 20 colunas,
125 gêneros. Fonte:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

## Módulos implementados

- ✅ Módulo 0 — Carregamento e inspeção do dataset real
- ✅ Módulo 1 — Núcleo estatístico próprio (`core/minhastats.py`),
  validado contra NumPy/SciPy com tolerância de 1e-9
  (`tests/test_minhastats.py`)
- ✅ Módulo 2 — Estatística descritiva interativa (tabelas de frequência,
  medidas de posição/dispersão, histograma/boxplot/barras/pizza,
  detecção de outliers via IQR, interpretação automática de assimetria)
- ✅ Módulo 3 — Simulação de Monte Carlo (Lei dos Grandes Números e
  Teorema Central do Limite, com parâmetros controláveis pelo usuário)
- ✅ Módulo 4 — Ajuste de distribuições teóricas (Normal + Uniforme ou
  Exponencial, com parâmetros estimados a partir dos dados)
- ✅ Módulo 5 — Correlação e regressão linear simples (mínimos quadrados
  implementados do zero, equação da reta, R², predição interativa,
  alerta de correlação ≠ causalidade)
- ✅ Módulo 6 — Relatório de descobertas (`docs/RELATORIO.md`)

## As 3 principais descobertas

1. **Energia e volume (loudness) são fortemente correlacionados** —
   faixas mais energéticas tendem a ser mixadas com volume mais alto.
2. **Danceability e valence têm correlação positiva moderada** — músicas
   mais dançantes tendem a soar mais "alegres", mas a relação é fraca o
   suficiente para não ser determinística.
3. **A popularidade das faixas não segue uma distribuição Normal** — o
   ajuste de distribuições teóricas revela uma cauda longa (poucos
   "hits" concentram a maior parte da popularidade), diferente do
   formato simétrico de uma Normal.

*(Números exatos e gráficos de suporte no `docs/RELATORIO.md`, gerados
pela própria aplicação a partir do dataset real.)*
