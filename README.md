# 🎧 Laboratório Estatístico Interativo — Spotify Tracks Dataset

Sistematização — Matemática/Estatística Computacional

## 👤 Identificação

- **Nome completo:** [Ricardo da Silva Mariano]
- **Matrícula:** [72601408]
- **Grupo:** Individual

## 📌 Descrição do projeto

Aplicação computacional (Streamlit) que carrega um dataset real de faixas
musicais do Spotify e permite ao usuário explorá-lo interativamente por
meio de:

- estatística descritiva (medidas de posição/dispersão, tabelas de
  frequência, detecção de outliers, interpretação automática de
  assimetria);
- simulação de Monte Carlo (Lei dos Grandes Números e Teorema Central do
  Limite);
- ajuste de distribuições teóricas (Normal, Uniforme, Exponencial);
- correlação e regressão linear simples com predição interativa.

O diferencial do projeto é que **todo o núcleo matemático foi implementado
do zero** em [`core/minhastats.py`](core/minhastats.py) — sem usar funções
prontas de estatística — e depois validado contra NumPy/SciPy em
[`tests/test_minhastats.py`](tests/test_minhastats.py).

## 📊 Dataset

- **Nome:** Spotify Tracks Dataset
- **Fonte original:** Kaggle —
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Tamanho:** ~114.000 faixas, 20 colunas
- **Variáveis numéricas usadas:** `popularity`, `duration_ms`,
  `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,
  `instrumentalness`, `liveness`, `valence`, `tempo` (bem mais que as 4
  exigidas)
- **Variáveis categóricas usadas:** `track_genre`, `explicit`, `mode`,
  `artists` (mais que as 2 exigidas)

> ⚠️ O dataset completo **não está versionado neste repositório** (é
> grande demais para o Git). Siga as instruções de instalação abaixo para
> baixá-lo.

## 🗂️ Estrutura do repositório

```
.
├── app/
│   └── app.py                # Aplicação Streamlit (interface + módulos)
├── core/
│   └── minhastats.py          # Núcleo estatístico implementado do zero
├── tests/
│   └── test_minhastats.py     # Testes automatizados (pytest) vs NumPy/SciPy
├── data/
│   ├── amostra_teste.csv      # Amostra sintética só para testar o código localmente
│   └── dataset.csv            # (você cria) — dataset REAL baixado do Kaggle
├── docs/
│   ├── RELATORIO.md
│   └── resumo_executivo.md
├── requirements.txt
└── README.md
```

## ⚙️ Instalação e execução (do zero)

```bash
# 1. Clonar o repositório
git clone <URL_DESTE_REPOSITORIO>
cd <pasta_do_repositorio>

# 2. Criar ambiente virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Baixar o dataset real do Kaggle e salvar como data/dataset.csv
#    (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
#    Alternativa: use a API do Kaggle
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/ --unzip
mv "data/dataset.csv" data/dataset.csv   # renomeie o CSV baixado, se necessário

# 5. Rodar os testes automatizados do núcleo estatístico
pytest tests/ -v

# 6. Rodar a aplicação
streamlit run app/app.py
```

A aplicação abre no navegador em `http://localhost:8501`. Se preferir não
mexer em `data/dataset.csv`, também é possível enviar o CSV diretamente
pela barra lateral da aplicação (botão "Carregar CSV do dataset").

## 🎬 Demonstração

- Vídeo (3–5 min): [LINK DO VÍDEO — YouTube não listado ou Google Drive]
- GIF/capturas de tela: [ADICIONAR AQUI]

## 🧪 Sobre os testes

`tests/test_minhastats.py` compara cada função de `core/minhastats.py`
com a referência de NumPy/SciPy, usando tolerância numérica de `1e-9`
(diferenças nessa ordem vêm apenas de arredondamento de ponto flutuante).
Rode com:

```bash
pytest tests/ -v
```

## 🔀 Versionamento

Este repositório deve conter histórico real de commits ao longo do
desenvolvimento (não um único commit final). Sugestão de sequência de
commits:

1. `estrutura inicial do projeto + requirements`
2. `núcleo estatístico: tendência central e dispersão`
3. `núcleo estatístico: quartis, outliers, covariância/correlação`
4. `núcleo estatístico: regressão linear e distribuições`
5. `testes automatizados vs numpy/scipy`
6. `app: módulo 0 e módulo 2 (descritiva)`
7. `app: módulo 3 (simulação Monte Carlo)`
8. `app: módulo 4 (distribuições teóricas)`
9. `app: módulo 5 (correlação e regressão)`
10. `documentação: README e RELATORIO`
