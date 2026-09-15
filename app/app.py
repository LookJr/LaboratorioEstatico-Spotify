"""
app.py — Laboratório Estatístico Interativo
============================================

Aplicação Streamlit que carrega o dataset "Spotify Tracks Dataset" (Kaggle)
e permite explorá-lo através de estatística descritiva, probabilidade,
simulação e regressão.

IMPORTANTE: toda medida numérica exibida ao usuário (média, desvio padrão,
correlação, coeficientes da regressão etc.) é calculada pela biblioteca
própria em core/minhastats.py — NÃO por numpy/pandas/scipy. Essas
bibliotecas aqui só carregam/manipulam os dados e desenham os gráficos.

Como rodar:
    pip install -r requirements.txt
    streamlit run app/app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
import minhastats as ms  # noqa: E402

st.set_page_config(page_title="Laboratório Estatístico Interativo", layout="wide")

CAMINHO_DADOS_PADRAO = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


# ---------------------------------------------------------------------------
# Módulo 0 — Carregamento dos dados
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    return df


def identificar_colunas(df: pd.DataFrame):
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numericas, categoricas


st.sidebar.title("🎧 Laboratório Estatístico")
st.sidebar.markdown(
    "**Dataset:** Spotify Tracks Dataset (Kaggle)\n\n"
    "Envie o CSV baixado do Kaggle ou use o arquivo em `data/dataset.csv`."
)

arquivo_enviado = st.sidebar.file_uploader("Carregar CSV do dataset", type=["csv"])

if arquivo_enviado is not None:
    df = pd.read_csv(arquivo_enviado)
elif os.path.exists(CAMINHO_DADOS_PADRAO):
    df = carregar_dados(CAMINHO_DADOS_PADRAO)
else:
    st.error(
        "Nenhum dataset encontrado. Baixe o dataset no Kaggle e coloque em "
        "`data/dataset.csv`, ou envie um CSV pela barra lateral."
    )
    st.stop()

# remove colunas totalmente vazias e a coluna de índice do CSV do Kaggle, se existir
df = df.dropna(axis=1, how="all")
if df.columns[0].lower() in ("unnamed: 0", "index"):
    df = df.drop(columns=[df.columns[0]])

numericas, categoricas = identificar_colunas(df)

modulo = st.sidebar.radio(
    "Navegar pelos módulos",
    [
        "0️⃣ Dados reais",
        "2️⃣ Estatística descritiva",
        "3️⃣ Probabilidade e simulação",
        "4️⃣ Distribuições teóricas",
        "5️⃣ Correlação e regressão",
        "6️⃣ Descobertas",
    ],
)

st.title("Laboratório Estatístico Interativo")

# ---------------------------------------------------------------------------
# MÓDULO 0
# ---------------------------------------------------------------------------
if modulo.startswith("0"):
    st.header("Módulo 0 — Dados reais")
    st.write(f"**Linhas:** {df.shape[0]:,} &nbsp;&nbsp; **Colunas:** {df.shape[1]}")
    st.write(f"**Variáveis numéricas ({len(numericas)}):** {', '.join(numericas)}")
    st.write(f"**Variáveis categóricas ({len(categoricas)}):** {', '.join(categoricas)}")
    st.dataframe(df.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# MÓDULO 2 — Estatística descritiva interativa
# ---------------------------------------------------------------------------
elif modulo.startswith("2"):
    st.header("Módulo 2 — Estatística descritiva interativa")

    tipo_variavel = st.radio("Tipo de variável", ["Numérica", "Categórica"], horizontal=True)

    if tipo_variavel == "Numérica":
        coluna = st.selectbox("Escolha uma variável numérica", numericas)
        dados = df[coluna].dropna().tolist()

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("Medidas (calculadas por minhastats.py)")
            st.metric("Média", f"{ms.media(dados):.4f}")
            st.metric("Mediana", f"{ms.mediana(dados):.4f}")
            moda_calc = ms.moda(dados)
            st.metric("Moda", ", ".join(f"{m:.2f}" for m in moda_calc[:5]) if moda_calc else "sem moda")
            st.metric("Amplitude", f"{ms.amplitude(dados):.4f}")
            st.metric("Variância (amostral)", f"{ms.variancia(dados):.4f}")
            st.metric("Desvio padrão (amostral)", f"{ms.desvio_padrao(dados):.4f}")
            st.metric("Coeficiente de variação", f"{ms.coeficiente_variacao(dados):.2f}%")
            q1, q2, q3 = ms.quartis(dados)
            st.metric("Q1 / Q2 / Q3", f"{q1:.2f} / {q2:.2f} / {q3:.2f}")

        with col_b:
            st.subheader("Gráficos")
            n_classes = st.slider("Número de classes do histograma", 5, 50, 20)
            fig, ax = plt.subplots()
            ax.hist(dados, bins=n_classes, color="#1DB954", edgecolor="black")
            ax.set_title(f"Histograma — {coluna}")
            ax.set_xlabel(coluna)
            ax.set_ylabel("Frequência")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(dados, vert=False)
            ax2.set_title(f"Boxplot — {coluna}")
            st.pyplot(fig2)

        st.subheader("Tabela de frequências por classes")
        tabela = ms.tabela_frequencias_por_classes(dados, n_classes=n_classes)
        df_tabela = pd.DataFrame(
            tabela, columns=["Limite inferior", "Limite superior", "Freq. absoluta", "Freq. relativa (%)"]
        )
        st.dataframe(df_tabela, use_container_width=True)

        st.subheader("Detecção de outliers (regra do IQR)")
        outliers = ms.detectar_outliers_iqr(dados)
        lim_inf, lim_sup = ms.limites_outliers_iqr(dados)
        st.write(f"Limites: [{lim_inf:.2f}, {lim_sup:.2f}] — **{len(outliers)}** outlier(s) detectado(s) "
                 f"({len(outliers)/len(dados)*100:.2f}% dos dados).")

        st.subheader("Interpretação automática")
        coef_assimetria = ms.assimetria_pearson(dados)
        st.info(
            f"A variável **{coluna}** apresenta {ms.interpretar_assimetria(coef_assimetria)} "
            f"(coeficiente de assimetria de Pearson = {coef_assimetria:.3f})."
        )

    else:
        coluna = st.selectbox("Escolha uma variável categórica", categoricas)
        dados_cat = df[coluna].dropna().astype(str).tolist()
        tabela = ms.tabela_frequencias_categorica(dados_cat)[:20]  # top 20 categorias

        df_tabela = pd.DataFrame(tabela, columns=["Categoria", "Freq. absoluta", "Freq. relativa (%)"])
        st.dataframe(df_tabela, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots()
            ax.bar(df_tabela["Categoria"].astype(str), df_tabela["Freq. absoluta"], color="#1DB954")
            ax.set_title(f"Barras — {coluna} (top 20)")
            plt.xticks(rotation=75, ha="right")
            st.pyplot(fig)
        with col_b:
            top_n = df_tabela.head(8)
            fig2, ax2 = plt.subplots()
            ax2.pie(top_n["Freq. absoluta"], labels=top_n["Categoria"], autopct="%1.1f%%")
            ax2.set_title(f"Pizza — {coluna} (top 8)")
            st.pyplot(fig2)

# ---------------------------------------------------------------------------
# MÓDULO 3 — Probabilidade e simulação (Monte Carlo)
# ---------------------------------------------------------------------------
elif modulo.startswith("3"):
    st.header("Módulo 3 — Probabilidade e simulação")

    aba_lgn, aba_tcl = st.tabs(["Lei dos Grandes Números", "Teorema Central do Limite"])

    with aba_lgn:
        st.subheader("Lei dos Grandes Números — lançamento de moeda simulado")
        st.caption(
            "Conforme o número de lançamentos aumenta, a frequência relativa de "
            "'cara' converge para a probabilidade teórica (0.5)."
        )
        n_lancamentos = st.slider("Número de lançamentos", 10, 20_000, 2_000, key="lgn_n")
        p_cara = st.slider("Probabilidade teórica de 'cara'", 0.05, 0.95, 0.5, key="lgn_p")
        semente = st.number_input("Semente aleatória", value=42, step=1, key="lgn_seed")

        rng = np.random.default_rng(int(semente))
        lancamentos = rng.random(n_lancamentos) < p_cara  # gerar aleatoriedade com numpy (permitido)

        # a MÉDIA MÓVEL (frequência relativa acumulada) é calculada com minhastats,
        # ponto a ponto, sem usar numpy.cumsum/mean:
        frequencias_acumuladas = []
        soma = 0
        for i, resultado in enumerate(lancamentos, start=1):
            soma += int(resultado)
            frequencias_acumuladas.append(soma / i)

        fig, ax = plt.subplots()
        ax.plot(frequencias_acumuladas, color="#1DB954", linewidth=1)
        ax.axhline(p_cara, color="red", linestyle="--", label=f"Probabilidade teórica ({p_cara})")
        ax.set_xlabel("Número de lançamentos")
        ax.set_ylabel("Frequência relativa acumulada de 'cara'")
        ax.set_title("Convergência da frequência relativa (Lei dos Grandes Números)")
        ax.legend()
        st.pyplot(fig)

        st.write(
            f"Frequência relativa final: **{frequencias_acumuladas[-1]:.4f}** "
            f"(teórica: {p_cara}) — diferença absoluta de "
            f"{abs(frequencias_acumuladas[-1] - p_cara):.4f}."
        )

    with aba_tcl:
        st.subheader("Teorema Central do Limite — médias amostrais de uma variável do dataset")
        coluna_tcl = st.selectbox("Variável do dataset a amostrar", numericas, key="tcl_col")
        populacao = df[coluna_tcl].dropna().tolist()

        tamanho_amostra = st.slider("Tamanho de cada amostra (n)", 2, 500, 30, key="tcl_n")
        n_amostras = st.slider("Número de amostras repetidas", 100, 10_000, 2_000, key="tcl_reps")
        semente_tcl = st.number_input("Semente aleatória", value=123, step=1, key="tcl_seed")

        rng = np.random.default_rng(int(semente_tcl))
        medias_amostrais = []
        for _ in range(n_amostras):
            amostra = rng.choice(populacao, size=tamanho_amostra, replace=True)
            medias_amostrais.append(ms.media(amostra.tolist()))  # média calculada por minhastats

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots()
            ax.hist(populacao, bins=30, color="#888", alpha=0.6)
            ax.set_title(f"Distribuição original — {coluna_tcl}")
            st.pyplot(fig)
        with col_b:
            fig2, ax2 = plt.subplots()
            ax2.hist(medias_amostrais, bins=30, color="#1DB954", edgecolor="black")
            ax2.set_title(f"Distribuição das médias amostrais (n={tamanho_amostra})")
            st.pyplot(fig2)

        media_das_medias = ms.media(medias_amostrais)
        dp_das_medias = ms.desvio_padrao(medias_amostrais)
        erro_padrao_teorico = ms.desvio_padrao(populacao) / (tamanho_amostra ** 0.5)
        st.success(
            f"Média das médias amostrais: **{media_das_medias:.3f}** (média populacional: "
            f"{ms.media(populacao):.3f}). Desvio padrão das médias amostrais: "
            f"**{dp_das_medias:.3f}** — próximo do erro padrão teórico "
            f"σ/√n = **{erro_padrao_teorico:.3f}**. À medida que n cresce, a distribuição "
            f"das médias amostrais se aproxima de uma Normal, conforme o TCL."
        )

# ---------------------------------------------------------------------------
# MÓDULO 4 — Distribuições teóricas
# ---------------------------------------------------------------------------
elif modulo.startswith("4"):
    st.header("Módulo 4 — Ajuste a distribuições teóricas")
    coluna = st.selectbox("Escolha uma variável numérica", numericas)
    dados = df[coluna].dropna().tolist()

    m = ms.media(dados)
    dp = ms.desvio_padrao(dados)

    st.write(f"Parâmetros estimados a partir dos dados: média = {m:.3f}, desvio padrão = {dp:.3f}")

    fig, ax = plt.subplots()
    ax.hist(dados, bins=40, density=True, color="#1DB954", alpha=0.6, label="Dados observados")

    xs = np.linspace(min(dados), max(dados), 300)
    ys_normal = [ms.pdf_normal(x, m, dp) for x in xs]
    ax.plot(xs, ys_normal, color="black", linewidth=2, label="Normal ajustada")

    segunda_dist = st.selectbox("Segunda distribuição candidata", ["Uniforme", "Exponencial"])
    if segunda_dist == "Uniforme":
        a, b = min(dados), max(dados)
        ys_unif = [ms.pdf_uniforme(x, a, b) for x in xs]
        ax.plot(xs, ys_unif, color="orange", linewidth=2, label="Uniforme ajustada")
    else:
        if m <= 0:
            st.warning("A Exponencial exige valores positivos; escolha outra variável.")
        else:
            lambd = 1.0 / m
            ys_exp = [ms.pdf_exponencial(x, lambd) for x in xs]
            ax.plot(xs, ys_exp, color="orange", linewidth=2, label=f"Exponencial ajustada (λ={lambd:.3f})")

    ax.set_title(f"Ajuste de distribuições — {coluna}")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "Discussão: quanto mais a curva teórica acompanha o formato do histograma, "
        "melhor o ajuste. Note que distribuições unimodais e assimétricas (como a "
        "maioria das variáveis de áudio do Spotify) tendem a se ajustar melhor a uma "
        "Exponencial ou a uma Normal truncada do que a uma Uniforme."
    )

# ---------------------------------------------------------------------------
# MÓDULO 5 — Correlação e regressão linear
# ---------------------------------------------------------------------------
elif modulo.startswith("5"):
    st.header("Módulo 5 — Correlação e regressão linear")

    col_x = st.selectbox("Variável X (independente)", numericas, index=0)
    col_y = st.selectbox("Variável Y (dependente)", numericas, index=min(1, len(numericas) - 1))

    dados_xy = df[[col_x, col_y]].dropna()
    x = dados_xy[col_x].tolist()
    y = dados_xy[col_y].tolist()

    r = ms.correlacao_pearson(x, y)
    b0, b1 = ms.regressao_linear(x, y)
    r2 = ms.r_quadrado(x, y, b0, b1)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.3, s=10, color="#1DB954")
        xs_linha = np.linspace(min(x), max(x), 100)
        ys_linha = [ms.prever(b0, b1, xv) for xv in xs_linha]
        ax.plot(xs_linha, ys_linha, color="red", linewidth=2, label="Reta de regressão")
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.legend()
        st.pyplot(fig)

    with col_b:
        st.metric("Correlação de Pearson (r)", f"{r:.4f}")
        st.metric("R²", f"{r2:.4f}")
        st.write(f"**Equação da reta:**  ŷ = {b0:.4f} + {b1:.4f} · x")

    st.subheader("Predição interativa")
    x_input = st.number_input(f"Digite um valor de {col_x}", value=float(ms.media(x)))
    y_previsto = ms.prever(b0, b1, x_input)
    st.success(f"Predição: para {col_x} = {x_input}, o modelo estima {col_y} ≈ **{y_previsto:.4f}**")

    st.warning(
        "⚠️ **Correlação não implica causalidade.** Um valor alto de r indica apenas que "
        "as duas variáveis variam juntas de forma consistente — não que uma causa a outra. "
        "Pode haver uma terceira variável (ou o acaso) por trás da associação observada."
    )

    if abs(r) > 0.7:
        interpretacao = "uma correlação forte"
    elif abs(r) > 0.3:
        interpretacao = "uma correlação moderada"
    else:
        interpretacao = "uma correlação fraca"
    sentido = "positiva" if r > 0 else "negativa"
    st.info(f"Interpretação: {col_x} e {col_y} apresentam {interpretacao} {sentido} (r = {r:.3f}).")

# ---------------------------------------------------------------------------
# MÓDULO 6 — Descobertas
# ---------------------------------------------------------------------------
else:
    st.header("Módulo 6 — Relatório de descobertas")
    st.markdown(
        """
        As três descobertas estatísticas mais interessantes reveladas pelo laboratório
        estão documentadas em detalhes no arquivo **`docs/RELATORIO.md`**, com os números
        e gráficos correspondentes gerados pelos módulos acima.

        Use os módulos 2 a 5 nesta aplicação para reproduzir e conferir cada uma das
        descobertas antes de finalizar o relatório.
        """
    )
