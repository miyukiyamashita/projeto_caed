
from __future__ import annotations

import inspect
from pathlib import Path

import joblib
import streamlit as st

from src.mutacoes import TrocaDePalavras
from src.troca_genero import TrocaDeGenero
from src.parafrase import Parafrase


RAIZ = Path(__file__).resolve().parent
ORDEM_NIVEIS = [
    "Ensino Fundamental I",
    "Ensino Fundamental II",
    "Ensino Médio",
    "Ensino Superior",
]

st.set_page_config(
    page_title="CAEd | Complexidade textual",
    page_icon="📚",
    layout="wide",
)

st.title("Analisador de complexidade textual")
st.caption(
    "Classifique textos, gere variações e compare as previsões "
    "antes e depois das mutações."
)


@st.cache_resource
def carregar_modelo():
    candidatos = [
        RAIZ / "models" / "modelo_complexidade.joblib",
        RAIZ / "models" / "t6" / "tf_idf_logreg_baseline.joblib",
        RAIZ / "models" / "t2" / "tf_idf_logreg_baseline.joblib",
    ]

    for caminho in candidatos:
        if caminho.exists():
            pacote = joblib.load(caminho)
            modelo = (
                pacote["modelo"]
                if isinstance(pacote, dict) and "modelo" in pacote
                else pacote
            )
            return modelo, caminho

    raise FileNotFoundError(
        "Nenhum modelo foi encontrado. Verifique a pasta models/."
    )


@st.cache_resource
def carregar_dicionario():
    # A construção do dicionário WordNet pode demorar.
    from src.mutacoes import dicionario_wordnet
    return dicionario_wordnet()


def classificar(modelo, texto):
    probabilidades = modelo.predict_proba([texto])[0]
    classes = list(modelo.classes_)

    resultado = {
        str(classe): float(probabilidade)
        for classe, probabilidade in zip(classes, probabilidades)
    }

    nivel = max(resultado, key=resultado.get)
    return nivel, resultado


def construir_operador(classe, probabilidade, semente):
    # Inspeciona os argumentos aceitos pelo operador para não depender
    # de parâmetros opcionais que possam variar entre implementações.
    assinatura = inspect.signature(classe.__init__)
    parametros = assinatura.parameters
    kwargs = {}

    if "prob" in parametros:
        kwargs["prob"] = probabilidade
    if "semente" in parametros:
        kwargs["semente"] = semente
    elif "seed" in parametros:
        kwargs["seed"] = semente

    if classe is TrocaDePalavras:
        return classe(carregar_dicionario(), **kwargs)

    return classe(**kwargs)


def executar_mutacao(texto, tipo, probabilidade, semente):
    classes = {
        "Troca de palavras": TrocaDePalavras,
        "Troca de gênero": TrocaDeGenero,
        "Paráfrase": Parafrase,
    }

    operador = construir_operador(
        classes[tipo], probabilidade, semente
    )
    novo_texto, alteracoes = operador.mutar(texto)
    return novo_texto, alteracoes


def formatar_alteracoes(alteracoes):
    if not alteracoes:
        return "Nenhuma alteração foi registrada pelo operador."

    linhas = []
    for item in alteracoes:
        if isinstance(item, dict):
            original = item.get("original", "?")
            novo = item.get("novo", "?")
            motivo = item.get("motivo", "")
        else:
            original = getattr(item, "original", "?")
            novo = getattr(item, "novo", "?")
            motivo = getattr(item, "motivo", "")

        linha = f"- **{original} → {novo}**"
        if motivo:
            linha += f" — {motivo}"
        linhas.append(linha)

    return "\n".join(linhas)


try:
    modelo, caminho_modelo = carregar_modelo()
    st.caption(f"Modelo carregado: `{caminho_modelo.relative_to(RAIZ)}`")
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()


with st.sidebar:
    st.header("Configurações da mutação")
    tipo_mutacao = st.selectbox(
        "Operador",
        ["Troca de palavras", "Troca de gênero", "Paráfrase"],
    )
    probabilidade = st.slider(
        "Probabilidade de mutação",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Probabilidade configurada no operador. Nem todo texto "
             "ou palavra elegível será necessariamente alterado.",
    )
    semente = st.number_input(
        "Semente aleatória",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
    )


texto = st.text_area(
    "Texto original",
    height=220,
    placeholder="Cole aqui o texto que deseja analisar...",
)

col1, col2 = st.columns(2)

with col1:
    analisar = st.button(
        "Classificar texto",
        type="secondary",
        use_container_width=True,
    )

with col2:
    gerar = st.button(
        "Gerar e comparar mutação",
        type="primary",
        use_container_width=True,
    )


if analisar or gerar:
    if not texto.strip():
        st.warning("Insira um texto antes de continuar.")
        st.stop()

    nivel_original, probs_original = classificar(modelo, texto)

    st.subheader("Texto original")
    st.write(f"**Complexidade prevista:** {nivel_original}")

    st.bar_chart(probs_original)

    with st.expander("Ver probabilidades do texto original"):
        st.json(probs_original)

    if gerar:
        try:
            with st.spinner("Aplicando mutação..."):
                texto_mutado, alteracoes = executar_mutacao(
                    texto,
                    tipo_mutacao,
                    float(probabilidade),
                    int(semente),
                )

            nivel_mutado, probs_mutado = classificar(
                modelo, texto_mutado
            )

            st.divider()
            st.subheader("Comparação")

            esquerda, direita = st.columns(2)

            with esquerda:
                st.markdown("**Original**")
                st.text_area(
                    "Texto original (comparação)",
                    value=texto,
                    height=260,
                    disabled=True,
                    key="original_comparacao",
                )
                st.metric("Nível previsto", nivel_original)

            with direita:
                st.markdown("**Mutado**")
                st.text_area(
                    "Texto após a mutação",
                    value=texto_mutado,
                    height=260,
                    disabled=True,
                    key="mutado_comparacao",
                )
                st.metric(
                    "Nível previsto",
                    nivel_mutado,
                    delta=(
                        "Mudou de nível"
                        if nivel_mutado != nivel_original
                        else "Mesmo nível previsto"
                    ),
                    delta_color="inverse",
                )

            st.subheader("Alterações registradas")
            st.markdown(formatar_alteracoes(alteracoes))

            st.subheader("Probabilidades após a mutação")
            st.bar_chart(probs_mutado)

            with st.expander("Ver probabilidades antes e depois"):
                st.json({
                    "original": probs_original,
                    "mutado": probs_mutado,
                })

            st.info(
                "A comparação representa a resposta do classificador. "
                "Uma mudança de previsão não comprova, isoladamente, "
                "uma mudança real na complexidade linguística."
            )

            st.download_button(
                "Baixar comparação em TXT",
                data=(
                    "TEXTO ORIGINAL\n\n"
                    + texto
                    + "\n\nTEXTO MUTADO\n\n"
                    + texto_mutado
                    + "\n\nNÍVEL ORIGINAL: "
                    + str(nivel_original)
                    + "\nNÍVEL MUTADO: "
                    + str(nivel_mutado)
                    + "\n\nALTERAÇÕES:\n"
                    + formatar_alteracoes(alteracoes)
                ),
                file_name="comparacao_mutacao.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as erro:
            st.error(
                "Ocorreu um erro ao executar a mutação. "
                "Confira as dependências do operador e os recursos "
                f"linguísticos instalados. Detalhe: {erro}"
            )
