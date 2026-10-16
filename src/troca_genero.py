
"""Operador de mutação de gênero com regras morfológicas.

Interface pública:
    mutar(texto) -> (novo_texto, lista_de_alteracoes)

Usa spaCy para análise sintática e regras morfológicas para flexionar
formas regulares. Formas irregulares ficam em um dicionário pequeno.
"""

from dataclasses import dataclass
import argparse
import random
from pathlib import Path

import spacy


@dataclass
class AlteracaoGenero:
    original: str
    novo: str
    posicao: int
    motivo: str


# Exceções irregulares ou que não podem ser obtidas com segurança
# apenas por regras de terminação.
_IRREGULARES = {
    "ator": "atriz",
    "atores": "atrizes",
    "rei": "rainha",
    "reis": "rainhas",
    "príncipe": "princesa",
    "príncipes": "princesas",
    "poeta": "poetisa",
    "poetas": "poetisas",
    "cidadão": "cidadã",
    "cidadãos": "cidadãs",
    "alemão": "alemã",
    "alemães": "alemãs",
    "aluno": "aluna",
    "alunos": "alunas",
}

_DETERMINANTES = {
    "o": "a", "a": "o",
    "os": "as", "as": "os",
    "um": "uma", "uma": "um",
    "uns": "umas", "umas": "uns",

    "meu": "minha", "minha": "meu",
    "meus": "minhas", "minhas": "meus",
    "teu": "tua", "tua": "teu",
    "teus": "tuas", "tuas": "teus",
    "seu": "sua", "sua": "seu",
    "seus": "suas", "suas": "seus",
    "nosso": "nossa", "nossa": "nosso",
    "nossos": "nossas", "nossas": "nossos",
    "vosso": "vossa", "vossa": "vosso",
    "vossos": "vossas", "vossas": "vossos",

    "esse": "essa", "essa": "esse",
    "esses": "essas", "essas": "esses",
    "este": "esta", "esta": "este",
    "estes": "estas", "estas": "estes",
    "esta": "este", "estas": "estes",
    "aquele": "aquela", "aquela": "aquele",
    "aqueles": "aquelas", "aquelas": "aqueles",
    "daquele": "daquela", "daquela": "daquele",
    "daqueles": "daquelas", "daquelas": "daqueles",
}

_TERMINACOES_HUMANAS = (
    "ólogo", "óloga", "ólogos", "ólogas",
    "ista", "istas",
    "dor", "dora", "dores", "doras",
    "tor", "tora", "tores", "toras",
    "sor", "sora", "sores", "soras",
    "eiro", "eira", "eiros", "eiras",
    "ário", "ária", "ários", "árias",
    "ense", "enses",
    "ante", "antes",
    "ente", "entes",
)

_REGRAS_SUBSTANTIVO = (
    ("ólogos", "ólogas"),
    ("ólogas", "ólogos"),
    ("ólogo", "óloga"),
    ("óloga", "ólogo"),
    ("dores", "doras"),
    ("doras", "dores"),
    ("dor", "dora"),
    ("dora", "dor"),
    ("tores", "toras"),
    ("toras", "tores"),
    ("tor", "tora"),
    ("tora", "tor"),
    ("sores", "soras"),
    ("soras", "sores"),
    ("sor", "sora"),
    ("sora", "sor"),
    ("eiros", "eiras"),
    ("eiras", "eiros"),
    ("eiro", "eira"),
    ("eira", "eiro"),
    ("ários", "árias"),
    ("árias", "ários"),
    ("ário", "ária"),
    ("ária", "ário"),
    ("êses", "esas"),
    ("eses", "esas"),
    ("esas", "eses"),
    ("ês", "esa"),
    ("esa", "ês"),
)

_REGRAS_ADJETIVO = (
    ("osos", "osas"),
    ("osas", "osos"),
    ("oso", "osa"),
    ("osa", "oso"),
    ("ados", "adas"),
    ("adas", "ados"),
    ("ado", "ada"),
    ("ada", "ado"),
    ("idos", "idas"),
    ("idas", "idos"),
    ("ido", "ida"),
    ("ida", "ido"),
    ("entos", "entas"),
    ("entas", "entos"),
    ("ento", "enta"),
    ("enta", "ento"),
    ("ivos", "ivas"),
    ("ivas", "ivos"),
    ("ivo", "iva"),
    ("iva", "ivo"),
    ("eiros", "eiras"),
    ("eiras", "eiros"),
    ("eiro", "eira"),
    ("eira", "eiro"),
    ("ários", "árias"),
    ("árias", "ários"),
    ("ário", "ária"),
    ("ária", "ário"),
    ("ões", "onas"),
    ("ona", "ão"),
    ("ãos", "ãs"),
    ("ão", "ã"),
    ("eses", "esas"),
    ("esas", "eses"),
    ("ês", "esa"),
    ("esa", "ês"),
    ("os", "as"),
    ("as", "os"),
    ("o", "a"),
    ("a", "o"),
)


def _preservar_caixa(original, novo):
    if original.isupper():
        return novo.upper()
    if original.istitle():
        return novo.capitalize()
    return novo


def _flexionar(palavra, regras):
    """Aplica a primeira regra de sufixo correspondente."""
    palavra_lower = palavra.lower()

    for origem, destino in regras:
        if palavra_lower.endswith(origem):
            radical = palavra_lower[:-len(origem)]
            novo = radical + destino
            return _preservar_caixa(palavra, novo)

    return None


class TrocaDeGenero:
    def __init__(self, prob=1.0, semente=None):
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob deve estar entre 0 e 1.")

        self.prob = prob
        self.rng = random.Random(semente)
        self.nlp = spacy.load("pt_core_news_sm")

    @staticmethod
    def _genero(token):
        generos = token.morph.get("Gender")

        if "Masc" in generos:
            return "Masc"
        if "Fem" in generos:
            return "Fem"

        return None

    @staticmethod
    def _forma_substantivo(token):
        palavra = token.text.lower()
        genero = TrocaDeGenero._genero(token)

        if genero is None:
            return None, None

        if genero == "Masc" and palavra in _IRREGULARES:
            return _preservar_caixa(token.text, _IRREGULARES[palavra]), "Fem"

        inversos = {
            feminino: masculino
            for masculino, feminino in _IRREGULARES.items()
        }

        if genero == "Fem" and palavra in inversos:
            return _preservar_caixa(token.text, inversos[palavra]), "Masc"

        if not palavra.endswith(_TERMINACOES_HUMANAS):
            return None, None

        if genero == "Masc":
            novo = _flexionar(palavra, _REGRAS_SUBSTANTIVO)
            destino = "Fem"
        else:
            regras_inversas = [
                (fem, masc) for masc, fem in _REGRAS_SUBSTANTIVO
            ]
            novo = _flexionar(palavra, regras_inversas)
            destino = "Masc"

        if novo is None or novo.lower() == palavra:
            return None, None

        return _preservar_caixa(token.text, novo), destino

    @staticmethod
    def _forma_adjetivo(token, genero_destino):
        palavra = token.text.lower()

        irregulares_adjetivos = {
            "francês": "francesa",
            "franceses": "francesas",
            "francesa": "francês",
            "francesas": "franceses",
            "bom": "boa",
            "bons": "boas",
            "boa": "bom",
            "boas": "bons",
            "mau": "má",
            "maus": "más",
            "má": "mau",
            "más": "maus",
            "espanhol": "espanhola",
            "espanhóis": "espanholas",
            "espanhola": "espanhol",
            "espanholas": "espanhóis",
            "famoso": "famosa",
            "famosos": "famosas",
            "famosa": "famoso",
            "famosas": "famosos",
        }

        if palavra in irregulares_adjetivos:
            candidato = irregulares_adjetivos[palavra]

            formas_femininas = {
                "francesa", "francesas", "boa", "boas", "má", "más",
                "espanhola", "espanholas", "famosa", "famosas",
            }
            formas_masculinas = {
                "francês", "franceses", "bom", "bons", "mau", "maus",
                "espanhol", "espanhóis", "famoso", "famosos",
            }

            if genero_destino == "Fem" and palavra in formas_masculinas:
                return _preservar_caixa(token.text, candidato)

            if genero_destino == "Masc" and palavra in formas_femininas:
                return _preservar_caixa(token.text, candidato)

        if genero_destino == "Fem":
            novo = _flexionar(palavra, _REGRAS_ADJETIVO)
        else:
            regras_inversas = [
                (fem, masc) for masc, fem in _REGRAS_ADJETIVO
            ]
            novo = _flexionar(palavra, regras_inversas)

        if novo is None or novo.lower() == palavra:
            return None

        return _preservar_caixa(token.text, novo)

    def mutar(self, texto):
        if not texto or self.prob == 0:
            return texto, []

        doc = self.nlp(texto)
        substituicoes = {}
        motivos = {}

        for token in doc:
            if token.pos_ != "NOUN":
                continue

            alternativa, genero_destino = self._forma_substantivo(token)

            if alternativa is None:
                continue

            if self.rng.random() > self.prob:
                continue

            substituicoes[token.i] = alternativa
            motivos[token.i] = "flexão morfológica de referência humana"

            # Determinantes e adjetivos ligados diretamente ao substantivo.
            for filho in token.children:
                if filho.pos_ == "DET":
                    forma = _DETERMINANTES.get(filho.text.lower())

                    if forma:
                        substituicoes[filho.i] = _preservar_caixa(
                            filho.text, forma
                        )
                        motivos[filho.i] = "concordância do determinante"

                if filho.pos_ == "ADJ":
                    novo_adj = self._forma_adjetivo(
                        filho, genero_destino
                    )

                    if novo_adj:
                        substituicoes[filho.i] = novo_adj
                        motivos[filho.i] = "concordância com a referência"

            # Adjetivos predicativos: em frases como "A cantora era famosa",
            # o adjetivo pode ser a raiz da oração e o substantivo seu sujeito.
            if token.dep_ in {"nsubj", "nsubj:pass"} and token.head.pos_ == "ADJ":
                adjetivo = token.head
                novo_adj = self._forma_adjetivo(adjetivo, genero_destino)

                if novo_adj:
                    substituicoes[adjetivo.i] = novo_adj
                    motivos[adjetivo.i] = (
                        "concordância do adjetivo predicativo"
                    )

        partes = []
        alteracoes = []
        cursor = 0

        tokens_alterados = sorted(
            substituicoes.items(),
            key=lambda item: doc[item[0]].idx,
        )

        for indice, novo in tokens_alterados:
            token = doc[indice]

            if token.idx < cursor:
                continue

            partes.append(texto[cursor:token.idx])
            partes.append(novo)
            cursor = token.idx + len(token.text)

            alteracoes.append(
                AlteracaoGenero(
                    original=token.text,
                    novo=novo,
                    posicao=token.idx,
                    motivo=motivos[indice],
                )
            )

        partes.append(texto[cursor:])
        return "".join(partes), alteracoes


def main():
    parser = argparse.ArgumentParser(
        description="Muta gênero usando regras morfológicas e análise sintática."
    )
    parser.add_argument("--arquivo", required=True)
    parser.add_argument("--prob", type=float, default=1.0)
    parser.add_argument("--semente", type=int, default=None)
    args = parser.parse_args()

    caminho = Path(args.arquivo)

    if not caminho.is_file():
        parser.error(f"Arquivo não encontrado: {caminho}")

    texto_original = caminho.read_text(encoding="utf-8")
    operador = TrocaDeGenero(prob=args.prob, semente=args.semente)
    texto_mutado, alteracoes = operador.mutar(texto_original)

    print("\n=== TEXTO ORIGINAL ===\n")
    print(texto_original)
    print("\n=== TEXTO MUTADO ===\n")
    print(texto_mutado)
    print(f"\n=== ALTERAÇÕES: {len(alteracoes)} ===")

    for alteracao in alteracoes:
        print(
            f"{alteracao.original} -> {alteracao.novo} "
            f"(posição: {alteracao.posicao}; {alteracao.motivo})"
        )


if __name__ == "__main__":
    main()
