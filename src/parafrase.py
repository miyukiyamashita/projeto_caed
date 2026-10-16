
"""Operador de paráfrase baseado em regras linguísticas.

Interface pública:
    mutar(texto) -> (novo_texto, lista_de_alteracoes)

Características:
    - Preserva espaços, pontuação e quebras de linha.
    - Registra as alterações e suas posições no texto original.
    - Aplica transformações com probabilidade configurável.
    - Permite adicionar novas regras de paráfrase.
"""

from dataclasses import dataclass
import argparse
import random
import re
from pathlib import Path


@dataclass
class AlteracaoParafrase:
    original: str
    novo: str
    posicao: int
    motivo: str


class Parafrase:
    def __init__(self, prob=1.0, semente=None):
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob deve estar entre 0 e 1.")

        self.prob = prob
        self.rng = random.Random(semente)

        # Cada regra recebe uma sentença e retorna uma versão modificada
        # ou None quando não pode ser aplicada.
        self.regras = [
            (
                self._substituir_construcoes,
                "substituição de construção linguística",
            ),
            (
                self._reordenar_temporal,
                "reordenação de expressão temporal",
            ),
            (
                self._reorganizar_aposto,
                "reorganização de título e nome próprio",
            ),
        ]

    @staticmethod
    def _preservar_maiusculas(original, substituto):
        """Adapta a capitalização da substituição à expressão original."""
        if original.isupper():
            return substituto.upper()

        if original.istitle():
            return substituto[0].upper() + substituto[1:]

        if original and original[0].isupper():
            return substituto[0].upper() + substituto[1:]

        return substituto

    @classmethod
    def _substituir_construcoes(cls, sentenca):
        """Substitui expressões por construções de sentido próximo.

        As substituições são deliberadamente limitadas a expressões
        conhecidas, evitando substituir palavras isoladas sem contexto.
        """
        regras = [
            (
                r"\bfaz\s+referência\s+a\b",
                "refere-se a",
            ),
            (
                r"\bfazem\s+referência\s+a\b",
                "referem-se a",
            ),
            (
                r"\bestá\s+relacionado\s+com\b",
                "relaciona-se com",
            ),
            (
                r"\bestá\s+relacionada\s+com\b",
                "relaciona-se com",
            ),
            (
                r"\bestão\s+relacionados\s+com\b",
                "relacionam-se com",
            ),
            (
                r"\bestão\s+relacionadas\s+com\b",
                "relacionam-se com",
            ),
            (
                r"\bcom\s+o\s+objetivo\s+de\b",
                "para",
            ),
            (
                r"\bcom\s+a\s+finalidade\s+de\b",
                "para",
            ),
            (
                r"\bdevido\s+ao\s+fato\s+de\s+que\b",
                "porque",
            ),
            (
                r"\bem\s+outras\s+palavras\b",
                "ou seja",
            ),
            (
                r"\balém\s+disso\b",
                "ademais",
            ),
            (
                r"\bno\s+entanto\b",
                "contudo",
            ),
        ]

        resultado = sentenca

        for padrao, substituto in regras:
            regex = re.compile(padrao, flags=re.IGNORECASE)

            resultado = regex.sub(
                lambda m: cls._preservar_maiusculas(
                    m.group(0), substituto
                ),
                resultado,
                count=1,
            )

        if resultado != sentenca:
            return resultado

        return None

    @staticmethod
    def _reordenar_temporal(sentenca):
        """Move uma expressão temporal inicial 'Há ... atrás' para o fim."""
        padrao = re.compile(
            r"^(Há\s+.+?\s+atrás)[,]?\s+(.+?)([.!?]?)$",
            flags=re.IGNORECASE,
        )

        m = padrao.match(sentenca.strip())

        if not m:
            return None

        expressao_temporal, restante, pontuacao = m.groups()

        # Evita mover uma expressão se o restante não parecer uma oração.
        if len(restante.split()) < 2:
            return None

        # Não aplica se a expressão temporal já aparecer no final.
        if not restante:
            return None

        restante = restante[0].upper() + restante[1:]
        expressao_temporal = (
            expressao_temporal[0].lower()
            + expressao_temporal[1:]
        )

        return f"{restante} {expressao_temporal}{pontuacao}"

    @staticmethod
    def _reorganizar_aposto(sentenca):
        """Reorganiza título ou profissão antes de nome próprio.

        Exemplo:
            O cantor Roberto Carlos fez aniversário.
            Roberto Carlos, o cantor, fez aniversário.

        É uma regra estrutural limitada, não uma análise geral de nomes.
        """
        titulos = (
            "cantor", "cantora", "escritor", "escritora",
            "ator", "atriz", "professor", "professora",
            "presidente", "jogador", "jogadora",
            "pesquisador", "pesquisadora", "autor", "autora",
        )

        titulos_regex = "|".join(
            re.escape(titulo) for titulo in titulos
        )

        nome = (
            r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ]"
            r"[\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç-]*"
        )

        padrao = re.compile(
            rf"^(O|A)\s+({titulos_regex})\s+"
            rf"({nome})\s+({nome})"
            rf"(\s+.+[.!?]?)$"
        )

        m = padrao.match(sentenca.strip())

        if not m:
            return None

        artigo, profissao, nome1, nome2, restante = m.groups()

        # Esta transformação exige um verbo ou complemento após o nome.
        restante = restante.lstrip()

        return (
            f"{nome1} {nome2}, "
            f"{artigo.lower()} {profissao.lower()}, "
            f"{restante}"
        )

    def mutar(self, texto):
        if not isinstance(texto, str):
            raise TypeError("texto deve ser uma string.")

        if not texto or self.prob == 0:
            return texto, []

        partes = []
        alteracoes = []
        deslocamento = 0

        # Processa cada linha separadamente para não perder parágrafos,
        # quebras de linha ou marcadores isolados, como formula_1.
        for linha in texto.splitlines(keepends=True):
            conteudo = linha.rstrip("\r\n")
            quebra = linha[len(conteudo):]

            cursor = 0

            # Localiza sentenças dentro da linha sem reconstruir
            # os espaços e delimitadores entre elas.
            padrao_sentenca = re.compile(
                r"\S.*?(?:[.!?](?=\s|$)|$)"
            )

            for m in padrao_sentenca.finditer(conteudo):
                inicio, fim = m.span()
                original = m.group(0)

                # Preserva integralmente o trecho anterior à sentença.
                partes.append(conteudo[cursor:inicio])

                novo = original
                motivo = None

                # No máximo uma regra é aplicada a cada sentença.
                for regra, descricao in self.regras:
                    if self.rng.random() >= self.prob:
                        continue

                    candidato = regra(novo)

                    if candidato and candidato != novo:
                        novo = candidato
                        motivo = descricao
                        break

                partes.append(novo)

                if motivo:
                    alteracoes.append(
                        AlteracaoParafrase(
                            original=original,
                            novo=novo,
                            posicao=deslocamento + inicio,
                            motivo=motivo,
                        )
                    )

                cursor = fim

            # Preserva todo o conteúdo que não foi reconhecido
            # como sentença e a quebra de linha original.
            partes.append(conteudo[cursor:])
            partes.append(quebra)

            deslocamento += len(linha)

        return "".join(partes), alteracoes


def main():
    parser = argparse.ArgumentParser(
        description="Aplica paráfrases por regras linguísticas."
    )
    parser.add_argument("--arquivo", required=True)
    parser.add_argument("--prob", type=float, default=1.0)
    parser.add_argument("--semente", type=int, default=None)
    args = parser.parse_args()

    caminho = Path(args.arquivo)

    if not caminho.is_file():
        parser.error(f"Arquivo não encontrado: {caminho}")

    texto_original = caminho.read_text(encoding="utf-8")

    try:
        operador = Parafrase(
            prob=args.prob,
            semente=args.semente,
        )
    except ValueError as erro:
        parser.error(str(erro))

    texto_mutado, alteracoes = operador.mutar(texto_original)

    print("\n=== TEXTO ORIGINAL ===\n")
    print(texto_original)

    print("\n=== TEXTO MUTADO ===\n")
    print(texto_mutado)

    print(f"\n=== ALTERAÇÕES: {len(alteracoes)} ===")

    for alteracao in alteracoes:
        print(
            f"{alteracao.original} -> {alteracao.novo} "
            f"(posição: {alteracao.posicao}; "
            f"{alteracao.motivo})"
        )


if __name__ == "__main__":
    main()
