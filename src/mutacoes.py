
"""Operadores de mutação de texto (desafio CAEd).

Operador implementado: troca de palavras por sinônimos.

Usa:
- NLTK/WordNet para gerar candidatos;
- spaCy para analisar lema, classe gramatical, gênero e número;
- wordfreq para aproximar a frequência lexical.

A mutação é conservadora: candidatos cuja forma não possa ser produzida
e validada com segurança são descartados.

Limitação: compatibilidade morfológica não garante preservação semântica.
"""


from __future__ import annotations
from functools import lru_cache
import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import spacy
from wordfreq import zipf_frequency


_PALAVRA = re.compile(r"[^\W\d_]+", re.UNICODE)
_INICIO_SENTENCA = re.compile(
    r"""(?:^|[.!?…:]\s*["“'(–-]*\s*)$"""
)

_NAO_TROCAR = {
    "mesmo", "mesma", "mesmos", "mesmas",
    "outro", "outra", "outros", "outras",
    "muito", "muita", "muitos", "muitas",
    "pouco", "pouca", "poucos", "poucas",
    "todo", "toda", "todos", "todas",
    "algum", "alguma", "alguns", "algumas",
    "nenhum", "nenhuma",
    "este", "esta", "esse", "essa",
    "aquele", "aquela", "tanto", "tanta", "cada",
    "certo", "certa", "certos", "certas",
    "próprio", "própria", "próprios", "próprias",
    "primeiro", "primeira", "segundo", "segunda",
    "terceiro", "terceira", "último", "última",
}

_TERMOS_PROTEGIDOS = {
    # Conceitos e termos técnicos de Física
    "matéria", "átomo", "átomos", "molécula", "moléculas",
    "íon", "íons", "massa", "densidade", "volume",
    "força", "forças", "movimento", "aceleração",
    "gravidade", "peso", "newton", "quilograma",
    "mecânica", "vetor", "vetores", "fórmula",
    "fórmulas", "dinâmica", "princípio",
}

# Pares identificados como inadequados nos testes atuais.
# São regras conservadoras para esta primeira rodada de validação.
_PARES_REJEITADOS = {
    ("fundamental", "crítico"),
    ("princípio", "começo"),
    ("química", "substância"),
    ("objeto", "objetivo"),
    ("possível", "potencial"),
    ("sentido", "significado"),
    ("desprezível", "vil"),
    ("fato", "evento"),
    ("decomposição", "deterioração"),
    ("medidas", "unidades"),
    ("unidade", "medida"),
    ("produção", "apresentação"),
    ("importante", "grande"),
}


_TERMOS_TECNICOS_PROTEGIDOS = {
    # Conceitos centrais de Física
    "matéria",
    "átomo",
    "átomos",
    "molécula",
    "moléculas",
    "íon",
    "íons",
    "massa",
    "densidade",
    "volume",
    "força",
    "forças",
    "movimento",
    "aceleração",
    "gravidade",
    "peso",
    "newton",
    "quilograma",
    "mecânica",
    "vetor",
    "vetores",
    "fórmula",
    "fórmulas",
    "dinâmica",
}

@lru_cache(maxsize=1)
def carregar_nlp():
    """Carrega o modelo de português uma única vez."""
    try:
        return spacy.load("pt_core_news_sm")
    except OSError as exc:
        raise RuntimeError(
            "Modelo spaCy de português não encontrado. Execute:\n"
            "python -m spacy download pt_core_news_sm"
        ) from exc


@lru_cache(maxsize=None)
def _zipf(palavra: str) -> float:
    return zipf_frequency(palavra, "pt")


@dataclass
class Troca:
    original: str
    novo: str
    posicao: int
    motivo: str = "compatibilidade_morfologica"
    validacao_semantica: str = "nao_avaliada"


def dicionario_wordnet() -> dict[str, list[str]]:
    """Constrói um dicionário de lemas e sinônimos do WordNet em português."""
    try:
        from nltk.corpus import wordnet as wn

        next(iter(wn.all_synsets("n")))
        wn.synsets("casa", lang="por")
    except (ImportError, LookupError) as exc:
        raise RuntimeError(
            "WordNet/OMW não encontrado. Execute:\n"
            "python -m pip install nltk\n"
            'python -c "import nltk; '
            'nltk.download(\'wordnet\'); nltk.download(\'omw-1.4\')"'
        ) from exc

    sinonimos: dict[str, set[str]] = defaultdict(set)

    for pos in ("n", "a", "s"):
        for synset in wn.all_synsets(pos):
            lemas = {
                lema.lower()
                for lema in synset.lemma_names("por")
                if lema.isalpha()
            }
            for lema in lemas:
                sinonimos[lema].update(lemas - {lema})

    return {
        lema: sorted(candidatos)
        for lema, candidatos in sinonimos.items()
        if candidatos
    }


def _atributos_morfologicos(palavra: str) -> dict:
    """Analisa uma palavra isolada com spaCy, corrigindo exceções conhecidas."""
    doc = carregar_nlp()(palavra)
    token = next(
        (t for t in doc if not t.is_space and not t.is_punct),
        None,
    )

    if token is None:
        return {}

    genero = token.morph.get("Gender")
    numero = token.morph.get("Number")

    # Exceções: substantivos masculinos terminados em "-a".
    excecoes_masculinas = {
        "mapa",
        "problema",
    }

    if palavra.strip().lower() in excecoes_masculinas:
        genero = ["Masc"]

    return {
        "lema": token.lemma_.lower(),
        "pos": token.pos_,
        "genero": genero,
        "numero": numero,
    }


def _tem_atributo(analise: dict, atributo: str, valor: str) -> bool:
    return valor in analise.get(atributo, [])


def _lemas_para_busca(analise: dict) -> list[str]:
    """Gera chaves possíveis para consultar o dicionário de sinônimos."""
    lema = analise["lema"]
    lemas = [lema]

    # Alguns adjetivos femininos são lematizados pelo modelo na forma feminina,
    # enquanto o WordNet pode registrar apenas a forma masculina.
    if (
        analise["pos"] == "ADJ"
        and _tem_atributo(analise, "genero", "Fem")
        and lema.endswith("a")
    ):
        lemas.append(lema[:-1] + "o")

    return list(dict.fromkeys(lemas))


def _pluralizar(forma: str) -> str | None:
    """Pluralização deliberadamente limitada a padrões regulares."""
    if forma.endswith(("a", "e", "o")):
        return forma + "s"

    if forma.endswith(("r", "z")):
        return forma + "es"

    if forma.endswith("m"):
        return forma[:-1] + "ns"

    # Plurais em -l e -ão exigem regras lexicais mais específicas.
    # Palavras terminadas em -s também podem ser singulares invariáveis.
    return None


def _flexionar_candidato(
    lema_candidato: str,
    original: dict,
) -> str | None:
    """Produz uma forma candidata sem inventar flexões não suportadas."""
    analise_base = _atributos_morfologicos(lema_candidato)

    if not analise_base:
        return None

    if analise_base["pos"] != original["pos"]:
        return None

    forma = lema_candidato

    # Substantivos têm gênero lexical. Não transformamos automaticamente
    # um substantivo masculino em feminino (ou vice-versa).
    if original["pos"] == "NOUN":
        genero_original = original["genero"]
        genero_candidato = analise_base["genero"]

        if genero_original and genero_candidato:
            if not set(genero_original).intersection(genero_candidato):
                return None

        elif genero_original != genero_candidato:
            return None

    # Adjetivos regulares em -o podem assumir a forma feminina em -a.
    if original["pos"] == "ADJ":
        precisa_feminino = _tem_atributo(original, "genero", "Fem")
        precisa_masculino = _tem_atributo(original, "genero", "Masc")
        genero_base = analise_base["genero"]

        if precisa_feminino and _tem_atributo(analise_base, "genero", "Masc"):
            if forma.endswith("o"):
                forma = forma[:-1] + "a"
            else:
                return None

        elif precisa_masculino and _tem_atributo(analise_base, "genero", "Fem"):
            if forma.endswith("a"):
                forma = forma[:-1] + "o"
            else:
                return None

        elif not genero_base and (precisa_feminino or precisa_masculino):
            return None

    if _tem_atributo(original, "numero", "Plur"):
        forma_plural = _pluralizar(forma)
        if forma_plural is None:
            return None
        forma = forma_plural

    # Valida a forma efetivamente gerada, não apenas o lema.
    final = _atributos_morfologicos(forma)

    if not final or final["pos"] != original["pos"]:
        return None

    for atributo in ("genero", "numero"):
        esperado = original[atributo]
        obtido = final[atributo]

        if esperado and not obtido:
            return None

        if esperado and obtido and not set(esperado).intersection(obtido):
            return None

    return forma


def _mesma_flexao(original: dict, candidato: dict) -> bool:
    """Compara classe gramatical e atributos morfológicos disponíveis."""
    if not original or not candidato:
        return False

    if original["pos"] != candidato["pos"]:
        return False

    for atributo in ("genero", "numero"):
        esperado = original[atributo]
        obtido = candidato[atributo]

        if esperado and not obtido:
            return False

        if esperado and obtido and not set(esperado).intersection(obtido):
            return False

    return True


@lru_cache(maxsize=1)
def carregar_modelo_semantico():
    """Carrega o modelo multilíngue de similaridade uma única vez."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def similaridade_contextual(texto_original: str, texto_mutado: str) -> float:
    """Calcula a similaridade entre dois trechos usando embeddings."""
    from sentence_transformers import util

    modelo = carregar_modelo_semantico()

    embeddings = modelo.encode(
        [texto_original, texto_mutado],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    return float(
        util.cos_sim(embeddings[0], embeddings[1]).item()
    )


def trecho_contextual(texto: str, inicio: int, fim: int, margem: int = 100) -> str:
    """Extrai uma janela de contexto ao redor da palavra."""
    esquerda = max(0, inicio - margem)
    direita = min(len(texto), fim + margem)
    return texto[esquerda:direita]

def preserva_morfologia_contextual(
    texto: str,
    inicio: int,
    fim: int,
    candidato: str,
) -> bool:
    """
    Rejeita substituições que alterem classe gramatical,
    gênero ou número identificados no contexto original.
    """
    nlp = carregar_nlp()

    doc_original = nlp(texto)

    texto_mutado = texto[:inicio] + candidato + texto[fim:]
    doc_mutado = nlp(texto_mutado)

    original = next(
        (
            token for token in doc_original
            if token.idx == inicio and token.is_alpha
        ),
        None,
    )

    substituto = next(
        (
            token for token in doc_mutado
            if token.idx == inicio and token.is_alpha
        ),
        None,
    )

    if original is None or substituto is None:
        return False

    if original.pos_ != substituto.pos_:
        return False

    for atributo in ("Gender", "Number"):
        esperado = original.morph.get(atributo)
        obtido = substituto.morph.get(atributo)

        if esperado and not obtido:
            return False

        if esperado and obtido and not set(esperado).intersection(obtido):
            return False

    return True


class TrocaDePalavras:
    def __init__(
        self,
        dicionario: dict[str, list[str]] | None = None,
        prob: float = 0.3,
        tolerancia_zipf: float = 0.7,
        tam_min: int = 4,
        max_candidatos: int = 3,
        semente: int | None = None,
        validar_semantica: bool = True,
        similaridade_minima: float = 0.995,
    ):
        if not 0 <= prob <= 1:
            raise ValueError("prob deve estar entre 0 e 1.")
        if tolerancia_zipf < 0:
            raise ValueError("tolerancia_zipf não pode ser negativa.")
        if not 0 <= similaridade_minima <= 1:
            raise ValueError("similaridade_minima deve estar entre 0 e 1.")

        self.dic = dicionario if dicionario is not None else dicionario_wordnet()
        self.prob = prob
        self.tolerancia = tolerancia_zipf
        self.tam_min = tam_min
        self.max_candidatos = max_candidatos
        self.rng = random.Random(semente)
        self.validar_semantica = validar_semantica
        self.similaridade_minima = similaridade_minima

    def candidatos(self, palavra: str) -> list[str]:
        """Retorna candidatos flexionados e com análise morfológica compatível."""
        original = _atributos_morfologicos(palavra.lower())

        if not original or original["pos"] not in {"NOUN", "ADJ"}:
            return []

        if palavra.lower() in _NAO_TROCAR:
            return []

        if palavra.lower() in _TERMOS_PROTEGIDOS:
            return []

        chaves = _lemas_para_busca(original)
        lemas = set()

        for chave in chaves:
            lemas.update(self.dic.get(chave, []))

        # A frequência é comparada usando o lema original.
        z0 = _zipf(original["lema"])
        encontrados = {}

        for candidato in lemas:
            candidato = candidato.lower()

            if not candidato.isalpha() or candidato == original["lema"]:
                continue

            zs = _zipf(candidato)
            if zs <= 0 or abs(zs - z0) > self.tolerancia:
                continue

            forma = _flexionar_candidato(candidato, original)
            if not forma or forma == palavra.lower():
                continue

            analise_final = _atributos_morfologicos(forma)
            if not _mesma_flexao(original, analise_final):
                continue

            # Mantém a menor diferença de frequência para cada forma.
            encontrados[forma] = min(
                encontrados.get(forma, float("inf")),
                abs(zs - z0),
            )

        ordenados = sorted(
            encontrados,
            key=lambda forma: (encontrados[forma], forma),
        )
        return ordenados[: self.max_candidatos]

    def mutar(self, texto: str) -> tuple[str, list[Troca]]:
        """Aplica substituições e registra a validação contextual."""
        partes = []
        trocas = []
        cursor = 0

        for match in _PALAVRA.finditer(texto):
            palavra = match.group()

            partes.append(texto[cursor:match.start()])
            cursor = match.end()

            if (
                not self._elegivel(palavra, texto[:match.start()])
                or self.rng.random() >= self.prob
            ):
                partes.append(palavra)
                continue

            candidatos = self.candidatos(palavra)
            candidatos_validos = []

            for candidato in candidatos:
                candidato_formatado = self._copiar_caixa(palavra, candidato)
                par = (palavra.lower(), candidato_formatado.lower())

                if par in _PARES_REJEITADOS:
                    continue

                texto_candidato = (
                    texto[:match.start()]
                    + candidato_formatado
                    + texto[match.end():]
                )

                if not preserva_morfologia_contextual(
                    texto,
                    match.start(),
                    match.end(),
                    candidato_formatado,
                ):
                    continue
            
                similaridade = None

                if self.validar_semantica:
                    contexto_original = trecho_contextual(
                        texto, match.start(), match.end()
                    )
                    contexto_mutado = trecho_contextual(
                        texto_candidato, match.start(),
                        match.start() + len(candidato_formatado)
                    )

                    try:
                        similaridade = similaridade_contextual(
                            contexto_original, contexto_mutado
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Não foi possível validar a similaridade contextual. "
                            "Confira a instalação do sentence-transformers "
                            "e o download do modelo."
                        ) from exc

                    if similaridade < self.similaridade_minima:
                        continue

                candidatos_validos.append(
                    (candidato_formatado, similaridade)
                )

            if candidatos_validos:
                novo, similaridade = self.rng.choice(candidatos_validos)
                partes.append(novo)

                trocas.append(
                    Troca(
                        original=palavra,
                        novo=novo,
                        posicao=match.start(),
                        motivo="compatibilidade_morfologica_e_contextual",
                        validacao_semantica=(
                            f"similaridade={similaridade:.4f}"
                            if similaridade is not None
                            else "nao_avaliada"
                        ),
                    )
                )
            else:
                partes.append(palavra)

        partes.append(texto[cursor:])
        return "".join(partes), trocas

    def _elegivel(self, palavra: str, antes: str) -> bool:
        if len(palavra) < self.tam_min:
            return False

        if palavra.lower() in _NAO_TROCAR or palavra.isupper():
            return False

        if palavra[0].isupper() and not _INICIO_SENTENCA.search(antes):
            return False

        analise = _atributos_morfologicos(palavra.lower())
        return bool(analise and analise["pos"] in {"NOUN", "ADJ"})

    @staticmethod
    def _copiar_caixa(original: str, novo: str) -> str:
        if original.isupper():
            return novo.upper()
        if original.istitle():
            return novo.capitalize()
        return novo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera uma variação do texto trocando palavras por sinônimos."
    )
    parser.add_argument("--arquivo", required=True, help="arquivo .txt com o texto original")
    parser.add_argument("--prob", type=float, default=0.3)
    parser.add_argument("--tolerancia", type=float, default=0.7)
    parser.add_argument("--semente", type=int, default=None)
    parser.add_argument("--dicionario-json", help="JSON no formato {palavra: [sinônimos]}")
    args = parser.parse_args()

    dicionario = (
        json.loads(Path(args.dicionario_json).read_text(encoding="utf-8"))
        if args.dicionario_json
        else None
    )

    mutador = TrocaDePalavras(
        dicionario,
        prob=args.prob,
        tolerancia_zipf=args.tolerancia,
        semente=args.semente,
    )
    texto = Path(args.arquivo).read_text(encoding="utf-8")
    novo, trocas = mutador.mutar(texto)

    print(novo)
    print(f"\n[{len(trocas)} trocas]")
    for troca in trocas:
        print(
            f"{troca.original} → {troca.novo} | "
            f"{troca.validacao_semantica} | "
            f"motivo={troca.motivo}"
        )


if __name__ == "__main__":
    main()
