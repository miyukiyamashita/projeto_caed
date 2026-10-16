
from __future__ import annotations
 
import math
import re
from functools import lru_cache
 
import numpy as np
from wordfreq import zipf_frequency
 
from .canonicidade import eh_canonica, tokenizar
from .pre_processamento import dividir_sentencas, limpar
 
# Conectivos / pronomes relativos que marcam orações subordinadas ou coordenadas
# (proxy simples de estrutura gramatical complexa, sem precisar de parser).
SUBORDINADORES = {
    "que", "quando", "embora", "porque", "pois", "se", "como", "onde", "cujo",
    "cuja", "cujos", "cujas", "qual", "quais", "enquanto", "conforme", "mas",
    "porém", "contudo", "entretanto", "portanto", "ou", "nem", "caso", "ainda",
    "visto", "apesar", "mesmo", "assim", "logo", "então",
}
 
# Sufixos de palavras abstratas/nominalizadas, frequentes em texto mais formal.
_SUF_ABSTRATOS = ("ção", "ções", "mento", "mentos", "idade", "idades", "ismo")
 
# Passiva analítica (heurística): verbo ser + particípio (-ado/-ido).
_PASSIVA = re.compile(
    r"\b(foi|foram|é|são|era|eram|sendo|ser|seja|sejam|será|serão)\s+\w+(ad|id)[oa]s?\b",
    re.IGNORECASE,
)
 
_PONT_COMPLEXA = re.compile(r"[;:()\[\]–—\"“”]")
 
FEATURE_NAMES = [
    # tamanho
    "n_palavras", "palavras_por_sentenca", "tam_medio_palavra",
    # vocabulário / familiaridade
    "prop_palavras_longas", "prop_palavras_muito_longas", "silabas_por_palavra",
    "ttr_raiz", "prop_abstratas",
    # frequência das palavras no português (lista externa, escala Zipf)
    "zipf_medio", "prop_pouco_frequentes", "prop_raras",
    # canonicidade (feature mínima exigida)
    "prop_canonicas",
    # estrutura / pontuação
    "virgulas_por_sentenca", "prop_pont_complexa", "prop_subordinadores",
    "prop_passiva",
]
 
 
@lru_cache(maxsize=None)
def _zipf(palavra: str) -> float:
    """Frequência da palavra em português na escala Zipf (≈1 rara ... ≈7 muito
    comum). Vem da biblioteca wordfreq: lista externa, independente do nosso
    dataset, então não há vazamento entre treino e teste. Palavra ausente = 0.
    """
    return zipf_frequency(palavra, "pt")
 
 
def _silabas_aprox(palavra: str) -> int:
    """Aproxima nº de sílabas pelo nº de grupos de vogais (núcleos silábicos).
 
    Ditongos contam 1 (o que é correto); hiatos contam 1 (subestima um pouco).
    Como o objetivo é comparar níveis, e não silabificar, a aproximação basta.
    """
    grupos = re.findall(r"[aeiouyáéíóúâêôãõàü]+", palavra.lower())
    return max(1, len(grupos))
 
 
def extrair(texto: str) -> dict[str, float]:
    """Calcula todas as features de um trecho."""
    limpo = limpar(texto)
    palavras = tokenizar(limpo)
    n = len(palavras)
    n_sent = max(1, len(dividir_sentencas(limpo)))
    if n == 0:
        return {k: 0.0 for k in FEATURE_NAMES}
 
    zipfs = [_zipf(p) for p in palavras]
    return {
        "zipf_medio": sum(zipfs) / n,
        "prop_pouco_frequentes": sum(z < 4.0 for z in zipfs) / n,
        "prop_raras": sum(z < 3.0 for z in zipfs) / n,
        "n_palavras": float(n),
        "palavras_por_sentenca": n / n_sent,
        "tam_medio_palavra": sum(map(len, palavras)) / n,
        "prop_palavras_longas": sum(len(p) >= 7 for p in palavras) / n,
        "prop_palavras_muito_longas": sum(len(p) >= 10 for p in palavras) / n,
        "silabas_por_palavra": sum(map(_silabas_aprox, palavras)) / n,
        "ttr_raiz": len(set(palavras)) / math.sqrt(n),
        "prop_abstratas": sum(p.endswith(_SUF_ABSTRATOS) for p in palavras) / n,
        "prop_canonicas": sum(eh_canonica(p) for p in palavras) / n,
        "virgulas_por_sentenca": limpo.count(",") / n_sent,
        "prop_pont_complexa": len(_PONT_COMPLEXA.findall(limpo)) / n,
        "prop_subordinadores": sum(p in SUBORDINADORES for p in palavras) / n,
        "prop_passiva": len(_PASSIVA.findall(limpo)) / n_sent,
    }
 
 
def para_features(textos) -> "np.ndarray":
    """Matriz numpy de features. Fica AQUI (e não em modelo.py) de propósito:
    os modelos salvos com joblib guardam o caminho desta função, e ele precisa
    ser um módulo importável (src.features), não '__main__'."""
    return np.array(extrair_lote(list(textos)))
 
 
def extrair_lote(textos: list[str]) -> list[list[float]]:
    """Matriz [n_textos x n_features] na ordem de FEATURE_NAMES."""
    linhas = []
    for t in textos:
        f = extrair(t)  # uma única passada por texto
        linhas.append([f[k] for k in FEATURE_NAMES])
    return linhas
 