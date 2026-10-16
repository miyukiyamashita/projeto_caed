
from __future__ import annotations
 
import re
import unicodedata
 
# Vogais do português, incluindo acentuadas, nasais (ã, õ) e trema.
VOGAIS = set("aeiouyáéíóúâêôãõàü")
 
# Dígrafos consonantais tratados como uma única consoante.
_DIGRAFOS = ("ch", "lh", "nh")
 
# qu / gu só viram uma consoante antes de e/i (que, qui, gue, gui).
_QU_GU_RE = re.compile(r"(qu|gu)(?=[eéêií])")
 
# Palavra = sequência de letras (inclui acentos); ignora números e símbolos.
_PALAVRA_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
 
 
def limpar_texto(texto: str) -> str:
    """Normaliza o texto antes de qualquer análise.
 
    - NFC: garante que "é" seja um único caractere (e não "e" + acento),
      senão o teste de vogal falharia.
    - Remove hífen opcional (U+00AD): aparece nos dados (ex.: "o­ceano" em
      281_5112.txt) e quebraria a palavra em duas.
    """
    texto = unicodedata.normalize("NFC", texto)
    return texto.replace("­", "")
 
 
def tokenizar(texto: str) -> list[str]:
    """Extrai as palavras do texto em minúsculas."""
    return [p.lower() for p in _PALAVRA_RE.findall(limpar_texto(texto))]
 
 
def _unidades(palavra: str) -> str:
    """Converte a palavra em uma string de 'C' e 'V'.
 
    Dígrafos (ch, lh, nh, qu/gu antes de e/i) viram uma única 'C'.
    """
    p = palavra.lower()
    p = _QU_GU_RE.sub("#", p)  
    for d in _DIGRAFOS:
        p = p.replace(d, "#")
    return "".join("V" if c in VOGAIS else "C" for c in p)
 
 
_CANONICA_RE = re.compile(r"(?:CV)+")
 
 
def eh_canonica(palavra: str) -> bool:
    """True se todas as sílabas da palavra forem do tipo CV."""
    if not palavra:
        return False
    return _CANONICA_RE.fullmatch(_unidades(palavra)) is not None
 
 
def proporcao_canonicas(texto: str) -> float:
    """Fração das palavras do texto que são canônicas (feature do modelo)."""
    palavras = tokenizar(texto)
    if not palavras:
        return 0.0
    return sum(eh_canonica(p) for p in palavras) / len(palavras)
 