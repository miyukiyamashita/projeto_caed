
from __future__ import annotations
 
import csv
import math
import random
import re
import unicodedata
from pathlib import Path
 
from .canonicidade import limpar_texto
 
# Níveis em ordem crescente de complexidade.
ORDEM_NIVEIS = [
    "Ensino Fundamental I",
    "Ensino Fundamental II",
    "Ensino Médio",
    "Ensino Superior",
]
 
# Nome de pasta "normalizado" -> nível. Aceita '1_Ensino_Fundamental_I',
# 'ensino fundamental i', 'fundamental1' etc. (ver _normalizar_pasta).
_PASTA_PARA_NIVEL = {
    "ensinofundamentali": ORDEM_NIVEIS[0], "fundamental1": ORDEM_NIVEIS[0],
    "ensinofundamentalii": ORDEM_NIVEIS[1], "fundamental2": ORDEM_NIVEIS[1],
    "ensinomedio": ORDEM_NIVEIS[2], "medio": ORDEM_NIVEIS[2],
    "ensinosuperior": ORDEM_NIVEIS[3], "superior": ORDEM_NIVEIS[3],
}
 
# Mínimo de arquivos por nível para separar treino/teste POR ARQUIVO.
MIN_ARQUIVOS_POR_NIVEL = 3
 
# Fim de sentença: . ! ? (com aspas/parênteses opcionais) seguido de espaço e
# início de nova sentença (maiúscula, dígito, aspas ou marcador "- ").
_FIM_SENTENCA = re.compile(
    r"(?<=[.!?])[\"”')]*\s+(?=[-–]?\s*[\"“(]?[A-ZÀ-ÝÇ0-9])"
)
 
 
def _normalizar_pasta(nome: str) -> str:
    """'1_Ensino_Médio' -> 'ensinomedio' (sem prefixo numérico, acento, símbolos)."""
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    sem_prefixo = re.sub(r"^\d+[\s_\-.]*", "", sem_acento)
    return re.sub(r"[^a-z0-9]", "", sem_prefixo.lower())
 
 
def nome_para_nivel(nome_pasta: str) -> str | None:
    """Nível correspondente a um nome de pasta, ou None se não reconhecer."""
    return _PASTA_PARA_NIVEL.get(_normalizar_pasta(nome_pasta))
 
 
def _remover_cabecalhos(texto: str) -> str:
    """Remove linhas curtas de metadado, como o 'breadcrumb' do texto de
    nível superior ('Logística/Manutenção de instalações/Plano de manutenção').
 
    Elas não são conteúdo do texto e, pior, entregam o tema ao modelo.
    """
    linhas = []
    for linha in texto.splitlines():
        s = linha.strip()
        eh_cabecalho = (
            "/" in s and len(s.split()) <= 8 and not re.search(r"[.!?:;]$", s)
        )
        if not eh_cabecalho:
            linhas.append(s)
    return " ".join(l for l in linhas if l)
 
 
def limpar(texto: str) -> str:
    """Limpeza completa: Unicode/hífen opcional, cabeçalhos e espaços."""
    texto = _remover_cabecalhos(limpar_texto(texto))
    return re.sub(r"\s+", " ", texto).strip()
 
 
def dividir_sentencas(texto: str) -> list[str]:
    """Divide em sentenças por regra simples (pontuação + maiúscula seguinte)."""
    return [s.strip() for s in _FIM_SENTENCA.split(texto) if s.strip()]
 
 
def criar_janelas(sentencas: list[str], tamanho: int = 2) -> list[str]:
    """Agrupa sentenças consecutivas em trechos SEM sobreposição.
 
    Se sobrar menos que metade de uma janela, o resto é anexado à anterior
    (evita um trecho minúsculo e não joga conteúdo fora).
    """
    janelas = [sentencas[i:i + tamanho] for i in range(0, len(sentencas), tamanho)]
    if len(janelas) > 1 and len(janelas[-1]) < math.ceil(tamanho / 2):
        janelas[-2].extend(janelas.pop())
    return [" ".join(j) for j in janelas]
 
 
def _ler(caminho: Path) -> str:
    """Lê o arquivo em UTF-8; se falhar, tenta Latin-1 (comum em dados antigos)."""
    try:
        return caminho.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return caminho.read_text(encoding="latin-1")
 
 
def carregar_corpus(raiz: str | Path) -> list[dict]:
    """Lê <raiz>/<pasta do nível>/**/*.txt e devolve
    [{nivel, arquivo, grupo, texto}, ...].
 
    - Só entram subpastas cujo nome seja reconhecido como um dos 4 níveis.
    - Os .txt são procurados recursivamente (funciona com subpastas).
    - 'grupo' identifica o arquivo de origem de forma única (pasta/caminho).
    """
    raiz = Path(raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {raiz}")
    corpus = []
    for pasta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        nivel = nome_para_nivel(pasta.name)
        if nivel is None:
            continue
        for f in sorted(pasta.rglob("*.txt")):
            m = re.match(r"\d+", f.stem)
            corpus.append({
                "nivel": nivel, "arquivo": f.name,
                "grupo": f"{pasta.name}/{f.relative_to(pasta).as_posix()}",
                "ordem": int(m.group()) if m else None,   # 1º número do nome
                "texto": _ler(f),
            })
    faltando = [n for n in ORDEM_NIVEIS if not any(d["nivel"] == n for d in corpus)]
    if faltando:
        achadas = sorted(p.name for p in raiz.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"Nenhum .txt encontrado para: {faltando}. "
            f"Subpastas em {raiz}: {achadas}")
    return corpus
 
 
def _particao_por_arquivo(corpus: list[dict], fracao_teste: float,
                          semente: int) -> dict[str, str]:
    """Sorteia arquivos inteiros para teste, mantendo a proporção em cada nível."""
    rng = random.Random(semente)
    particao = {}
    for nivel in ORDEM_NIVEIS:
        # únicos, preservando a ordem (com blocos, vários arquivos têm o mesmo grupo)
        grupos = list(dict.fromkeys(d["grupo"] for d in corpus if d["nivel"] == nivel))
        rng.shuffle(grupos)
        n_teste = max(1, round(len(grupos) * fracao_teste))
        for i, g in enumerate(grupos):
            particao[g] = "teste" if i < n_teste else "treino"
    return particao
 
 
def _agrupar_em_blocos(corpus: list[dict], bloco: int) -> None:
    """Troca o 'grupo' de cada arquivo pelo seu BLOCO de arquivos consecutivos.
 
    Os arquivos de cada nível são ordenados pelo 1º número do nome e agrupados
    de `bloco` em `bloco`. Se arquivos vizinhos vierem do mesmo material, o
    bloco inteiro vai para treino ou para teste (e para a mesma dobra da
    validação cruzada), eliminando esse vazamento de fonte.
    """
    for nivel in ORDEM_NIVEIS:
        docs = sorted((d for d in corpus if d["nivel"] == nivel),
                      key=lambda d: (d["ordem"] is None, d["ordem"] or 0, d["arquivo"]))
        for rank, d in enumerate(docs):
            d["grupo"] = f"{nivel}/bloco{rank // bloco:04d}"
 
 
def montar_trechos(raiz: str | Path, tamanho: int = 2,
                   fracao_teste: float = 0.2, semente: int = 42,
                   bloco: int = 1) -> list[dict]:
    """Gera a tabela de trechos com a partição treino/teste.
 
    Modo 'arquivo' (preferido): cada arquivo vai INTEIRO para treino ou teste.
    Trechos do mesmo material didático compartilham tema e vocabulário; se
    ficassem nos dois lados, o modelo seria testado em algo parecido com o que
    já viu e a métrica seria inflada.
 
    Modo 'bloco' (fallback, p/ poucos arquivos): dentro de cada arquivo, os
    trechos finais vão para teste. Só serve para testar o pipeline.
    """
    corpus = carregar_corpus(raiz)
    if bloco > 1:
        _agrupar_em_blocos(corpus, bloco)
    por_nivel = {n: len({d["grupo"] for d in corpus if d["nivel"] == n})
                 for n in ORDEM_NIVEIS}
    modo = ("arquivo" if min(por_nivel.values()) >= MIN_ARQUIVOS_POR_NIVEL
            else "bloco")
    particao_arq = (_particao_por_arquivo(corpus, fracao_teste, semente)
                    if modo == "arquivo" else {})
 
    linhas = []
    for doc in corpus:
        janelas = criar_janelas(dividir_sentencas(limpar(doc["texto"])), tamanho)
        if modo == "bloco":
            n_teste = max(1, round(len(janelas) * 0.3))
            corte = len(janelas) - n_teste
        for i, trecho in enumerate(janelas):
            if modo == "arquivo":
                part = particao_arq[doc["grupo"]]
            else:
                part = "treino" if i < corte else "teste"
            linhas.append({
                "id": f"{doc['grupo']}#{i:02d}",
                "nivel": doc["nivel"],
                "grupo": doc["grupo"],
                "indice": i,
                "particao": part,
                "modo_split": modo,
                "texto": trecho,
            })
    return linhas
 
 
def salvar_csv(linhas: list[dict], destino: str | Path) -> None:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
 