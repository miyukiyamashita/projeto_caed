 
from __future__ import annotations
 
import argparse
from collections import Counter, defaultdict
from pathlib import Path
 
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
 
from .features import FEATURE_NAMES, extrair_lote
from .modelo import ORDEM, RAIZ, localizar_dados
from .pre_processamento import carregar_corpus, montar_trechos
 
linhas_saida: list[str] = []
 
 
def p(texto: str = "") -> None:
    print(texto)
    linhas_saida.append(texto)
 
 
def estrutura(dados: Path) -> None:
    p("=== 1. ESTRUTURA DOS ARQUIVOS ===")
    corpus = carregar_corpus(dados)
    por_nivel = defaultdict(list)
    for d in corpus:
        por_nivel[d["nivel"]].append(d)
    prefixos_por_nivel = {}
    for n in ORDEM:
        docs = por_nivel[n]
        palavras = [len(d["texto"].split()) for d in docs]
        prefixos = Counter(d["arquivo"].split("_")[0] for d in docs)
        prefixos_por_nivel[n] = set(prefixos)
        p(f"{n:22s} arquivos={len(docs):4d}  palavras/arquivo: "
          f"mediana={int(np.median(palavras))} mín={min(palavras)} máx={max(palavras)}  "
          f"prefixos distintos (antes do '_')={len(prefixos)}")
        # Segundo número do nome (ex.: '665' em 108_665.txt): se muitos arquivos
        # repetem o mesmo valor, ele pode identificar o material de origem.
        sufixos = Counter(Path(d["arquivo"]).stem.split("_")[-1] for d in docs)
        mais_comuns = ", ".join(f"{s}×{c}" for s, c in sufixos.most_common(3))
        p(f"{'':22s} segundo número: {len(sufixos)} distintos; mais repetidos: {mais_comuns}")
    todos = Counter(x for s in prefixos_por_nivel.values() for x in s)
    repetidos = [x for x, c in todos.items() if c > 1]
    p(f"Prefixos que aparecem em mais de um nível: {len(repetidos)}")
    p("Dica: se vários arquivos de um nível compartilham o mesmo prefixo, "
      "podem ser do mesmo material, e separar por arquivo ainda vaza fonte.")
    p()
 
 
def medias_features(trechos: list[dict]) -> None:
    p("=== 2. FEATURES: MÉDIA POR NÍVEL e correlação com o nível (Spearman) ===")
    X = np.array(extrair_lote([t["texto"] for t in trechos]))
    y = np.array([ORDEM.index(t["nivel"]) for t in trechos])
    cab = f"{'feature':28s}" + "".join(f"{n.replace('Ensino ', '')[:14]:>16s}" for n in ORDEM) + f"{'rho':>8s}"
    p(cab)
    linhas = []
    for j, nome in enumerate(FEATURE_NAMES):
        medias = [X[y == i, j].mean() for i in range(len(ORDEM))]
        rho = spearmanr(X[:, j], y).statistic
        linhas.append((abs(rho), nome, medias, rho))
    for _, nome, medias, rho in sorted(linhas, reverse=True):
        p(f"{nome:28s}" + "".join(f"{m:16.3f}" for m in medias) + f"{rho:8.2f}")
    p("(ordenado por |rho|: quanto maior, mais a feature acompanha o nível)")
    p()
 
 
def termos_tfidf(trechos: list[dict], k: int = 15) -> None:
    p("=== 3. TERMOS MAIS CARACTERÍSTICOS DE CADA NÍVEL (TF-IDF + LogReg) ===")
    textos = [t["texto"] for t in trechos]
    y = [t["nivel"] for t in trechos]
    vet = TfidfVectorizer(min_df=5, sublinear_tf=True)
    X = vet.fit_transform(textos)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y)
    vocab = np.array(vet.get_feature_names_out())
    for i, nivel in enumerate(clf.classes_):
        top = vocab[np.argsort(clf.coef_[i])[::-1][:k]]
        p(f"{nivel}: " + ", ".join(top))
    p("Dica: termos de tema (ex.: nomes de disciplinas, personagens) ou de "
      "ortografia/fonte (ex.: 'actividades') indicam que o TF-IDF pode estar "
      "aprendendo a ORIGEM do texto, não a sua complexidade.")
    p()
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados")
    parser.add_argument("--tamanho", type=int, default=2)
    args = parser.parse_args()
    dados = localizar_dados(args.dados)
    estrutura(dados)
    trechos = montar_trechos(dados, tamanho=args.tamanho)
    medias_features(trechos)
    termos_tfidf(trechos)
    (RAIZ / "relatorio").mkdir(exist_ok=True)
    (RAIZ / "relatorio" / "diagnostico.txt").write_text(
        "\n".join(linhas_saida), encoding="utf-8")
 
 
if __name__ == "__main__":
    main()
 