
from __future__ import annotations
 
import argparse
import json
import re
import warnings
from pathlib import Path
 
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
 
from .features import FEATURE_NAMES
from .features import para_features as _para_features
from .pre_processamento import ORDEM_NIVEIS, montar_trechos, nome_para_nivel
 
ORDEM = ORDEM_NIVEIS                   # ordem crescente de complexidade
SEMENTE = 42
RAIZ = Path(__file__).resolve().parents[1]
 
 
def criar_modelos() -> dict:
    """Candidatos. Os três primeiros usam as features desenhadas à mão;
    o último (TF-IDF) é uma linha de base lexical para comparação."""
    feats = lambda: FunctionTransformer(_para_features)  # noqa: E731
    tfidf = lambda: TfidfVectorizer(  # noqa: E731
        ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    # class_weight="balanced": as classes têm tamanhos diferentes (Superior tem
    # ~2x mais trechos que Fundamental I); sem isso o modelo "chuta" a maior.
    return {
        "LogReg (features)": make_pipeline(
            feats(), StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEMENTE)),
        "RandomForest (features)": make_pipeline(
            feats(), RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=SEMENTE, n_jobs=-1)),
        "GradBoosting (features)": make_pipeline(
            feats(), GradientBoostingClassifier(random_state=SEMENTE)),
        # Híbrido: features de complexidade + vocabulário (TF-IDF) no mesmo modelo.
        "Híbrido (features + TF-IDF)": make_pipeline(
            FeatureUnion([
                ("feats", make_pipeline(feats(), StandardScaler())),
                ("tfidf", tfidf()),
            ]),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEMENTE)),
        # Controle de tema: só as 300 palavras mais comuns do treino (em sua
        # maioria palavras funcionais: que, de, não, você, foi...). Quase não
        # carregam assunto, então medem estilo/registro, não disciplina.
        "Palavras comuns (300) (controle)": make_pipeline(
            TfidfVectorizer(max_features=300, sublinear_tf=True),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEMENTE)),
        "Features + palavras comuns (300)": make_pipeline(
            FeatureUnion([
                ("feats", make_pipeline(feats(), StandardScaler())),
                ("comuns", TfidfVectorizer(max_features=300, sublinear_tf=True)),
            ]),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEMENTE)),
        "TF-IDF + LogReg (baseline)": make_pipeline(
            tfidf(),
            LogisticRegression(max_iter=5000, class_weight="balanced",
                               random_state=SEMENTE)),
    }
 
 
def erro_ordinal(y_real, y_prev) -> float:
    """Distância média, em níveis, entre real e previsto (errar por 1 nível
    é menos grave do que errar por 3)."""
    idx = {n: i for i, n in enumerate(ORDEM)}
    return float(np.mean([abs(idx[a] - idx[b]) for a, b in zip(y_real, y_prev)]))
 
 
def avaliar(trechos: list[dict]) -> dict:
    treino = [r for r in trechos if r["particao"] == "treino"]
    teste = [r for r in trechos if r["particao"] == "teste"]
    X_tr, y_tr = [r["texto"] for r in treino], [r["nivel"] for r in treino]
    X_te, y_te = [r["texto"] for r in teste], [r["nivel"] for r in teste]
    grupos = [r["grupo"] for r in treino]
    modo = trechos[0]["modo_split"]
 
    resultados, modelos_treinados = {}, {}
    for nome, modelo in criar_modelos().items():
        res = {}
        if modo == "arquivo":
            # CV agrupada por arquivo: nenhum arquivo aparece em treino e validação.
            cv = GroupKFold(n_splits=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = cross_val_score(modelo, X_tr, y_tr, groups=grupos,
                                    cv=cv, scoring="f1_macro")
            res["cv_f1_macro_media"] = float(s.mean())
            res["cv_f1_macro_desvio"] = float(s.std())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        res.update({
            "teste_f1_macro": float(f1_score(y_te, pred, average="macro",
                                             labels=ORDEM, zero_division=0)),
            "teste_acuracia": float(accuracy_score(y_te, pred)),
            "teste_erro_ordinal": erro_ordinal(y_te, pred),
            "matriz_confusao": confusion_matrix(y_te, pred, labels=ORDEM).tolist(),
            "relatorio": classification_report(y_te, pred, labels=ORDEM,
                                               zero_division=0, output_dict=True),
        })
        resultados[nome] = res
        modelos_treinados[nome] = modelo
 
    # Escolha do melhor modelo: pela CV (nunca pelo teste, para não "vazar" o
    # teste na seleção). Sem CV (modo bloco), não há critério justo -> nenhum.
    melhor = None
    if modo == "arquivo":
        elegiveis = [
            n for n in resultados
            if "controle" not in n
        ]
        melhor = max(
            elegiveis,
            key=lambda n: resultados[n]["cv_f1_macro_media"]
        )
    return {"modo_split": modo, "n_treino": len(treino), "n_teste": len(teste),
            "melhor": melhor, "resultados": resultados}, modelos_treinados
 
 
def salvar_modelo(modelo, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "features": FEATURE_NAMES, "ordem": ORDEM}, destino)
 
 
def carregar_modelo(caminho: Path = RAIZ / "models" / "modelo_complexidade.joblib"):
    return joblib.load(caminho)["modelo"]
 
 
def prever_proba(modelo, texto: str) -> dict[str, float]:
    """Probabilidade de cada nível para um texto (usada para medir mutações)."""
    p = modelo.predict_proba([texto])[0]
    return dict(zip(modelo.classes_, map(float, p)))
 
 
def localizar_dados(informado: str | None) -> Path:
    """Pasta que contém as 4 subpastas de nível.
 
    Se o usuário informar --dados, usa essa. Senão procura nos locais comuns.
    """
    if informado:
        return Path(informado)
    candidatos = [RAIZ / "dataset" / "dataset_caed", RAIZ / "dataset",
                  RAIZ / "data" / "raw"]
    for c in candidatos:
        if c.is_dir() and any(nome_para_nivel(p.name) for p in c.iterdir() if p.is_dir()):
            return c
    raise SystemExit("Não achei a pasta do dataset. Use: python -m src.modelo --dados <pasta>")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e avalia o classificador de complexidade.")
    parser.add_argument("--dados", help="pasta com as subpastas dos 4 níveis "
                        "(padrão: dataset/dataset_caed)")
    parser.add_argument("--tamanho", type=int, default=2,
                        help="sentenças por trecho (padrão: 2)")
    parser.add_argument("--bloco", type=int, default=1,
                        help="arquivos consecutivos por bloco na separação (padrão: 1 = "
                             "arquivo a arquivo). Use p.ex. 10 para testar vazamento de fonte")
    args = parser.parse_args()
    sufixo = f"t{args.tamanho}" + (f"_b{args.bloco}" if args.bloco > 1 else "")
    dados = localizar_dados(args.dados)
    print(f"Lendo dados de: {dados}  |  sentenças por trecho: {args.tamanho}  |  "
          f"arquivos por bloco: {args.bloco}")
    trechos = montar_trechos(dados, tamanho=args.tamanho, bloco=args.bloco)
    saida, modelos = avaliar(trechos)
    (RAIZ / "relatorio").mkdir(exist_ok=True)
    (RAIZ / "relatorio" / f"resultados_{sufixo}.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    # Salva TODOS os modelos (servirão para comparar como cada um reage às
    # mutações) e, à parte, o melhor pela validação cruzada.
    for nome, m in modelos.items():
        slug = re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")
        salvar_modelo(m, RAIZ / "models" / sufixo / f"{slug}.joblib")
    if saida["melhor"]:
        salvar_modelo(modelos[saida["melhor"]],
                      RAIZ / "models" / "modelo_complexidade.joblib")
 
    print(f"modo_split={saida['modo_split']}  treino={saida['n_treino']}  "
          f"teste={saida['n_teste']}  melhor={saida['melhor']}")
    for nome, r in saida["resultados"].items():
        cv = (f"CV F1={r['cv_f1_macro_media']:.3f}±{r['cv_f1_macro_desvio']:.3f}  "
              if "cv_f1_macro_media" in r else "")
        print(f"{nome:28s} {cv}teste F1={r['teste_f1_macro']:.3f}  "
              f"acc={r['teste_acuracia']:.3f}  erro_ord={r['teste_erro_ordinal']:.2f}")
 
 
if __name__ == "__main__":
    main()
 