
"""Compara o efeito dos operadores de mutação na complexidade prevista.

Operadores:
    - troca_palavras
    - troca_genero
    - parafrase

Todos são avaliados sobre os mesmos textos de teste e o mesmo modelo.

Uso:
    python -m src.avaliar_mutacoes --tamanho 4 --bloco 10
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import __main__

from .features import para_features
from .modelo import ORDEM, RAIZ, localizar_dados
from .pre_processamento import montar_trechos

from .mutacoes import TrocaDePalavras, dicionario_wordnet
from .troca_genero import TrocaDeGenero
from .parafrase import Parafrase


if not hasattr(__main__, "_para_features"):
    __main__._para_features = para_features


MODELOS_PADRAO = ["tf_idf_logreg_baseline"]
OPERADORES = ["troca_palavras", "troca_genero", "parafrase"]
_PALAVRA = re.compile(r"[^\W\d_]+", re.UNICODE)


def amostrar_por_nivel(
    trechos: list[dict],
    n_por_nivel: int,
    semente: int,
) -> list[dict]:
    """Seleciona uma amostra reproduzível e equilibrada por nível."""
    rng = random.Random(semente)
    saida = []

    for nivel in ORDEM:
        do_nivel = [t for t in trechos if t["nivel"] == nivel]
        rng.shuffle(do_nivel)
        saida.extend(do_nivel[:n_por_nivel])

    return saida


def _probas(modelo, textos: list[str]):
    """Retorna probabilidades por classe e nível esperado."""
    P = modelo.predict_proba(textos)
    niveis_colunas = np.array(
        [ORDEM.index(c) for c in modelo.classes_]
    )
    return P, P @ niveis_colunas


def efeito_no_modelo(
    modelo,
    originais: list[str],
    niveis_reais: list[str],
    variacoes: list[list[str]],
) -> dict:
    """Compara as previsões originais com várias rodadas de mutação."""
    P0, E0 = _probas(modelo, originais)
    classes0 = P0.argmax(axis=1)
    reais = np.array(niveis_reais)

    medias_por_nivel = [
        E0[reais == nivel].mean()
        for nivel in ORDEM
        if np.any(reais == nivel)
    ]

    passo = (
        float(np.mean(np.abs(np.diff(medias_por_nivel))))
        if len(medias_por_nivel) > 1
        else 0.0
    )

    mudou = []
    deslocamentos = []
    assinados = []
    variacoes_probabilidade = []

    for mutados in variacoes:
        P1, E1 = _probas(modelo, mutados)

        mudou.append(float(np.mean(P0.argmax(axis=1) != P1.argmax(axis=1))))
        deslocamentos.append(float(np.mean(np.abs(E1 - E0))))
        assinados.append(float(np.mean(E1 - E0)))

        # Distância de variação total entre as distribuições.
        # 0 = probabilidades idênticas; 1 = máxima diferença.
        variacoes_probabilidade.append(
            float(np.mean(0.5 * np.abs(P1 - P0).sum(axis=1)))
        )

    deslocamento = float(np.mean(deslocamentos))

    return {
        "mudou_classe_pct": 100 * float(np.mean(mudou)),
        "desloc_nivel": deslocamento,
        "desloc_assinado": float(np.mean(assinados)),
        "passo_entre_niveis": passo,
        "passo_de_nivel_pct": (
            100 * deslocamento / passo if passo > 0 else None
        ),
        "mudanca_probabilidade_media": float(
            np.mean(variacoes_probabilidade)
        ),
    }


def contar_palavras_alteradas(original: str, mutado: str) -> int:
    """Estima quantas palavras foram afetadas pela mutação.

    Compara sequências de palavras e contabiliza substituições,
    exclusões e inserções. É uma estimativa textual, não uma medida
    de equivalência semântica.
    """
    a = _PALAVRA.findall(original)
    b = _PALAVRA.findall(mutado)

    matcher = difflib.SequenceMatcher(
        a=a,
        b=b,
        autojunk=False,
    )

    total = 0

    for operacao, i1, i2, j1, j2 in matcher.get_opcodes():
        if operacao != "equal":
            total += max(i2 - i1, j2 - j1)

    return total


def criar_operador(
    nome: str,
    prob: float,
    semente: int,
    dicionario: dict | None,
    tolerancia: float,
):
    if nome == "troca_palavras":
        return TrocaDePalavras(
            dicionario,
            prob=prob,
            tolerancia_zipf=tolerancia,
            semente=semente,
        )

    if nome == "troca_genero":
        return TrocaDeGenero(
            prob=prob,
            semente=semente,
        )

    if nome == "parafrase":
        return Parafrase(
            prob=prob,
            semente=semente,
        )

    raise ValueError(f"Operador desconhecido: {nome}")


def executar_operador(
    nome: str,
    originais: list[str],
    prob: float,
    repeticoes: int,
    semente: int,
    dicionario: dict | None,
    tolerancia: float,
):
    """Gera as variações e reúne estatísticas das alterações."""
    variacoes = []
    total_alteracoes = []
    palavras_afetadas = []
    textos_alterados = []
    exemplos = []

    for rep in range(repeticoes):
        operador = criar_operador(
            nome=nome,
            prob=prob,
            semente=semente + rep,
            dicionario=dicionario,
            tolerancia=tolerancia,
        )

        mutados = []

        for indice, original in enumerate(originais):
            novo, alteracoes = operador.mutar(original)
            mutados.append(novo)

            total_alteracoes.append(len(alteracoes))
            palavras_afetadas.append(
                contar_palavras_alteradas(original, novo)
            )
            textos_alterados.append(novo != original)

            if rep == 0 and len(exemplos) < 30:
                exemplos.append({
                    "indice_amostra": indice,
                    "original": original,
                    "mutado": novo,
                    "quantidade_alteracoes": len(alteracoes),
                    "alteracoes": [
                        {
                            "original": a.original,
                            "novo": a.novo,
                            "posicao": a.posicao,
                            "motivo": a.motivo,
                        }
                        for a in alteracoes
                    ],
                })

        variacoes.append(mutados)

    n_textos_por_repeticao = len(originais)

    return {
        "variacoes": variacoes,
        "estatisticas": {
            "alteracoes_por_texto": float(np.mean(total_alteracoes)),
            "palavras_afetadas_por_texto": float(
                np.mean(palavras_afetadas)
            ),
            "pct_textos_alterados": 100 * float(
                np.mean(textos_alterados)
            ),
            "pct_palavras_afetadas": (
                100 * float(np.sum(palavras_afetadas))
                / (n_textos_por_repeticao * repeticoes)
                / float(np.mean([
                    len(_PALAVRA.findall(t)) for t in originais
                ]))
                if originais and np.mean([
                    len(_PALAVRA.findall(t)) for t in originais
                ]) > 0
                else 0.0
            ),
        },
        "exemplos": exemplos,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compara o efeito dos operadores na complexidade prevista."
    )

    parser.add_argument("--dados")
    parser.add_argument("--tamanho", type=int, default=4)
    parser.add_argument("--bloco", type=int, default=1)
    parser.add_argument(
        "--prob",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
    )
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument("--n-por-nivel", type=int, default=150)
    parser.add_argument(
        "--modelos",
        nargs="+",
        default=MODELOS_PADRAO,
    )
    parser.add_argument(
        "--operadores",
        nargs="+",
        choices=OPERADORES,
        default=OPERADORES,
    )
    parser.add_argument("--tolerancia", type=float, default=0.7)
    parser.add_argument("--dicionario-json")
    parser.add_argument("--semente", type=int, default=42)

    args = parser.parse_args()

    if any(p < 0 or p > 1 for p in args.prob):
        parser.error("Todas as probabilidades devem estar entre 0 e 1.")

    if args.repeticoes < 1:
        parser.error("--repeticoes deve ser pelo menos 1.")

    if args.n_por_nivel < 1:
        parser.error("--n-por-nivel deve ser pelo menos 1.")

    sufixo = f"t{args.tamanho}" + (
        f"_b{args.bloco}" if args.bloco > 1 else ""
    )

    pasta_modelos = RAIZ / "models" / sufixo
    modelos = {
        nome: joblib.load(
            pasta_modelos / f"{nome}.joblib"
        )["modelo"]
        for nome in args.modelos
    }

    dados = localizar_dados(args.dados)
    teste = [
        trecho
        for trecho in montar_trechos(
            dados,
            tamanho=args.tamanho,
            bloco=args.bloco,
        )
        if trecho["particao"] == "teste"
    ]

    amostra = amostrar_por_nivel(
        teste,
        args.n_por_nivel,
        args.semente,
    )

    if not amostra:
        parser.error("Nenhum trecho de teste foi encontrado.")

    originais = [t["texto"] for t in amostra]
    niveis = [t["nivel"] for t in amostra]

    print(
        f"{len(amostra)} trechos de teste | "
        f"modelos: {', '.join(modelos)} | pasta: {sufixo}"
    )
    print(f"Operadores: {', '.join(args.operadores)}")

    dicionario = None

    if "troca_palavras" in args.operadores:
        if args.dicionario_json:
            dicionario = json.loads(
                Path(args.dicionario_json).read_text(encoding="utf-8")
            )
        else:
            print("Carregando WordNet...")
            dicionario = dicionario_wordnet()

    resultados = []
    pasta_relatorio = RAIZ / "relatorio"
    pasta_relatorio.mkdir(exist_ok=True)

    for nome_operador in args.operadores:
        for prob in args.prob:
            print(f"Avaliando {nome_operador} | prob={prob:.2f}...")

            execucao = executar_operador(
                nome=nome_operador,
                originais=originais,
                prob=prob,
                repeticoes=args.repeticoes,
                semente=args.semente,
                dicionario=dicionario,
                tolerancia=args.tolerancia,
            )

            variacoes = execucao["variacoes"]
            estatisticas = execucao["estatisticas"]

            for nome_modelo, modelo in modelos.items():
                efeito = efeito_no_modelo(
                    modelo,
                    originais,
                    niveis,
                    variacoes,
                )

                resultados.append({
                    "operador": nome_operador,
                    "prob": prob,
                    "repeticoes": args.repeticoes,
                    "n_textos": len(originais),
                    "modelo": nome_modelo,
                    **estatisticas,
                    **efeito,
                })

            nome_arquivo = (
                f"exemplos_{nome_operador}_{sufixo}_p{prob:.2f}.json"
            )
            (pasta_relatorio / nome_arquivo).write_text(
                json.dumps(
                    execucao["exemplos"],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print("\n=== COMPARAÇÃO DOS OPERADORES ===")
    cabecalho = (
        f"{'operador':<17} {'prob':>5} {'alter./texto':>12} "
        f"{'textos alter.':>14} {'mudou classe':>13} "
        f"{'desloc.':>9} {'passo %':>9} {'viés':>8}"
    )
    print(cabecalho)

    for r in resultados:
        passo = r["passo_de_nivel_pct"]
        passo_str = f"{passo:.1f}%" if passo is not None else "N/A"

        print(
            f"{r['operador']:<17} {r['prob']:5.2f} "
            f"{r['alteracoes_por_texto']:12.2f} "
            f"{r['pct_textos_alterados']:13.1f}% "
            f"{r['mudou_classe_pct']:12.1f}% "
            f"{r['desloc_nivel']:9.3f} "
            f"{passo_str:>9} "
            f"{r['desloc_assinado']:+8.3f}"
        )

    arquivo_resultados = pasta_relatorio / f"comparacao_mutacoes_{sufixo}.json"
    arquivo_resultados.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nResultados salvos em: {arquivo_resultados}")
    print("Os exemplos de cada operador e probabilidade também foram salvos.")


if __name__ == "__main__":
    main()
