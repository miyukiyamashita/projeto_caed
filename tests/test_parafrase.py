import pytest

from src.parafrase import Parafrase


def test_reordenar_expressao_temporal():
    operador = Parafrase(prob=1.0, semente=42)

    original = "Há duas semanas atrás estava chovendo."
    resultado, alteracoes = operador.mutar(original)

    assert resultado == "Estava chovendo há duas semanas atrás."
    assert len(alteracoes) == 1


def test_reorganizar_aposto():
    operador = Parafrase(prob=1.0, semente=42)

    original = "O cantor Roberto Carlos fez aniversário."
    resultado, alteracoes = operador.mutar(original)

    assert resultado == (
        "Roberto Carlos, o cantor, fez aniversário."
    )
    assert len(alteracoes) == 1


def test_texto_sem_regra_aplicavel():
    operador = Parafrase(prob=1.0, semente=42)

    original = "A água ferve a cem graus Celsius."
    resultado, alteracoes = operador.mutar(original)

    assert resultado == original
    assert alteracoes == []


def test_probabilidade_zero():
    operador = Parafrase(prob=0.0, semente=42)

    original = "Há duas semanas atrás estava chovendo."
    resultado, alteracoes = operador.mutar(original)

    assert resultado == original
    assert alteracoes == []


def test_probabilidade_invalida():
    with pytest.raises(ValueError):
        Parafrase(prob=1.5)

def test_preserva_quebras_de_linha_e_marcadores():
    operador = Parafrase(prob=1.0, semente=42)

    original = (
        "Título\n"
        "Primeiro parágrafo.\n"
        "formula_1\n"
        "Segundo parágrafo."
    )

    resultado, _ = operador.mutar(original)

    assert resultado.count("\n") == original.count("\n")
    assert "formula_1" in resultado
    assert "Título\n" in resultado