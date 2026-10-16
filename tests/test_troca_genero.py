from src.troca_genero import TrocaDeGenero


def test_operador_nao_altera_texto_vazio():
    operador = TrocaDeGenero(prob=1.0, semente=42)

    texto, alteracoes = operador.mutar("")

    assert texto == ""
    assert alteracoes == []


def test_operador_probabilidade_zero():
    operador = TrocaDeGenero(prob=0.0, semente=42)

    original = "O aluno chegou."
    texto, _ = operador.mutar(original)

    assert texto == original


def test_operador_retorna_texto_e_lista():
    operador = TrocaDeGenero(prob=1.0, semente=42)

    texto, alteracoes = operador.mutar(
        "O professor chegou."
    )

    assert isinstance(texto, str)
    assert isinstance(alteracoes, list)