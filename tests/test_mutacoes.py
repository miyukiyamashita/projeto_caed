import pytest

from src.mutacoes import _atributos_morfologicos


@pytest.mark.parametrize(
    "palavra, genero, numero",
    [
        ("menina", "Fem", "Sing"),
        ("meninas", "Fem", "Plur"),
        ("bonito", "Masc", "Sing"),
        ("bonitas", "Fem", "Plur"),
        ("problema", "Masc", "Sing"),
        ("mapa", "Masc", "Sing"),
    ],
)
def test_atributos_morfologicos(palavra, genero, numero):
    resultado = _atributos_morfologicos(palavra)

    assert genero in resultado["genero"]
    assert numero in resultado["numero"]