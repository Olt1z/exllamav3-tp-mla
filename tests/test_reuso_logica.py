"""
A logica pura do reaproveitamento de prompt (`exllamav3/generator/reuso.py`), sem torch.

Carrega o modulo pelo caminho, sem passar por `exllamav3/__init__` (que importa torch): roda tambem
numa maquina sem GPU nem torch, com `python3 tests/test_reuso_logica.py`.
"""
import importlib.util
import pathlib
import sys

_CAMINHO = pathlib.Path(__file__).resolve().parents[1] / "exllamav3" / "generator" / "reuso.py"
_spec = importlib.util.spec_from_file_location("reuso_isolado", _CAMINHO)
reuso = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reuso             # o @dataclass procura o modulo aqui
_spec.loader.exec_module(reuso)

P = 256


def test_config_desligada_por_padrao():
    c = reuso.ler_config({})
    assert not c.replay and not c.guarda and not c.proteger_prompt
    assert c.intervalo_pp is None and c.recorrente_bytes is None and c.recorrente_max == 0
    assert c.tokens_de_guarda == ["<|user|>"]
    assert c.limite_parciais(4) == 4
    assert not reuso.ler_config({"EXL3_REPLAY": "0", "EXL3_PONTOS_DE_GUARDA": "off"}).proteger_prompt


def test_config_ligada():
    c = reuso.ler_config({
        "EXL3_REPLAY": "1", "EXL3_PONTOS_DE_GUARDA": "1", "EXL3_GUARDA_TOKENS": "<|user|>, <|observation|>",
        "EXL3_INTERVALO_PP": "8192", "EXL3_RECORRENTE_GIB": "12", "EXL3_RECORRENTE_MAX": "64",
    })
    assert c.replay and c.guarda and c.proteger_prompt
    assert c.tokens_de_guarda == ["<|user|>", "<|observation|>"]
    assert c.intervalo_pp == 8192 and c.recorrente_bytes == 12 * 1024**3 and c.recorrente_max == 64
    # Replay sobe o teto de parciais (um por conversa), a menos que o env diga outro
    assert c.limite_parciais(4) == reuso.MAX_PARCIAIS_REPLAY
    assert reuso.ler_config({"EXL3_REPLAY": "1", "EXL3_MAX_PARCIAIS": "6"}).limite_parciais(4) == 6


def test_config_rejeita_valor_invalido():
    for env in ({"EXL3_INTERVALO_PP": "0"}, {"EXL3_MAX_PARCIAIS": "0"}, {"EXL3_INTERVALO_PP": "x"}):
        try:
            reuso.ler_config(env)
        except ValueError:
            continue
        raise AssertionError(f"aceitou {env}")


def test_pontos_descem_para_a_pagina_e_ficam_entre_o_inicio_e_a_ultima_fronteira():
    fim = 20 * P + 100                          # prefill ate o penultimo token do prompt
    pts = reuso.pontos_de_guarda(P, 0, fim, aberturas = [7 * P + 200])
    assert pts == [7 * P]
    # Antes do inicio em cache, no inicio, ou na ultima fronteira (que o prefill ja guarda): fora
    assert reuso.pontos_de_guarda(P, 8 * P, fim, aberturas = [7 * P + 200]) == []
    assert reuso.pontos_de_guarda(P, 7 * P, fim, aberturas = [7 * P + 200]) == []
    assert reuso.pontos_de_guarda(P, 0, fim, aberturas = [20 * P + 50]) == []
    # Um sistema menor que uma pagina nao tem ponto
    assert reuso.pontos_de_guarda(P, 0, fim, aberturas = [100]) == []


def test_prioridade_e_teto_dos_pontos():
    fim = 100 * P
    pts = reuso.pontos_de_guarda(
        P, 0, fim, explicitos = [50 * P, 10 * P + 3], aberturas = [10 * P + 9, 30 * P], prefixo_kv = 60 * P,
        max_pontos = 3,
    )
    # Explicitos primeiro, sem repetir a mesma pagina; o prefixo de K/V e o ultimo a entrar
    assert pts == [10 * P, 30 * P, 50 * P]
    assert reuso.pontos_de_guarda(P, 0, fim, prefixo_kv = 60 * P) == [60 * P]
    assert reuso.pontos_de_guarda(P, 0, fim, prefixo_kv = 0) == []


def test_prefixo_de_kv_para_na_primeira_lacuna():
    assert reuso.prefixo_de_kv([True, True, False, True]) == 2
    assert reuso.prefixo_de_kv([]) == 0
    assert reuso.prefixo_de_kv([False, True]) == 0


def test_limite_do_reuso_parcial():
    lim = reuso.limite_do_reuso_parcial
    # Sem replay: sempre sobra um token
    assert lim(300, True, False, True, False) == 299
    # Replay no fim do prompt: tudo
    assert lim(300, True, True, False, False) == 300
    # ... menos quando o trecho nao chega ao fim do prompt
    assert lim(300, False, True, True, False) == 299
    # ... e, com rascunho MTP, so se o carry veio junto
    assert lim(300, True, True, False, True) == 299
    assert lim(300, True, True, True, True) == 300


if __name__ == "__main__":
    n = 0
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"{n} testes OK")
