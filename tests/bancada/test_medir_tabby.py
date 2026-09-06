"""Parser da linha Metrics do TabbyAPI e determinismo dos prompts. `python -m pytest tests/bancada/test_medir_tabby.py`"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from medir_tabby import parse_metrics, registros, prompt_dificil, desquebrar

# Como sai no log de verdade (TR3 em TP4, 06/09/2026 04:04Z): o logger quebra a ~80 colunas
QUEBRADO = """2026-09-06 04:04:21.722 INFO:     Metrics (ID:
133d6431af7c486a8ed0820d01972528): 196 tokens generated in 14.98 seconds (Queue:
0.01 s, Process: 0 cached tokens and 5436 new tokens at 474.76 T/s, Generate:
55.68 T/s, Context: 5436 tokens, Draft: 141 / 165 tokens accepted (85.45%))
2026-09-06 04:04:21.724 WARNING:  Unable to switch model to x because
"inline_model_loading" is not True in config.yml.
"""


def test_desquebrar_cola_a_linha_metrics():
    regs = desquebrar(QUEBRADO)
    assert len(regs) == 2
    m = parse_metrics(regs[0])
    assert m["id"] == "133d6431af7c486a8ed0820d01972528" and m["gerados"] == 196
    assert m["novos_tok"] == 5436 and m["prefill_tps"] == 474.76 and m["decode_tps"] == 55.68 and m["draft_pct"] == 85.45

LINHA = ("INFO:     Metrics (ID: 6c6371d0a1b24c3e9f0d): 596 tokens generated in 9.34 seconds "
         "(Queue: 0.0 s, Process: 0 cached tokens and 52 new tokens at 312.5 T/s, Generate: 63.8 T/s, "
         "Context: 52 tokens, Draft: 287 / 596 tokens accepted (48.15%))")


def test_parse_metrics():
    m = parse_metrics(LINHA)
    assert m["id"] == "6c6371d0a1b24c3e9f0d" and m["gerados"] == 596 and m["segundos"] == 9.34
    assert m["cache_tok"] == 0 and m["novos_tok"] == 52 and m["contexto"] == 52
    assert m["prefill_tps"] == 312.5 and m["decode_tps"] == 63.8 and m["draft_pct"] == 48.15


def test_parse_sem_draft_e_lixo():
    m = parse_metrics(LINHA.replace(", Draft: 287 / 596 tokens accepted (48.15%)", ""))
    assert m["decode_tps"] == 63.8 and m["draft_pct"] is None
    assert parse_metrics("Received chat completion request 6c6371d0") is None


def test_prompts_deterministicos():
    a, soma_a, agulha_a = registros(1000)
    b, soma_b, agulha_b = registros(1000)
    assert a == b and soma_a == soma_b and agulha_a == agulha_b
    assert agulha_a.startswith("CHAVE-") and a.count(agulha_a) == 1
    assert soma_a > 0 and "Ouro Preto" in a
    prompt, soma, agulha = prompt_dificil()
    assert prompt.endswith("(formato CHAVE-XXXXXX)?") and soma == soma_a and agulha == agulha_a
