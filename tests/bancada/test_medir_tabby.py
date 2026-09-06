"""Parser da linha Metrics do TabbyAPI e determinismo dos prompts. `python -m pytest tests/bancada/test_medir_tabby.py`"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from medir_tabby import parse_metrics, registros, prompt_dificil

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
