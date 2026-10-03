"""
Contagem de tokens gerados atraves de varios requeues (max_rq_tokens).

O "new_tokens" do resultado final e rq_new_tokens + new_tokens do ultimo segmento. Antes, o
prepare_for_requeue passava adiante so o new_tokens do segmento que acabava, entao do segundo
requeue em diante os segmentos anteriores sumiam da conta: uma resposta de ~47 mil tokens saia
no usage do TabbyAPI como os ~4,7 mil dos dois ultimos segmentos, com mais rascunho aceito que token
gerado. Sem GPU: o prepare_for_queue e trocado por um que so prende o gerador.
"""
from types import SimpleNamespace

import torch

from exllamav3.generator.job import Job


def _gera(job, n):
    seq = job.sequences[0]
    for _ in range(n):
        seq.sequence_ids.append(torch.tensor([[7]], dtype = torch.long))
    job.new_tokens += n
    job.accepted_draft_tokens += n // 2


def test_new_tokens_soma_todos_os_segmentos(monkeypatch):
    def prepare_for_queue(self, generator, serial_number, rq = False):
        self.generator = generator
        self.serial_number = serial_number
    monkeypatch.setattr(Job, "prepare_for_queue", prepare_for_queue)

    job = Job(input_ids = torch.ones((1, 100), dtype = torch.long), max_new_tokens = 100000, max_rq_tokens = 4096)
    job.prepare_for_queue(SimpleNamespace(page_tokens = 256), 0)
    segmentos = [6076, 4096, 4096, 4096]
    for n in segmentos:
        _gera(job, n)
        job = job.prepare_for_requeue()
    _gera(job, 4683)

    total = sum(segmentos) + 4683
    assert job.rq_new_tokens + job.new_tokens == total
    assert job.accepted_draft_tokens == sum(n // 2 for n in segmentos) + 4683 // 2
    # O limite que resta tambem desconta todos os segmentos
    assert job.max_new_tokens == 100000 - sum(segmentos)
    assert job.rq_prompt_tokens == 100
