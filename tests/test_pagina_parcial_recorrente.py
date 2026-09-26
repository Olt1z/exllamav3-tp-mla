"""
O reaproveitamento da pagina parcial em modelo recorrente (`Job.reusar_pagina_parcial`).

Modelo recorrente so retomava o cache de prefixo na fronteira de pagina, porque o estado recorrente
so era guardado ali. Sob context parallel 4 a pagina logica tem 1.024 tokens, e um agente que
acrescenta ~66 tokens por turno recalculava ~1.000 a cada turno (GLM-5.3-Flash a 295k, 4x A100,
26/09/2026). Agora o estado e guardado tambem no fim de cada prompt, e o turno seguinte retoma dali.

O que estes testes prendem, sem GPU (so indice, hash e contabilidade):
  - o estado e guardado no fim do PROMPT, antes da geracao: restaurar um estado que viu a resposta
    contaminaria o turno seguinte com um texto que o historico dele nao tem;
  - so reaproveita quando os tokens sao os mesmos, e cai no prefill normal em qualquer divergencia;
  - a limpeza do fim de fila nao apaga o checkpoint parcial antes do turno seguinte chegar;
  - sobra sempre ao menos um token para o prefill.
"""
from types import SimpleNamespace

import torch

from exllamav3.cache.recurrent import RecurrentCache, MAX_PARCIAIS
from exllamav3.generator import job as job_mod
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import CachePage, Sequence, chave_parcial, tensor_hash_checksum

PT = 1024  # pagina logica sob context parallel 4


class Estado:
    """O estado recorrente de mentira: so posicao, e o stash que o RecurrentCache guarda."""

    def __init__(self, position, origem = "vivo"):
        self.position = position
        self.origem = origem
        self.checkpoint_size = 10
        self.liberado = False

    def stash(self):
        return {"position": self.position, "checkpoint_size": self.checkpoint_size, "origem": self.origem}

    def free(self):
        self.liberado = True


class TabelaFalsa:
    """O PageTable que o RecurrentCache e o Job consultam, com cadeias declaradas a mao."""

    def __init__(self, paginas = (), vivas = ()):
        self.all_pages = list(paginas)
        self.vivas = set(vivas)
        self.metrics = {}

    def is_resumable(self, h):
        return h in self.vivas

    # O metodo real, emprestado: a busca pelo conteudo e o que esta sob teste
    from exllamav3.generator.pagetable import PageTable as _PT
    pagina_parcial = _PT.pagina_parcial


def _cache_recorrente(tabela):
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    rc.pagetable = tabela
    return rc


def _pagina(indice, prev_hash, tokens, kv_position = None):
    seq = torch.zeros((1, PT), dtype = torch.long)
    seq[0, :len(tokens)] = torch.tensor(tokens, dtype = torch.long)
    return CachePage(
        pagetable = None, page_index = indice, phash = bytes(16), phash_revert = bytes(16),
        prev_hash = prev_hash, prev_hash_revert = None, ref_count = 0, access_serial = 0,
        access_serial_revert = 0, kv_position = len(tokens) if kv_position is None else kv_position,
        kv_position_revert = 0, sequence = seq, can_revert = False, new_page_index = indice,
        children = [], longest_chain = 1,
    )


def _job(rc, tabela, seq, estado, mtp = False):
    copias = []
    cache = SimpleNamespace(
        copy_page = lambda alvo, de, para, n: copias.append((de, para, n)),
        new_from_stashed = lambda stashed, position: Estado(position, origem = stashed["origem"]),
    )
    gen = SimpleNamespace(
        recurrent_cache = rc, cache = cache, draft_model = None, draft_cache = None, mtp_draft = mtp,
        page_tokens = PT,
    )
    j = object.__new__(Job)
    j.generator = gen
    j.pagetable = tabela
    j.sequences = [seq]
    j.embeddings = []
    j.recurrent_state = estado
    j.last_recurrent_checkpoint_pos = None
    j.cached_tokens = 0
    return j, copias


def _seq(tokens, paginas):
    ids = torch.tensor([tokens], dtype = torch.long)
    s = Sequence(ids, ids)
    s.allocated_pages = paginas
    return s


def test_chave_parcial_nao_colide_com_a_de_pagina_inteira():
    ids = torch.arange(PT, dtype = torch.long).view(1, -1)
    anterior = b"\x01" * 16
    assert chave_parcial(ids, anterior) != tensor_hash_checksum(ids, anterior)
    # Encadeada: o mesmo texto depois de outro prefixo e outra chave
    assert chave_parcial(ids[:, :10], anterior) != chave_parcial(ids[:, :10], b"\x02" * 16)
    assert chave_parcial(ids[:, :10], None) == chave_parcial(ids[:, :10].clone(), None)


def test_guarda_no_fim_do_prompt_e_nao_depois():
    """O estado guardado e o do fim do prompt, com a posicao do fim do prompt."""
    anterior = b"\xaa" * 16
    tabela = TabelaFalsa(vivas = {anterior})
    rc = _cache_recorrente(tabela)
    prompt = list(range(100, 100 + PT + 300))
    paginas = [_pagina(0, None, prompt[:PT]), _pagina(7, anterior, prompt[PT:])]
    paginas[0].phash = anterior
    seq = _seq(prompt + [9], paginas)            # o ultimo token do prompt vai no primeiro passo
    seq.kv_position = PT + 300
    j, _ = _job(rc, tabela, seq, Estado(PT + 300))

    Job.maybe_stash_recurrent_parcial(j, seq)

    [(chave, n, parcial)] = list(rc.parciais(anterior))
    assert n == 300
    assert rc[chave]["position"] == PT + 300
    assert chave == chave_parcial(torch.tensor([prompt[PT:]]), anterior)


def test_nao_guarda_na_fronteira_nem_com_estado_fora_de_posicao():
    tabela = TabelaFalsa()
    rc = _cache_recorrente(tabela)
    prompt = list(range(PT))
    seq = _seq(prompt + [1], [_pagina(0, None, prompt), _pagina(1, None, [])])
    seq.kv_position = PT
    j, _ = _job(rc, tabela, seq, Estado(PT))
    Job.maybe_stash_recurrent_parcial(j, seq)       # fronteira: o checkpoint normal ja cobre
    assert len(rc) == 0

    seq2 = _seq(list(range(50)) + [1], [_pagina(0, None, list(range(50)))])
    seq2.kv_position = 50
    j2, _ = _job(rc, tabela, seq2, Estado(48))       # estado nao esta onde o K/V esta
    Job.maybe_stash_recurrent_parcial(j2, seq2)
    assert len(rc) == 0


def _turno_anterior(rc, tabela, prompt_anterior, pagina_fonte):
    """Grava o checkpoint do fim do prompt anterior, como o prefill dele gravaria."""
    seq = _seq(prompt_anterior + [0], [pagina_fonte])
    seq.kv_position = len(prompt_anterior)
    j, _ = _job(rc, tabela, seq, Estado(len(prompt_anterior), origem = "turno anterior"))
    Job.maybe_stash_recurrent_parcial(j, seq)


def test_turno_seguinte_retoma_do_fim_do_prompt_anterior():
    tabela = TabelaFalsa()
    rc = _cache_recorrente(tabela)
    anterior = list(range(1, 301))
    fonte = _pagina(3, None, anterior)
    tabela.all_pages.append(fonte)
    _turno_anterior(rc, tabela, anterior, fonte)
    # Depois do checkpoint o turno anterior GEROU na mesma pagina: a resposta ocupa 300..339
    fonte.sequence[0, 300:340] = 77
    fonte.kv_position = 340

    novo = anterior + list(range(500, 566))         # +66 tokens: a resposta sem raciocinio e a ferramenta
    destino = _pagina(9, None, [], kv_position = 0)
    tabela.all_pages.append(destino)
    seq = _seq(novo + [5], [destino])
    seq.kv_position = 0
    velho = Estado(0)
    j, copias = _job(rc, tabela, seq, velho)

    n = Job.reusar_pagina_parcial(j, seq, 0, seq.sequence_ids.torch_slice(0, len(novo)))

    assert n == 300
    assert copias == [(3, 9, 300)]                   # so o prefixo: a resposta do turno anterior fica
    assert seq.kv_position == 300 and destino.kv_position == 300
    assert destino.sequence[0, :300].tolist() == anterior
    assert velho.liberado
    assert j.recurrent_state.position == 300 and j.recurrent_state.origem == "turno anterior"
    assert j.cached_tokens == 300


def test_qualquer_divergencia_cai_no_prefill_normal():
    tabela = TabelaFalsa()
    rc = _cache_recorrente(tabela)
    anterior = list(range(1, 301))
    fonte = _pagina(3, None, anterior)
    tabela.all_pages.append(fonte)
    _turno_anterior(rc, tabela, anterior, fonte)

    def tenta(prompt, paginas_extra = ()):
        destino = _pagina(9, None, [], kv_position = 0)
        seq = _seq(prompt + [5], [destino])
        seq.kv_position = 0
        estado = Estado(0)
        j, copias = _job(rc, tabela, seq, estado)
        n = Job.reusar_pagina_parcial(j, seq, 0, seq.sequence_ids.torch_slice(0, len(prompt)))
        return n, copias, j, estado

    # Um token diferente no meio do prompt anterior (o historico foi editado)
    editado = list(anterior)
    editado[150] = 9999
    n, copias, j, estado = tenta(editado + list(range(500, 566)))
    assert (n, copias) == (0, []) and j.recurrent_state is estado and not estado.liberado

    # O mesmo prompt, sem nada novo: tem de sobrar um token para o prefill
    n, copias, _, _ = tenta(anterior)
    assert (n, copias) == (0, [])

    # A pagina fonte foi reaproveitada por outra conversa: chave bate, K/V nao existe mais
    fonte.sequence[0, :300] = 4242
    n, copias, _, _ = tenta(anterior + [1, 2, 3])
    assert (n, copias) == (0, [])


def test_outra_pagina_anterior_e_outra_conversa():
    """O mesmo texto parcial depois de outro prefixo nao serve: a chave e encadeada."""
    tabela = TabelaFalsa(vivas = {b"\x01" * 16, b"\x02" * 16})
    rc = _cache_recorrente(tabela)
    trecho = list(range(1, 201))
    fonte = _pagina(3, b"\x01" * 16, trecho)
    tabela.all_pages.append(fonte)
    seq = _seq(list(range(PT)) + trecho + [0], [_pagina(0, None, list(range(PT))), fonte])
    seq.allocated_pages[0].phash = b"\x01" * 16
    seq.kv_position = PT + 200
    j, _ = _job(rc, tabela, seq, Estado(PT + 200))
    Job.maybe_stash_recurrent_parcial(j, seq)
    assert len(list(rc.parciais(b"\x01" * 16))) == 1

    outra = _pagina(0, None, list(range(PT)))
    outra.phash = b"\x02" * 16
    destino = _pagina(8, b"\x02" * 16, [], kv_position = 0)
    seq2 = _seq(list(range(PT)) + trecho + [7, 8, 0], [outra, destino])
    seq2.kv_position = PT
    j2, copias = _job(rc, tabela, seq2, Estado(PT))
    n = Job.reusar_pagina_parcial(j2, seq2, 1, seq2.sequence_ids.torch_slice(PT, PT + 202))
    assert (n, copias) == (0, [])


def test_limpeza_do_fim_de_fila_preserva_o_parcial_ancorado():
    raiz, viva, morta = None, b"\x0a" * 16, b"\x0b" * 16
    tabela = TabelaFalsa(vivas = {viva})
    rc = _cache_recorrente(tabela)
    for prev, tag in ((raiz, 1), (viva, 2), (morta, 3)):
        rc.put(bytes([tag]) * 16, Estado(tag), parcial = {"prev_hash": prev, "n": 10 * tag, "mtp_carry": None})
    rc.put(viva, Estado(PT))            # checkpoint de pagina inteira, ancorado
    rc.put(b"\x0c" * 16, Estado(PT))    # de pagina inteira, encalhado

    rc.prune_stranded()

    assert set(rc.keys()) == {b"\x01" * 16, b"\x02" * 16, viva}


def test_no_maximo_max_parciais_os_mais_antigos_saem():
    tabela = TabelaFalsa()
    rc = _cache_recorrente(tabela)
    rc.put(b"\xff" * 16, Estado(PT))                # de pagina inteira: nunca sai por este limite
    for i in range(MAX_PARCIAIS + 2):
        rc.put(bytes([i + 1]) * 16, Estado(i), parcial = {"prev_hash": None, "n": i + 2, "mtp_carry": None})
    parciais = [k for k, v in rc.items() if "parcial" in v]
    assert len(parciais) == MAX_PARCIAIS
    assert parciais[0] == bytes([3]) * 16
    assert b"\xff" * 16 in rc
    assert rc.metrics["parciais_descartados"] == 2


def test_interruptor_desliga(monkeypatch):
    tabela = TabelaFalsa()
    rc = _cache_recorrente(tabela)
    seq = _seq(list(range(50)) + [1], [_pagina(0, None, list(range(50)))])
    seq.kv_position = 50
    j, _ = _job(rc, tabela, seq, Estado(50))
    monkeypatch.setattr(job_mod, "_PAGINA_PARCIAL", False)
    Job.maybe_stash_recurrent_parcial(j, seq)
    assert len(rc) == 0
