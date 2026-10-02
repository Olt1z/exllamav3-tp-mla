"""
Reaproveitamento de prompt em modelo hibrido: replay do prompt identico, pontos de guarda e o LRU dos
checkpoints recorrentes (EXL3_REPLAY, EXL3_PONTOS_DE_GUARDA; ver `exllamav3/generator/reuso.py`).

Sem GPU: o `Job.prefill` de verdade roda sobre um modelo de mentira que so avanca a posicao do estado
recorrente, e o que se confere e onde os blocos de prefill caem, onde os checkpoints sao guardados e
quando o prefill e pulado. A prova de que a saida e a mesma esta em `tests/bancada/medir_reuso.py`.

O que estes testes prendem:
  - com replay, um prompt identico ao guardado nao roda nenhum token de prefill (sem replay, roda o
    trecho da ultima pagina, como antes);
  - com rascunho MTP o replay so acontece se o checkpoint trouxe o carry, e o carry vira o
    `mtp_last_hidden` que o primeiro rascunho consome;
  - um ponto de guarda corta o bloco de prefill e guarda o estado ali, sem deslocar os blocos
    seguintes (os checkpoints de intervalo continuam nas mesmas posicoes);
  - os pontos automaticos: o primeiro `<|user|>` e o fim do prefixo de K/V ja em cache;
  - o LRU despeja checkpoints de geracao antes dos do prompt, e respeita o teto de quantidade.
"""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache.recurrent import RecurrentCache
from exllamav3.generator import reuso
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import CachePage, PageTable, Sequence, chave_parcial

PT = 256
CHUNK = 4 * PT
USER = 151336           # um id qualquer fazendo o papel de <|user|>


@pytest.fixture
def config(monkeypatch):
    def liga(**env):
        monkeypatch.setattr(reuso, "CONFIG", reuso.ler_config({k: str(v) for k, v in env.items()}))
    liga()
    return liga


class Estado:
    def __init__(self, position, origem = "vivo"):
        self.position = position
        self.origem = origem
        self.checkpoint_size = 10
        self.liberado = False

    def stash(self):
        return {"position": self.position, "checkpoint_size": self.checkpoint_size, "origem": self.origem}

    def free(self):
        self.liberado = True

    def rewind(self, n):
        self.position -= n


class Tabela:
    def __init__(self):
        self.all_pages = []
        self.metrics = {}

    def is_resumable(self, h):
        return True

    pagina_parcial = PageTable.pagina_parcial


class Modelo:
    """Avanca o estado recorrente como um forward avancaria, e anota cada bloco (inicio, tamanho)."""
    caps = {}

    def __init__(self):
        self.blocos = []

    def prefill(self, input_ids, params):
        params["recurrent_states"][0].position += input_ids.shape[-1]
        self.blocos.append((int(params["cache_seqlens"][0]), input_ids.shape[-1]))


def _pagina(indice, kv_position = 0):
    return CachePage(
        pagetable = None, page_index = indice, phash = bytes([indice + 1]) * 16, phash_revert = bytes(16),
        prev_hash = None, prev_hash_revert = None, ref_count = 1, access_serial = 0,
        access_serial_revert = 0, kv_position = kv_position, kv_position_revert = 0,
        sequence = torch.zeros((1, PT), dtype = torch.long), can_revert = False, new_page_index = indice,
        children = [], longest_chain = 1,
    )


def _gerador(rc, mtp = False):
    copias = []
    cache = SimpleNamespace(
        copy_page = lambda alvo, de, para, n: copias.append((de, para, n)),
        new_from_stashed = lambda stashed, position: Estado(position, origem = stashed["origem"]),
    )
    gen = SimpleNamespace(
        recurrent_cache = rc, cache = cache, draft_model = None, draft_cache = None, mtp_draft = mtp,
        dflash_draft = False, page_tokens = PT, max_chunk_size = CHUNK, model = Modelo(),
        recurrent_checkpoint_interval = 8 * PT, recurrent_checkpoint_interval_pp = 128 * PT,
        ids_de_guarda = [USER],
    )
    return gen, copias


def _job(gen, tabela, prompt, paginas, estado, cached_pages = 0):
    ids = torch.tensor([prompt], dtype = torch.long)
    seq = Sequence(ids, ids)
    seq.allocated_pages = paginas
    seq.block_index_tensor = torch.zeros((1, len(paginas)), dtype = torch.int32)
    seq.kv_position = cached_pages * PT
    j = object.__new__(Job)
    j.generator = gen
    j.pagetable = tabela
    j.sequences = [seq]
    j.embeddings = []
    j.alt_rope_freqs = None
    j.recurrent_state = estado
    j.last_recurrent_checkpoint_pos = estado.position if cached_pages else None
    j.cached_pages = cached_pages
    j.cached_tokens = 0
    j.time_first_prefill = None
    j.identifier = None
    j.serial_number = 0
    j.rq_prompt_tokens = None
    j.mtp_last_hidden = None
    j.pontos_de_guarda_pedidos = []
    j.pontos_de_guarda = []
    j.retomar_bloco_ate = None
    return j, seq


def _prefill_ate_o_fim(j, seq, rc):
    """O laco do Generator.iterate: um bloco de prefill e os checkpoints de intervalo, ate acabar."""
    for _ in range(100):
        if j.is_prefill_done():
            return
        j.prefill([])
        j.maybe_stash_recurrent(rc)
    raise AssertionError("o prefill nao terminou")


def _primeira_vez(guarda = ()):
    """Um prompt de 10 paginas + 50 tokens, prefilado do zero."""
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    tabela = Tabela()
    rc.pagetable = tabela
    gen, copias = _gerador(rc)
    prompt = [(7 * i) % 1000 + 1 for i in range(10 * PT + 51)]        # fim do prefill em 10 * PT + 50
    paginas = [_pagina(i) for i in range(11)]
    tabela.all_pages.extend(paginas)
    j, seq = _job(gen, tabela, prompt, paginas, Estado(0))
    j.pontos_de_guarda = list(guarda)
    _prefill_ate_o_fim(j, seq, rc)
    return rc, tabela, gen, prompt, paginas


def test_ponto_de_guarda_corta_o_bloco_sem_deslocar_os_seguintes(config):
    config(EXL3_PONTOS_DE_GUARDA = 1)
    rc, tabela, gen, _, paginas = _primeira_vez(guarda = [3 * PT])

    assert gen.model.blocos == [
        (0, 3 * PT),            # corte no ponto de guarda
        (3 * PT, PT),           # o resto do bloco cortado: termina em 4 * PT, como terminaria
        (4 * PT, 4 * PT),
        (8 * PT, 2 * PT),       # ate a ultima fronteira do prompt
        (10 * PT, 50),          # a ultima pagina, parcial
    ]
    guardados = {v["position"] for v in rc.values()}
    # 3P: ponto de guarda; 8P: intervalo perto do fim (so cai ali porque os blocos nao se deslocaram);
    # 10P: ultima fronteira; 10P + 50: fim do prompt (parcial)
    assert guardados == {3 * PT, 8 * PT, 10 * PT, 10 * PT + 50}
    assert rc[paginas[2].phash]["position"] == 3 * PT
    assert not any(v.get("geracao") for v in rc.values())
    assert tabela.metrics["pontos_de_guarda"] == 1


def test_ponto_de_guarda_no_fim_de_um_bloco_tambem_e_guardado(config):
    # O ponto cai exatamente onde o bloco ja terminaria (4P, com blocos de 4P): nao ha corte, mas o
    # estado tem de ser guardado ali do mesmo jeito. Sob context parallel 4 (pagina de 1.024 tokens,
    # blocos de 2.048) metade dos pontos cai assim
    config(EXL3_PONTOS_DE_GUARDA = 1)
    rc, tabela, gen, _, paginas = _primeira_vez(guarda = [4 * PT])

    assert gen.model.blocos == [(0, 4 * PT), (4 * PT, 4 * PT), (8 * PT, 2 * PT), (10 * PT, 50)]
    assert rc[paginas[3].phash]["position"] == 4 * PT
    assert tabela.metrics["pontos_de_guarda"] == 1


def test_corte_nao_mexe_nas_paginas_alem_do_ponto(config):
    # As paginas depois do ponto ainda guardam o K/V de outra conversa com o mesmo comeco: o corte so
    # atualiza ate a pagina do ponto (e a seguinte, como qualquer bloco), nao ate o fim do bloco cortado
    config(EXL3_PONTOS_DE_GUARDA = 1)
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    tabela = Tabela()
    rc.pagetable = tabela
    gen, _ = _gerador(rc)
    prompt = [(7 * i) % 1000 + 1 for i in range(10 * PT + 51)]
    paginas = [_pagina(i, PT if i < 8 else 0) for i in range(11)]
    j, seq = _job(gen, tabela, prompt, paginas, Estado(0))
    j.pontos_de_guarda = [PT]
    j.prefill([])
    assert gen.model.blocos == [(0, PT)]
    assert paginas[1].kv_position == 0                    # a seguinte ao bloco, como sempre
    assert [p.kv_position for p in paginas[2:8]] == [PT] * 6


def _de_novo(rc, tabela, gen, prompt, paginas, mtp = False):
    """O mesmo prompt outra vez: as 10 paginas inteiras vem do cache, a parcial e uma pagina nova."""
    gen2, copias = _gerador(rc, mtp = mtp)
    for p in paginas[:10]:
        assert p.kv_position == PT
    destino = _pagina(20)
    destino.phash = b"\xee" * 16
    tabela.all_pages.append(destino)
    stash = rc.get_stashed(paginas[9].phash)
    assert stash["position"] == 10 * PT
    j, seq = _job(gen2, tabela, prompt, paginas[:10] + [destino], Estado(10 * PT, "pagina"), cached_pages = 10)
    return gen2, copias, j, seq, destino


def test_replay_do_prompt_identico_nao_roda_prefill(config):
    config(EXL3_REPLAY = 1)
    rc, tabela, gen, prompt, paginas = _primeira_vez()
    gen2, copias, j, seq, destino = _de_novo(rc, tabela, gen, prompt, paginas)

    _prefill_ate_o_fim(j, seq, rc)

    assert gen2.model.blocos == []                       # nenhum token de prefill
    assert copias == [(10, 20, 50)]                      # o K/V da pagina parcial, da primeira vez
    assert seq.kv_position == 10 * PT + 50 and seq.prefill_complete
    assert j.recurrent_state.position == 10 * PT + 50 and j.recurrent_state.origem == "vivo"
    assert j.cached_pages * PT + j.cached_tokens == 10 * PT + 50
    assert tabela.metrics["replays"] == 1
    # O passo de geracao recebe o ultimo token do prompt, como depois de um prefill
    assert j.get_input_ids_list()[0].tolist() == [prompt[-1:]]


def test_sem_replay_o_prompt_identico_refaz_a_ultima_pagina(config):
    rc, tabela, gen, prompt, paginas = _primeira_vez()
    gen2, copias, j, seq, _ = _de_novo(rc, tabela, gen, prompt, paginas)

    _prefill_ate_o_fim(j, seq, rc)

    assert gen2.model.blocos == [(10 * PT, 50)]
    assert copias == []
    assert "replays" not in tabela.metrics


def test_replay_com_mtp_exige_o_carry_e_o_entrega_ao_rascunho(config):
    config(EXL3_REPLAY = 1)
    rc, tabela, gen, prompt, paginas = _primeira_vez()
    # O parcial da primeira vez foi guardado sem rascunho: sem carry, o MTP nao pode pular tudo
    gen2, copias, j, seq, destino = _de_novo(rc, tabela, gen, prompt, paginas, mtp = True)
    ids = seq.sequence_ids.torch_slice(10 * PT, 10 * PT + 50)
    assert Job.reusar_pagina_parcial(j, seq, 10, ids, fim_do_prompt = True) == 0
    assert j.mtp_last_hidden is None

    # Com o carry guardado junto, pula tudo e o carry vira o hidden do primeiro rascunho
    carry = torch.full((1, 1, 8), 3.0)
    chave = chave_parcial(ids, paginas[9].phash)
    del rc[chave]                                        # o mesmo prompt, agora guardado com o carry
    rc.put(chave, Estado(10 * PT + 50, "com carry"),
           parcial = {"prev_hash": paginas[9].phash, "n": 50, "mtp_carry": carry})
    n = Job.reusar_pagina_parcial(j, seq, 10, ids, fim_do_prompt = True)
    assert n == 50 and copias == [(10, 20, 50)]
    assert torch.equal(j.mtp_last_hidden, carry) and j.mtp_last_hidden is seq.mtp_carry_hidden
    assert j.recurrent_state.origem == "com carry"


def test_replay_nao_pula_tudo_fora_do_fim_do_prompt(config):
    config(EXL3_REPLAY = 1)
    rc, tabela, gen, prompt, paginas = _primeira_vez()
    gen2, copias, j, seq, _ = _de_novo(rc, tabela, gen, prompt, paginas)
    ids = seq.sequence_ids.torch_slice(10 * PT, 10 * PT + 50)
    assert Job.reusar_pagina_parcial(j, seq, 10, ids, fim_do_prompt = False) == 0


def test_parcial_de_um_token_so_com_replay(config):
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    tabela = Tabela()
    rc.pagetable = tabela
    gen, _ = _gerador(rc)
    prompt = list(range(1, PT + 3))                       # fim do prefill em PT + 1: um token na pagina
    paginas = [_pagina(0, PT), _pagina(1, 1)]
    j, seq = _job(gen, tabela, prompt, paginas, Estado(PT + 1))
    seq.kv_position = PT + 1
    Job.maybe_stash_recurrent_parcial(j, seq)
    assert len(rc) == 0
    config(EXL3_REPLAY = 1)
    Job.maybe_stash_recurrent_parcial(j, seq)
    assert [v["parcial"]["n"] for v in rc.values()] == [1]


def _para_escolher(prompt, kv_cheias, cached_pages, embeddings = ()):
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    gen, _ = _gerador(rc)
    n_ctx = (len(prompt) - 1) // PT
    paginas = [_pagina(i, PT if i < kv_cheias else 0) for i in range(n_ctx + 1)]
    j, seq = _job(gen, Tabela(), prompt, paginas, Estado(cached_pages * PT), cached_pages = cached_pages)
    seq.page_hashes = [p.phash for p in paginas[:n_ctx]]
    j.embeddings = list(embeddings)
    return j, seq


def test_pontos_automaticos_fim_do_sistema_e_fim_do_prefixo_compartilhado(config):
    prompt = [5] * (20 * PT + 10)
    prompt[5 * PT + 10] = USER                            # o sistema termina aqui
    prompt[9 * PT] = USER                                 # so a primeira ocorrencia conta
    # 12 paginas com K/V de outra conversa, mas o estado recorrente so cobre 2
    j, seq = _para_escolher(prompt, kv_cheias = 12, cached_pages = 2)

    assert j.escolher_pontos_de_guarda(seq) == []         # desligado
    config(EXL3_PONTOS_DE_GUARDA = 1)
    assert j.escolher_pontos_de_guarda(seq) == [5 * PT, 12 * PT]
    j.embeddings = ["imagem"]
    assert j.escolher_pontos_de_guarda(seq) == []


def test_ponto_pedido_pelo_chamador_vale_sem_o_env(config):
    prompt = [5] * (20 * PT + 10)
    j, seq = _para_escolher(prompt, kv_cheias = 0, cached_pages = 0)
    j.pontos_de_guarda_pedidos = [7 * PT + 100]
    assert j.escolher_pontos_de_guarda(seq) == [7 * PT]


def test_lru_despeja_geracao_antes_do_prompt():
    def cache(proteger):
        rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 30)   # cabem 3
        rc.proteger_prompt = proteger
        rc.put(b"A" * 16, Estado(1))
        rc.put(b"B" * 16, Estado(2), geracao = True)
        rc.put(b"C" * 16, Estado(3))
        rc.put(b"D" * 16, Estado(4))
        return rc

    rc = cache(False)
    assert list(rc) == [b"B" * 16, b"C" * 16, b"D" * 16]       # antes: o mais antigo, seja qual for
    rc = cache(True)
    assert list(rc) == [b"A" * 16, b"C" * 16, b"D" * 16]
    assert rc.metrics["despejos_geracao"] == 1
    # Sem nenhum de geracao, volta a ser o mais antigo
    rc.put(b"E" * 16, Estado(5))
    assert list(rc) == [b"C" * 16, b"D" * 16, b"E" * 16]


def test_teto_de_quantidade():
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    rc.max_entradas = 2
    for i in range(4):
        rc.put(bytes([i + 1]) * 16, Estado(i))
    assert list(rc) == [bytes([3]) * 16, bytes([4]) * 16]
    assert rc.metrics["despejos_por_quantidade"] == 2


def test_checkpoint_classificado_pela_posicao(config):
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), max_size = 10**9)
    gen, _ = _gerador(rc)
    prompt = list(range(1, 2 * PT + 1))                   # fim do prompt em 2 * PT - 1
    paginas = [_pagina(i, PT) for i in range(4)]
    j, seq = _job(gen, Tabela(), prompt, paginas, Estado(2 * PT))
    seq.kv_position = 2 * PT
    j.maybe_stash_recurrent(rc, PT)
    assert rc[paginas[1].phash].get("geracao")            # depois do fim do prompt
    seq.kv_position = 3 * PT
    j.maybe_stash_recurrent(rc, PT, geracao = False)      # o do job que volta a fila
    assert not rc[paginas[2].phash].get("geracao")
    # O mesmo checkpoint, ja guardado como de geracao, guardado de novo como prompt: deixa de ser de geracao
    rc.put(paginas[1].phash, j.recurrent_state, geracao = False)
    assert not rc[paginas[1].phash].get("geracao")
