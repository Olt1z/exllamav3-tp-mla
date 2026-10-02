"""
Rascunho por cópia (exllamav3/generator/copia.py): a busca do casamento e a montagem da janela.

Roda sem placa, sem torch e sem a extensão: o módulo é só numpy e é carregado pelo caminho, sem
passar pelo `exllamav3/__init__.py` (que exige torch). A integração com o gerador (MTP, DFlash2,
saída exata) está em `test_rascunho_copia_gpu_.py`, para a máquina de GPU.
"""
import importlib.util
import os
import random
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "copia_isolado", os.path.join(RAIZ, "exllamav3", "generator", "copia.py"))
copia = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = copia            # o dataclass procura o módulo dele aqui
_spec.loader.exec_module(copia)

CopiaIndice = copia.CopiaIndice
CopiaConfig = copia.CopiaConfig
combinar_janela = copia.combinar_janela


# ---- referência por força bruta (a semântica dos patches 0007/0032 do TensorFold)

def referencia(ctx, prompt, n, n_resposta, maximo, espaco = None):
    L = len(ctx)
    k = maximo if espaco is None else min(maximo, espaco)
    if k < 1 or L <= n:
        return []
    q = ctx[L - n:]
    extra = n_resposta - n if n_resposta > n else 0
    hits = []
    for s in range(0, L - n):
        if ctx[s:s + n] != q:
            continue
        if extra and s >= prompt:
            if s < extra or L - n - extra < 0:
                continue
            if any(ctx[s - i] != ctx[L - n - i] for i in range(1, extra + 1)):
                continue
        hits.append(s)
    if not hits:
        return []
    cheias = [s for s in hits if s <= L - n - k]
    s = cheias[-1] if cheias else hits[0]
    return ctx[s + n:min(s + n + k, L)]


def test_citacao_do_prompt():
    prompt = list(range(100, 140))
    ctx = prompt + [7, 8] + list(range(110, 118))       # a resposta cita 110..117 do prompt
    idx = CopiaIndice(ctx, prompt = len(prompt), casamento = 8, casamento_resposta = 16, maximo = 15)
    assert idx.propor() == list(range(118, 133))
    assert idx.propor(espaco = 4) == [118, 119, 120, 121]


def test_sem_casamento():
    ctx = list(range(50))
    idx = CopiaIndice(ctx, prompt = 40, casamento = 8)
    assert idx.propor() == []
    assert len(idx.ocorrencias()) == 0


def test_contexto_curto():
    idx = CopiaIndice([1, 2, 3], prompt = 3, casamento = 8)
    assert idx.propor() == []


def test_ocorrencia_mais_recente_com_espaco():
    bloco = [1, 2, 3, 4, 5, 6, 7, 8]
    ctx = bloco + [10, 11, 12, 13] + bloco + [20, 21, 22, 23] + [99] + bloco
    idx = CopiaIndice(ctx, prompt = len(ctx), casamento = 8, casamento_resposta = 0, maximo = 4)
    assert idx.propor() == [20, 21, 22, 23]                  # a mais recente com 4 depois


def test_sem_espaco_usa_a_mais_antiga():
    # Período curto: nenhuma ocorrência tem 6 tokens depois dela antes do fim, então vem a mais
    # antiga, que é a que tem mais
    ctx = [1, 2, 3] * 4
    idx = CopiaIndice(ctx, prompt = 0, casamento = 3, casamento_resposta = 0, maximo = 15)
    assert idx.propor() == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_copia_da_resposta_exige_casamento_maior():
    prompt = list(range(1000, 1050))
    trecho = [5, 6, 7, 8, 9, 10, 11, 12]                     # 8 tokens: casam o mínimo do prompt
    resposta = [1, 2, 3] + trecho + [40, 41, 42] + [77, 78] + trecho
    ctx = prompt + resposta
    # Ocorrência na resposta, mas só 8 tokens casam: com 16 exigidos, não serve
    idx = CopiaIndice(ctx, prompt = len(prompt), casamento = 8, casamento_resposta = 16)
    assert idx.propor() == []
    # Com o mesmo mínimo do prompt, serve
    idx = CopiaIndice(ctx, prompt = len(prompt), casamento = 8, casamento_resposta = 0)
    assert idx.propor(espaco = 3) == [40, 41, 42]
    # Com 16 tokens repetidos de fato, serve também com 16 exigidos
    longo = list(range(200, 216))
    ctx = prompt + longo + [50, 51, 52] + [9] + longo
    idx = CopiaIndice(ctx, prompt = len(prompt), casamento = 8, casamento_resposta = 16)
    assert idx.propor(espaco = 3) == [50, 51, 52]


def test_copia_do_prompt_com_8_mesmo_exigindo_16_na_resposta():
    trecho = [5, 6, 7, 8, 9, 10, 11, 12]
    prompt = [900, 901] + trecho + [30, 31, 32] + [903]
    ctx = prompt + [1, 2] + trecho
    idx = CopiaIndice(ctx, prompt = len(prompt), casamento = 8, casamento_resposta = 16)
    assert idx.propor(espaco = 3) == [30, 31, 32]


def test_teto_id_corta_marcadores():
    trecho = list(range(10, 18))
    ctx = trecho + [20, 21, 10 ** 9, 22] + [0] + trecho
    idx = CopiaIndice(ctx, prompt = len(ctx), casamento = 8, casamento_resposta = 0)
    assert idx.propor(espaco = 4, teto_id = 1000) == [20, 21]
    assert idx.propor(espaco = 4) == [20, 21, 10 ** 9, 22]


def test_estender_e_truncar_seguem_a_referencia():
    rng = random.Random(1234)
    for rodada in range(40):
        vocab = rng.choice([3, 5, 20])
        prompt = [rng.randrange(vocab) for _ in range(rng.randrange(0, 60))]
        n = rng.choice([2, 3, 4, 8])
        nr = rng.choice([0, n, n + 2, 16])
        maximo = rng.choice([1, 4, 15])
        idx = CopiaIndice(prompt, prompt = len(prompt), casamento = n, casamento_resposta = nr,
                          maximo = maximo)
        ctx = list(prompt)
        for passo in range(60):
            acao = rng.random()
            if acao < 0.15 and len(ctx) > 0:
                novo = rng.randrange(0, len(ctx) + 1)       # rebobina (às vezes para dentro da base)
                del ctx[novo:]
                idx.truncar(novo)
            else:
                novos = [rng.randrange(vocab) for _ in range(rng.randrange(1, 5))]
                ctx += novos
                idx.estender(novos)
            assert idx.tokens() == ctx
            espaco = rng.choice([None, 1, 3, 20])
            esperado = referencia(ctx, len(prompt), n, nr, maximo, espaco)
            assert idx.propor(espaco) == esperado, (rodada, passo)


def test_colisao_de_hash_nao_engana(monkeypatch):
    # Com todo n-grama no mesmo hash, a conferência dos tokens é que separa os candidatos
    monkeypatch.setattr(copia, "_hash_py", lambda toks: 0)
    monkeypatch.setattr(copia, "_hashes_np",
                        lambda ctx, n: np.zeros((max(0, ctx.shape[0] - n + 1),), dtype = np.uint64))
    rng = random.Random(7)
    for _ in range(20):
        ctx = [rng.randrange(4) for _ in range(rng.randrange(10, 80))]
        p = rng.randrange(0, len(ctx))
        idx = CopiaIndice(ctx[:p], prompt = p, casamento = 3, casamento_resposta = 5, maximo = 6)
        idx.estender(ctx[p:])
        assert idx.propor() == referencia(ctx, p, 3, 5, 6)


def test_hash_numpy_igual_ao_python():
    rng = np.random.default_rng(0)
    ctx = rng.integers(0, 200_000, size = 300).astype(np.int64)
    h = copia._hashes_np(ctx, 8)
    for s in (0, 1, 17, 292):
        assert int(h[s]) == copia._hash_py(ctx[s:s + 8].tolist())


def test_contexto_longo_rapido():
    # 400k tokens: a base indexa uma vez; cada proposta é uma busca binária, não uma varredura
    import time
    rng = np.random.default_rng(1)
    ctx = rng.integers(0, 150_000, size = 400_000)
    idx = CopiaIndice(ctx, prompt = ctx.shape[0], casamento = 8, casamento_resposta = 16)
    trecho = ctx[123_456:123_456 + 8].tolist()
    idx.estender(trecho)
    t0 = time.perf_counter()
    for _ in range(200):
        out = idx.propor()
    dt = (time.perf_counter() - t0) / 200
    assert out == ctx[123_464:123_464 + 15].tolist()
    assert dt < 2e-3, f"proposta levou {dt * 1e3:.2f} ms"



def test_balde_grande_segue_a_referencia():
    # Contexto periodico: o balde do sufixo passa do teto de candidatos, na base e no extra. Os
    # mais recentes ficam, e a proposta e a da referencia sem teto
    ctx = [1, 2, 3] * 3000
    for corte in (len(ctx), 400):                             # tudo na base / quase tudo no extra
        idx = CopiaIndice(ctx[:corte], prompt = len(ctx), casamento = 8, casamento_resposta = 0)
        idx.estender(ctx[corte:])
        assert len(idx.ocorrencias()) == copia._MAX_CANDIDATOS
        assert idx.propor() == referencia(ctx, len(ctx), 8, 0, 15)
        assert idx.propor(espaco = 4) == referencia(ctx, len(ctx), 8, 0, 15, 4)


def test_contexto_repetitivo_rapido():
    # 1M tokens iguais: o balde do sufixo tem 1M posicoes; a proposta nao pode passar por ele
    import time
    ctx = np.zeros(1_000_000, dtype = np.int64)
    idx = CopiaIndice(ctx, prompt = ctx.shape[0], casamento = 8, casamento_resposta = 16)
    idx.estender([0] * 64)
    t0 = time.perf_counter()
    for _ in range(50):
        out = idx.propor()
    dt = (time.perf_counter() - t0) / 50
    assert out == [0] * 15
    assert dt < 2e-3, f"proposta levou {dt * 1e3:.2f} ms"

# ---- configuração

def test_config_desligada_por_padrao():
    assert CopiaConfig.do_ambiente({}) is None
    assert CopiaConfig.do_ambiente({"EXL3_RASCUNHO_COPIA": "0"}) is None


def test_config_ligada_e_ajustes():
    c = CopiaConfig.do_ambiente({"EXL3_RASCUNHO_COPIA": "1"})
    assert (c.casamento, c.casamento_resposta, c.maximo, c.linhas_lote, c.passo) == (8, 16, 15, 16, 4)
    c = CopiaConfig.do_ambiente({"EXL3_RASCUNHO_COPIA": "1", "EXL3_COPIA_MAX": "7",
                                 "EXL3_COPIA_MIN_PROMPT": "6", "EXL3_COPIA_MIN_RESPOSTA": "0",
                                 "EXL3_COPIA_LINHAS_LOTE": "32", "EXL3_COPIA_PASSO": "1"})
    assert (c.casamento, c.casamento_resposta, c.maximo, c.linhas_lote, c.passo) == (6, 0, 7, 32, 1)


@pytest.mark.parametrize("env", [
    {"EXL3_RASCUNHO_COPIA": "sim"},
    {"EXL3_RASCUNHO_COPIA": "1", "EXL3_COPIA_MAX": "16"},       # passa da janela de decode da MLA
    {"EXL3_RASCUNHO_COPIA": "1", "EXL3_COPIA_MAX": "0"},
    {"EXL3_RASCUNHO_COPIA": "1", "EXL3_COPIA_MIN_PROMPT": "8", "EXL3_COPIA_MIN_RESPOSTA": "4"},
])
def test_config_recusa_valor_ruim(env):
    with pytest.raises(ValueError):
        CopiaConfig.do_ambiente(env)


def test_limite_por_job():
    c = CopiaConfig()
    assert c.limite_por_job(1, 3) == 15
    assert c.limite_por_job(2, 3) == 7
    assert c.limite_por_job(4, 3) == 3                       # quatro ou mais: a janela do MTP
    assert c.limite_por_job(16, 3) == 3
    assert c.limite_por_job(16, 15) == 15                    # DFlash2 já verifica 16 linhas
    assert c.limite_por_job(0, 3) == 0
    assert CopiaConfig(maximo = 5).limite_por_job(1, 3) == 5


# ---- janela do lote

def test_janela_todos_copiam():
    linhas, largura, usa = combinar_janela([[1, 2, 3, 4, 5], [7, 8]], [[], []], [100, 200], 15, passo = 1)
    assert largura == 5 and usa == [True, True]
    assert linhas == [[1, 2, 3, 4, 5], [7, 8, 8, 8, 8]]


def test_janela_mista_e_corte_no_limite():
    copias = [list(range(10, 25)), []]
    modelo = [[1, 2, 3], [4, 5, 6]]
    linhas, largura, usa = combinar_janela(copias, modelo, [0, 0], 7, passo = 1)
    assert largura == 7 and usa == [True, False]
    assert linhas[0] == list(range(10, 17))
    assert linhas[1] == [4, 5, 6, 6, 6, 6, 6]


def test_janela_arredondada_pelo_passo():
    # 5 rascunhos = 6 linhas, arredondadas para 8 (7 rascunhos), dentro do limite
    linhas, largura, _ = combinar_janela([[1, 2, 3, 4, 5]], [[]], [9], 15, passo = 4)
    assert largura == 7 and linhas == [[1, 2, 3, 4, 5, 5, 5]]
    # O limite manda: não passa de 6 rascunhos
    _, largura, _ = combinar_janela([[1, 2, 3, 4, 5]], [[]], [9], 6, passo = 4)
    assert largura == 6
    # Janela que a cópia não estendeu além do modelo não arredonda
    _, largura, _ = combinar_janela([[1, 2]], [[3, 4, 5]], [9], 15, passo = 4)
    assert largura == 2
    _, largura, _ = combinar_janela([[1, 2], []], [[], [3, 4, 5]], [9, 9], 15, passo = 4)
    assert largura == 3


def test_janela_linha_vazia_usa_o_pendente():
    linhas, largura, usa = combinar_janela([[1, 2, 3], []], [[], []], [50, 60], 15, passo = 1)
    assert largura == 3 and usa == [True, False]
    assert linhas[1] == [60, 60, 60]


def test_janela_sem_copia_igual_ao_modelo():
    linhas, largura, usa = combinar_janela([[], []], [[1, 2, 3], [4, 5, 6]], [0, 0], 15, passo = 4)
    assert largura == 3 and usa == [False, False] and linhas == [[1, 2, 3], [4, 5, 6]]
