"""
O slot de staging tem de caber o LOTE de experts, e não um número redondo em bytes.

`_submit_prefill_streamed` transmite os experts para a GPU em lotes de
`min(wslot_size // expert_bytes, batch_experts)`. O lote declarado é 24 — o `moe_handoff.h`
diz isso em letra maiúscula — mas um slot fixo de 32 MB o corta para o que couber, **em
silêncio**.

Medido em 10/09/2026 no `keys-GLM-5.3-EXL3-Abliterated`: expert de 9 MB, `32 // 9 = 3`. O
motor rodava com 3 de 24, e cada lote paga um giro de flag mais dois `wait_event` — 1.776
apertos de mão por chunk de prefill em vez de 222.

O 32 não era arbitrário: foi calibrado num modelo de experts pequenos. É por isso que a
correção não é trocar 32 por 224 — é parar de fixar bytes e passar a fixar a INTENÇÃO.

Roda sem placa, sem modelo e sem a extensão: o método só faz aritmética sobre `self`.
"""
import os
import sys
import types

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

# Só a aritmética do método interessa; o módulo inteiro arrastaria a extensão CUDA por JIT.
_ns = {"_align64": lambda x: (x + 63) // 64 * 64}
fonte = open(os.path.join(RAIZ, "exllamav3", "model", "moe_cpu_host.py")).read()
ini = fonte.index("    def _dimensionar_wslot(self):")
fim = fonte.index("\n    def ", ini + 10)
exec("class _Host:\n" + fonte[ini:fim], _ns)
_Host = _ns["_Host"]

MB = 1024 * 1024


def _host(expert_mb, wslot_mb=32, explicito=False, teto_mb=256, lote=24, wslots=2):
    h = _Host()
    h.specs = [{"expert_bytes": int(expert_mb * MB)}]
    h.wslot_size = wslot_mb * MB
    h.wslot_max = teto_mb * MB
    h.wslot_explicit = explicito
    h.batch_experts = lote
    h.num_wslots = wslots
    return h


def test_cresce_para_caber_o_lote():
    """O caso que motivou tudo: expert de 9 MB, lote de 24, slot de 32 MB."""
    h = _host(9)
    h._dimensionar_wslot()
    assert h.wslot_size // (9 * MB) == 24, h.wslot_size / MB


def test_respeita_o_teto_porque_a_vram_e_expert():
    """Cada wslot custa duas cópias em VRAM, e VRAM aqui é expert que sai da placa."""
    h = _host(40, teto_mb=256)
    h._dimensionar_wslot()
    assert h.wslot_size <= 256 * MB, h.wslot_size / MB
    # Ainda assim tem de caber ao menos um expert, senão o streaming é pulado inteiro.
    assert h.wslot_size >= 40 * MB


def test_expert_maior_que_o_teto_ainda_cabe_um():
    """Sem isto o `spec['expert_bytes'] > wslot_size` desligaria o streaming em silêncio."""
    h = _host(400, teto_mb=256)
    h._dimensionar_wslot()
    assert h.wslot_size >= 400 * MB


def test_expert_pequeno_nao_encolhe_o_padrao():
    """Modelo de experts pequenos continua como estava: o padrão é PISO, não alvo."""
    h = _host(1)
    h._dimensionar_wslot()
    assert h.wslot_size == 32 * MB


def test_pedido_a_mao_vence():
    """Quem digitou EXL3_MOE_CPU_WSLOT_MB sabe o que quer — mas é avisado se cortar o lote."""
    h = _host(9, wslot_mb=32, explicito=True)
    h._dimensionar_wslot()
    assert h.wslot_size == 32 * MB


def test_sem_specs_nao_faz_nada():
    """Ordem de carga fora do esperado não pode derrubar o dimensionamento."""
    h = _host(9)
    h.specs = []
    h._dimensionar_wslot()
    assert h.wslot_size == 32 * MB


if __name__ == "__main__":
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
