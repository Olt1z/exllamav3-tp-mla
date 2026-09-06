"""
Isola a exportação/importação TP de cada bloco do modelo, no mesmo processo e numa placa só.

Para cada TransformerBlock: forward original × forward do bloco reconstruído por
tp_export -> tp_import com o plano cheio (todos os canais em um rank) × soma de dois blocos
importados com metade dos canais cada (o que o all-reduce faria com 2 ranks). Depois o mesmo
para o filho `mlp` sozinho. O plano sai do próprio make_tp_allocation do bloco, então o teste
vale para qualquer arquitetura.

    python3 tests/test_tp_block_import.py /workspace/corte
    python3 tests/test_tp_block_import.py /workspace/modelo-inteiro --autosplit   # não cabe numa placa

Com `--autosplit` o modelo é repartido por camadas entre todas as placas (o carregador padrão), e
cada bloco é exportado e reimportado na placa onde já vive: serve para o modelo inteiro, como o
TR3 do Flash (163 GiB em 4× 96 GB), sem precisar de corte.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.model.model_tp_shared import SMProducer, SMConsumer

torch.manual_seed(0)
config = Config.from_directory(sys.argv[1])
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 4096)
autosplit = "--autosplit" in sys.argv[2:]
if autosplit:
    model.load(progressbar = True)
else:
    model.load(device = "cuda:0")
dev = 0

def dif(a, b):
    a, b = a.float(), b.float()
    return f"|dif| máx {(a - b).abs().max():.5f}  rel {(a - b).norm() / (a.norm() + 1e-9):.2e}"

def planos(mod):
    comps = mod.make_tp_allocation({})
    cheio, a, b = {}, {}, {}
    for c in comps:
        n = c.channels_to_split * (c.channel_width or 1)
        u = c.channel_unit
        cheio[c.key] = (0, n, u)
        meio = (c.channels_to_split // 2) * (c.channel_width or 1)
        a[c.key] = (0, meio, u)
        b[c.key] = (meio, n, u)
    return cheio, a, b

producer = SMProducer()
consumidores = {}

def contexto(d):
    """Consumidor e contexto local da placa `d`, criados uma vez por placa."""
    if d not in consumidores:
        consumidores[d] = SMConsumer(producer, device = d)
    return {"device": d, "consumer": consumidores[d], "output_device": d, "recurrent_modules": []}

class BackendDeUmRank:
    """Um rank só: difusão e all-reduce são identidade."""
    def broadcast(self, tensor, src_device = None): pass
    def all_reduce(self, tensor, contribution = True): pass

params = lambda: {"attn_mode": "flash_attn_nc", "causal": True, "position": 0, "backend": BackendDeUmRank()}

_im = torch.inference_mode()   # os kernels da MoE atualizam buffers in-place e exigem inference_mode
_im.__enter__()                # a referência fica viva; um temporário sairia do modo na hora
for i, blk in enumerate(model.modules):
    if not hasattr(blk, "attn") or not hasattr(blk, "mlp"):
        continue
    d = blk.device
    dev = d.index if isinstance(d, torch.device) else (0 if d is None else int(d))
    local = contexto(dev)
    x = torch.randn(1, 8, config.hidden_size, dtype = torch.half, device = dev)
    ref = blk.forward(x.clone(), params())
    cheio, pa, pb = planos(blk)
    exported = blk.tp_export(cheio, producer)
    cls = type(blk)
    b_full = cls.tp_import(local, exported, cheio)
    b_a = cls.tp_import(local, exported, pa)
    b_b = cls.tp_import(local, exported, pb)
    for m in (b_full, b_a, b_b):
        # sem all-reduce: somamos à mão
        for sub in ("attn", "mlp"):
            s = getattr(m, sub, None)
            if s is not None and hasattr(s, "tp_reduce"):
                s.tp_reduce = False
        if hasattr(m, "tp_reduce"):
            m.tp_reduce = False
    out_full = b_full.forward(x.clone(), params())
    print(f"\n== bloco {i} ({blk.key}) attn={type(blk.attn).__name__} mlp={type(blk.mlp).__name__}")
    print("  bloco importado cheio vs original:", dif(out_full, ref))
    # A soma das metades só vale para os filhos (o bloco soma o residual duas vezes)
    for sub in ("attn", "mlp"):
        o, f, a, b = getattr(blk, sub), getattr(b_full, sub), getattr(b_a, sub), getattr(b_b, sub)
        if o is None:
            continue
        xi = torch.randn(1, 8, config.hidden_size, dtype = torch.half, device = dev)
        r = o.forward(xi.clone(), params())
        print(f"  {sub}: importado cheio vs original:", dif(f.forward(xi.clone(), params()), r))
        print(f"  {sub}: soma das metades vs original:", dif(a.forward(xi.clone(), params()) + b.forward(xi.clone(), params()), r))
    producer.clear()
    del b_full, b_a, b_b, exported
    torch.cuda.empty_cache()   # as três cópias do bloco não podem se acumular no modelo inteiro
print("FIM_TESTE")
