#!/usr/bin/env python3
"""A bancada do TP na MLAttention, pela API da Vast: buscar, comprar, acompanhar, destruir.

    python3 vast-bancada.py buscar [--placas 2] [--disco 60] [--placa "RTX 3090"]
    python3 vast-bancada.py comprar <offer_id> [--placas 2] [--disco 60]
    python3 vast-bancada.py log <instance_id> [--linhas 200]
    python3 vast-bancada.py destruir <instance_id>

Nada é comprado sem o dono: `buscar` só lista; `comprar` exige o id da oferta que ele escolheu.
A instância sobe a imagem do PyTorch com nvcc e roda `onstart-bancada.sh`. O token do HF e a
chave da Vast vêm do .env da API do hub (bl4ck0ut-hub/apps/api/.env, pasta irmã deste fork).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parents[3] / "bl4ck0ut-hub/apps/api/.env"
IMAGEM = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"


def variavel(nome: str) -> str:
    m = re.search(rf"^{nome}=(.*)$", ENV.read_text(), re.M)
    if not m:
        raise SystemExit(f"{nome} ausente em {ENV}")
    return m.group(1).strip().strip("'\"")


def vast(caminho: str, metodo: str = "GET", corpo: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0{caminho}",
        data=json.dumps(corpo).encode() if corpo is not None else None,
        method=metodo,
        headers={"Authorization": f"Bearer {variavel('VAST_API_KEY')}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def buscar(placas: int, disco: int, placa: str | None, vram: int = 22, ram: int = 0) -> list[dict]:
    q = {
        "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False}, "type": "on-demand",
        "num_gpus": {"eq": placas}, "gpu_ram": {"gte": vram * 1024}, "disk_space": {"gte": disco},
        "allocated_storage": disco, "inet_down": {"gte": 500}, "cuda_max_good": {"gte": 12.8},
        "reliability2": {"gte": 0.9}, "gpu_frac": {"gte": 0.5},
        "order": [["dph_total", "asc"]], "limit": 40,
    }
    if placa:
        q["gpu_name"] = {"eq": placa}
    if ram:
        q["cpu_ram"] = {"gte": ram * 1024}
    return vast("/bundles/", "POST", q).get("offers", [])


def cmd_buscar(a: argparse.Namespace) -> None:
    ofertas = buscar(a.placas, a.disco, a.placa, a.vram, a.ram)
    print(f"{len(ofertas)} ofertas · {a.placas} placas · disco ≥ {a.disco} GB · VRAM ≥ {a.vram} GiB por placa")
    print(f"{'oferta':>10}  {'placas':<26} {'$/h':>6}  {'rede Mb/s':>9}  {'disco':>6}  {'conf':>5}  {'RAM GB':>6}  cpu · região")
    for o in ofertas[:25]:
        print(f"{o['id']:>10}  {o['num_gpus']}× {o['gpu_name']:<22} {o['dph_total']:>6.2f}  {o.get('inet_down', 0):>9.0f}  {o.get('disk_space', 0):>6.0f}  {o.get('reliability2', 0):>5.2f}  {(o.get('cpu_ram') or 0) / 1024:>6.0f}  {(o.get('cpu_name') or '')[:26]} · {o.get('geolocation')}")


def cmd_comprar(a: argparse.Namespace) -> None:
    onstart = (Path(__file__).parent / "onstart-bancada.sh").read_text()
    prova_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    env = " ".join(
        [
            f"-e HF_TOKEN={variavel('HF_TOKEN')}",
            f"-e PROVA_ID={prova_id}",
            f"-e FORK={a.fork}",
            f"-e BRANCH={a.branch}",
            f"-e CORTE={a.corte}",
            f"-e BASE_DEVICES={a.base_devices}",
            f"-e TOKENS={a.tokens}",
        ]
        + [f"-e {kv}" for kv in (a.env or [])]
    )
    corpo = {
        "image": IMAGEM, "disk": a.disco, "label": f"bl4ck0ut bancada tp-mla {prova_id}", "env": env,
        "runtype": "ssh", "onstart": onstart, "target_state": "running", "cancel_unavail": True,
    }
    r = vast(f"/asks/{a.oferta}/", "PUT", corpo)
    print(json.dumps(r, indent=1))
    print(f"bancada {prova_id} · instância {r.get('new_contract')} · resultado em saidas/tp-mla/{prova_id}/")


def cmd_log(a: argparse.Namespace) -> None:
    r = vast(f"/instances/request_logs/{a.instancia}/", "PUT", {"tail": str(a.linhas)})
    url = r.get("result_url")
    if not url:
        raise SystemExit(f"sem result_url: {r}")
    for _ in range(10):
        time.sleep(3)
        try:
            with urllib.request.urlopen(url, timeout=30) as f:
                print(f.read().decode(errors="replace"))
                return
        except Exception:
            continue
    raise SystemExit("o log não ficou disponível")


def cmd_destruir(a: argparse.Namespace) -> None:
    print(json.dumps(vast(f"/instances/{a.instancia}/", "DELETE"), indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("buscar"); b.add_argument("--placas", type=int, default=2); b.add_argument("--disco", type=int, default=60)
    b.add_argument("--placa", default=None); b.add_argument("--vram", type=int, default=22)
    b.add_argument("--ram", type=int, default=0, help="RAM mínima do host em GB (experts na RAM)")
    c = sub.add_parser("comprar"); c.add_argument("oferta", type=int); c.add_argument("--disco", type=int, default=60)
    c.add_argument("--fork", default="https://github.com/Olt1z/exllamav3-tp-mla"); c.add_argument("--branch", default="tp-mla")
    c.add_argument("--corte", default="Olt1z/GLM-5.3-podado-4L-EXL3-balanced-bl4ck0ut")
    c.add_argument("--base-devices", default="0", help="'0' para corte; 'all' para modelo grande (base em autosplit)")
    c.add_argument("--tokens", type=int, default=64)
    c.add_argument("--env", action="append", help="variável extra para o onstart, CHAVE=valor (repetível; ex.: SO_7=1)")
    lg = sub.add_parser("log"); lg.add_argument("instancia", type=int); lg.add_argument("--linhas", type=int, default=200)
    d = sub.add_parser("destruir"); d.add_argument("instancia", type=int)
    a = p.parse_args()
    {"buscar": cmd_buscar, "comprar": cmd_comprar, "log": cmd_log, "destruir": cmd_destruir}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
