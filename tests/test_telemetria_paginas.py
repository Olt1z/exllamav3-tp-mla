"""
A linha de páginas do job (`[exl3_tel job] paginas: ...`) sai no stderr com
EXL3_TEL=1 e não sai sem. Roda na máquina, com a extensão compilada:

    EXL3_TEL=1 EXL3_TEL_NVTX=0 python -m pytest tests/test_telemetria_paginas.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EXL3_TEL", "0") == "0",
    reason = "precisa de EXL3_TEL=1 no ambiente",
)


def test_linha_de_paginas(capsys):
    from exllamav3.util import telemetria as tel
    tel.paginas_do_job(330_000, 1290, 1281, 3)
    err = capsys.readouterr().err
    assert err == "[exl3_tel job] paginas: prompt 330000 tokens, 1290 paginas, 1281 do cache, 3 nao sequenciais\n"
