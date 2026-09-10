"""
O reprodutor de traço tem de DETECTAR o ganho da colocação por popularidade — e tem de não
inventar um quando ele não existe.

Ele é a ferramenta que vai decidir onde os experts moram sem alugar máquina, então um erro
aqui não aparece como travamento: aparece como uma decisão de arquitetura errada, barata de
tomar e cara de desfazer. Os dois casos abaixo são os extremos que cercam qualquer traço
real, e cada um mata um jeito diferente de a ferramenta mentir:

  concentrado   os experts quentes em índices SORTEADOS, como num checkpoint de verdade
                (o id não tem relação com o uso). `índice` tem de sofrer e `popularidade`
                tem de salvar. Se as duas colunas empatarem aqui, a ferramenta não mede o
                que existe para medir.
  achatado      uso uniforme, que é o modelo do Kimi K3 com Quantile Balancing. Nenhuma
                colocação pode ganhar da outra, e `popularidade` prometendo vantagem seria
                ruído lido como sinal — o erro que o próprio kimi-k3-in-c documenta.

Roda sem placa e sem modelo.
"""
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FERRAMENTA = os.path.join(RAIZ, "tests", "bancada", "reproduzir_traco.py")

CAMADAS, EXPERTS, TOP_K, TOKENS = 8, 64, 8, 400


def _escrever(caminho, escolher):
    pares = []
    for _ in range(TOKENS):
        for c in range(CAMADAS):
            for e in escolher(c):
                pares.append((c, e))
    np.array(pares, dtype=np.int32).tofile(caminho)


def _concentrado(rng):
    """Top 12,5 % dos experts levam ~50 % das seleções, em índices sorteados por camada."""
    quentes = {c: rng.sample(range(EXPERTS), EXPERTS // 8) for c in range(CAMADAS)}
    def escolher(c):
        return [rng.choice(quentes[c]) if rng.random() < 0.5 else rng.randrange(EXPERTS)
                for _ in range(TOP_K)]
    return escolher


def _achatado(rng):
    return lambda c: [rng.randrange(EXPERTS) for _ in range(TOP_K)]


def _rodar(caminho):
    saida = subprocess.run(
        [sys.executable, FERRAMENTA, caminho, "--top-k", str(TOP_K), "--fracoes", "0.5"],
        capture_output=True, text=True, check=True).stdout
    linha = [l for l in saida.splitlines() if l.startswith("50%")]
    assert linha, saida
    campos = linha[0].split()
    return {"indice": float(campos[1].rstrip("%")), "dinamica": float(campos[2].rstrip("%")),
            "popularidade": float(campos[3].rstrip("%")), "otimo": float(campos[4].rstrip("%"))}


def test_uso_concentrado_a_popularidade_ganha():
    rng = random.Random(11)
    with tempfile.TemporaryDirectory() as d:
        alvo = os.path.join(d, "t.bin")
        _escrever(alvo, _concentrado(rng))
        r = _rodar(alvo)
        # Metade dos experts na CPU: por índice isso arrasta ~metade das seleções.
        assert 40 < r["indice"] < 60, r
        # Por popularidade, os 50 % mais frios carregam MUITO menos que metade.
        assert r["popularidade"] < r["indice"] * 0.75, r
        # E o teto do futuro conhecido fica abaixo ainda, sem ser absurdo.
        assert r["otimo"] <= r["popularidade"] + 0.5, r


def test_uso_achatado_ninguem_ganha():
    """Com uso uniforme não há o que colocar melhor, e prometer ganho seria ler ruído."""
    rng = random.Random(23)
    with tempfile.TemporaryDirectory() as d:
        alvo = os.path.join(d, "t.bin")
        _escrever(alvo, _achatado(rng))
        r = _rodar(alvo)
        assert abs(r["popularidade"] - r["indice"]) < 3.0, r


def test_dinamica_parte_do_indice_e_nao_piora():
    """A varredura do fork começa na identidade; ela pode não achar nada, nunca regredir."""
    rng = random.Random(11)
    with tempfile.TemporaryDirectory() as d:
        alvo = os.path.join(d, "t.bin")
        _escrever(alvo, _concentrado(rng))
        r = _rodar(alvo)
        assert r["dinamica"] <= r["indice"] + 1.0, r


if __name__ == "__main__":
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_"):
            fn()
            print(f"ok  {nome}")
