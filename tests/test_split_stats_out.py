"""EXL3_MOE_CPU_SPLIT_STATS_OUT: o perfil de roteamento que a colocação estática lê de volta.

    python tests/test_split_stats_out.py

Sem GPU, sem modelo e sem dependência: roda em qualquer máquina, inclusive na do dono, que é
onde o defeito seria notado primeiro. O lado que LÊ (`EXL3_MOE_CPU_SPLIT_STATS`) existia desde
sempre; o que ESCREVE não, então a colocação por frequência só podia usar contagens feitas à
mão. Estes testes prendem o formato (contagens por id de ROTEADOR, na ordem do checkpoint), a
soma da janela viva e a atomicidade da escrita.
"""
import json, os, sys, tempfile, types, unittest


def _carregar_funcao():
    """Importa só a função, sem arrastar a extensão C++ do pacote (que exige GPU e build)."""
    caminho = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "exllamav3", "modules", "block_sparse_mlp_cpu.py",
    )
    with open(caminho) as f:
        fonte = f.read()
    inicio = fonte.index("def despejar_stats_de_roteamento(reg):")
    fim = fonte.index("\nclass BlockSparseMLP_CPU", inicio)
    escopo = {"os": os, "json": json}
    exec(compile(fonte[inicio:fim], caminho, "exec"), escopo)
    return escopo["despejar_stats_de_roteamento"]


despejar = _carregar_funcao()


class Hist:
    """O mínimo de um tensor que a função toca: somar dois e virar lista."""
    def __init__(self, vs): self.vs = list(vs)
    def __add__(self, o): return Hist([a + b for a, b in zip(self.vs, o.vs)])
    def tolist(self): return list(self.vs)


def camada(key, total, vivo = None):
    return types.SimpleNamespace(
        key = key,
        _split_hist_total = Hist(total),
        _split_hist = None if vivo is None else Hist(vivo),
    )


class DespejoDeStats(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.alvo = os.path.join(self.dir.name, "perfil.json")
        os.environ.pop("EXL3_MOE_CPU_SPLIT_STATS_OUT", None)
        self.addCleanup(lambda: os.environ.pop("EXL3_MOE_CPU_SPLIT_STATS_OUT", None))

    def ligar(self, caminho = None):
        os.environ["EXL3_MOE_CPU_SPLIT_STATS_OUT"] = caminho or self.alvo

    def ler(self, cru = False):
        with open(self.alvo) as f:
            texto = f.read()
        return texto if cru else json.loads(texto)

    def test_sem_a_variavel_nao_escreve_nada(self):
        despejar([camada("model.layers.0.mlp", [1.0, 2.0])])
        self.assertEqual(os.listdir(self.dir.name), [])

    def test_formato_e_o_que_o_lado_da_leitura_espera(self):
        """`EXL3_MOE_CPU_SPLIT_STATS` faz `json.load(...).get(self.key)` e indexa por id de expert."""
        self.ligar()
        despejar([camada("model.layers.3.mlp", [5.0, 1.0, 9.0]),
                  camada("model.layers.4.mlp", [0.0, 7.0, 2.0])])
        lido = self.ler()
        self.assertEqual(set(lido), {"model.layers.3.mlp", "model.layers.4.mlp"})
        self.assertEqual(lido["model.layers.3.mlp"], [5.0, 1.0, 9.0])
        # É assim que o lado da leitura ordena: o mais quente primeiro, a cauda fria vai para a CPU
        counts = lido["model.layers.3.mlp"]
        self.assertEqual(sorted(range(3), key = lambda e: -counts[e]), [2, 0, 1])

    def test_a_janela_viva_entra_na_conta(self):
        """O sweep só soma ao total quando roda até o fim; perder a janela distorce mais."""
        self.ligar()
        despejar([camada("l", [10.0, 0.0], vivo = [1.0, 4.0])])
        self.assertEqual(self.ler()["l"], [11.0, 4.0])

    def test_camada_sem_histograma_nao_entra(self):
        """Swap desligado: sem histograma, e uma camada sem perfil não pode virar zeros —
        zeros seriam lidos depois como 'nunca usada' e mandariam experts quentes para a RAM."""
        self.ligar()
        sem = types.SimpleNamespace(key = "sem", _split_hist_total = None, _split_hist = None)
        despejar([camada("com", [3.0, 1.0]), sem])
        self.assertEqual(list(self.ler()), ["com"])

    def test_nenhuma_camada_com_perfil_nao_cria_arquivo(self):
        self.ligar()
        despejar([types.SimpleNamespace(key = "x", _split_hist_total = None, _split_hist = None)])
        self.assertFalse(os.path.exists(self.alvo))

    def test_escrita_e_atomica(self):
        """Máquina alugada morre sem desligar limpo: uma falha no meio da escrita não pode
        deixar JSON truncado no lugar do perfil anterior."""
        self.ligar()
        despejar([camada("l", [1.0, 2.0])])
        anterior = self.ler(cru = True)

        real = os.replace
        os.replace = lambda *a, **k: (_ for _ in ()).throw(OSError("disco cheio"))
        try:
            despejar([camada("l", [9.0, 9.0])])
        finally:
            os.replace = real
        self.assertEqual(self.ler(cru = True), anterior, "o perfil anterior tem de sobreviver")

    def test_erro_de_disco_nao_derruba_a_geracao(self):
        """O despejo roda no hook do gerador: exceção aqui mataria a requisição do usuário."""
        self.ligar(os.path.join(self.dir.name, "sem", "caminho", "p.json"))
        despejar([camada("l", [1.0])])  # não levanta


if __name__ == "__main__":
    unittest.main(verbosity = 2)
