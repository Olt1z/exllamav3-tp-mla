"""EXL3_MOE_CPU_SPLIT_STATS: ler o perfil de roteamento de volta, sem derrubar o carregamento.

    python tests/test_split_stats_in.py

O ciclo pretendido é: a primeira subida colhe o perfil (STATS_OUT) e as seguintes o usam
(STATS). Para isso o hub precisa poder emitir a variável SEMPRE, e o motor decidir sozinho —
mas `json.load(open(stats_path))` levantava `FileNotFoundError` na primeira subida, quando o
arquivo ainda não existe, e derrubava o carregamento no meio (é chamado por camada, 42 vezes
num GLM-5.3-Flash, reabrindo o arquivo a cada uma).
"""
import json, os, sys, tempfile, unittest


def _carregar():
    caminho = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "exllamav3", "modules", "block_sparse_mlp_cpu.py",
    )
    with open(caminho) as f:
        fonte = f.read()
    inicio = fonte.index("_perfil_de_roteamento: dict | None = None")
    fim = fonte.index("\ndef despejar_stats_de_roteamento", inicio)
    escopo = {"os": os, "json": json}
    exec(compile(fonte[inicio:fim], caminho, "exec"), escopo)
    return escopo["ler_perfil_de_roteamento"], escopo


class LeituraDoPerfil(unittest.TestCase):

    def setUp(self):
        self.ler, self.escopo = _carregar()
        self.escopo["_perfil_de_roteamento"] = None
        self.escopo["_perfil_lido_de"] = None
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.alvo = os.path.join(self.dir.name, "perfil.json")

    def escrever(self, dados):
        with open(self.alvo, "w") as f:
            json.dump(dados, f)

    def test_arquivo_ausente_nao_derruba_o_carregamento(self):
        """Primeira subida: o perfil ainda não existe, e o modelo tem de carregar assim mesmo."""
        self.assertIsNone(self.ler(self.alvo, "camada.0", 288))

    def test_perfil_valido_volta_como_lista(self):
        self.escrever({"camada.0": [3.0, 1.0, 2.0]})
        self.assertEqual(self.ler(self.alvo, "camada.0", 3), [3.0, 1.0, 2.0])

    def test_camada_ausente_no_perfil_nao_derruba(self):
        self.escrever({"outra": [1.0, 2.0, 3.0]})
        self.assertIsNone(self.ler(self.alvo, "camada.0", 3))

    def test_contagem_com_tamanho_errado_e_recusada(self):
        """Perfil de outro modelo: usar seria permutar experts pelo índice errado."""
        self.escrever({"camada.0": [1.0, 2.0]})
        self.assertIsNone(self.ler(self.alvo, "camada.0", 288))

    def test_json_corrompido_nao_derruba(self):
        with open(self.alvo, "w") as f:
            f.write("{ isto não é json")
        self.assertIsNone(self.ler(self.alvo, "camada.0", 3))

    def test_le_o_arquivo_uma_vez_so(self):
        """42 camadas reabrindo o mesmo JSON é desperdício no caminho de carga."""
        self.escrever({"c1": [1.0, 2.0], "c2": [2.0, 1.0]})
        n = [0]
        real = self.escopo["json"].load

        def contando(f):
            n[0] += 1
            return real(f)

        self.escopo["json"] = type("J", (), {"load": staticmethod(contando)})()
        self.assertEqual(self.ler(self.alvo, "c1", 2), [1.0, 2.0])
        self.assertEqual(self.ler(self.alvo, "c2", 2), [2.0, 1.0])
        self.assertEqual(n[0], 1, "o perfil tem de ficar em cache entre camadas")

    def test_caminho_vazio_devolve_nada(self):
        self.assertIsNone(self.ler(None, "c", 3))
        self.assertIsNone(self.ler("", "c", 3))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
