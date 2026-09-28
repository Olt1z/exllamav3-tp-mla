"""
So as imagens cujos ids aparecem no passo vao nos params do forward.

Sob tensor parallel cada embedding passado e serializado para todos os ranks a cada forward. Em 28/09/2026
(i5v7s, GLM-5.3-Flash, TP4) duas imagens no contexto viravam 68 MB por passo de decode, sem que o decode
tivesse um id de imagem sequer. Sem GPU nem modelo: o filtro olha so os ids, na CPU.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX, embeddings_presentes


class Imagem:
    def __init__(self, first_index, mm_length):
        self.first_index = first_index
        self.mm_length = mm_length


A = Imagem(FIRST_MM_EMBEDDING_INDEX, 2640)
B = Imagem(FIRST_MM_EMBEDDING_INDEX + 2640, 2940)


class TestEmbeddingsPresentes(unittest.TestCase):

    def test_decode_nao_leva_imagem_nenhuma(self):
        # O token amostrado mais o bloco do rascunho: tudo do vocabulario
        ids = torch.tensor([[151329, 13, 2048, 9, 77, 1, 0, 42]])
        self.assertEqual(embeddings_presentes(ids, [A, B]), [])

    def test_bloco_de_prefill_leva_so_a_imagem_que_cai_nele(self):
        ids = torch.tensor([[5, 6, B.first_index + 10, B.first_index + 11, 7]])
        self.assertEqual(embeddings_presentes(ids, [A, B]), [B])

    def test_bloco_com_as_duas_mantem_a_ordem(self):
        ids = torch.tensor([[A.first_index + A.mm_length - 1, 3, B.first_index]])
        self.assertEqual(embeddings_presentes(ids, [A, B]), [A, B])

    def test_fronteira_da_faixa(self):
        # O id logo depois da ultima linha de A e a primeira de B, nao de A
        ids = torch.tensor([[A.first_index + A.mm_length]])
        self.assertEqual(embeddings_presentes(ids, [A, B]), [B])

    def test_sem_imagem_na_conversa_nada_muda(self):
        ids = torch.tensor([[1, 2, 3]])
        self.assertIsNone(embeddings_presentes(ids, None))
        self.assertEqual(embeddings_presentes(ids, []), [])


if __name__ == "__main__":
    unittest.main()
