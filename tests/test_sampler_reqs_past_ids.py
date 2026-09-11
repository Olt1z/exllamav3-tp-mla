"""
O sampler só pede o histórico se alguma etapa que SOBROU o usa.

Antes, `CustomSampler` somava os requisitos de todas as etapas e só depois
trocava as neutras por no-op: `SS_RepP(1.0)` e `SS_PresFreqP(0, 0)` pediam
`past_ids`, viravam no-op, e o job copiava a sequência inteira para a GPU a cada
passo — 8 bytes por token de contexto — para um kernel que nunca rodava. O
TabbyAPI empilha as duas em toda requisição (achado A5 da revisão externa de
11/09/2026).
"""

from exllamav3.generator.sampler.custom import (
    CustomSampler, SS_RepP, SS_PresFreqP, SS_Argmax, SS_Temperature, SS_Sample, SS_Sample_mn,
)


def test_penalidades_neutras_nao_pedem_historico():
    s = CustomSampler([SS_RepP(1.0, 1024, 0), SS_PresFreqP(0.0, 0.0, 1024, 0), SS_Argmax()])
    assert s.reqs_past_ids is False


def test_penalidade_ativa_pede_historico():
    s = CustomSampler([SS_RepP(1.1, 1024, 0), SS_Argmax()])
    assert s.reqs_past_ids is True


def test_semente_so_com_a_amostragem_que_a_usa():
    # `SS_Sample` tira o Gumbel do gerador próprio (`rand_u32`); só a variante
    # `_mn` passa pelo gerador do torch e precisa da semente dele.
    assert CustomSampler([SS_Argmax()]).reqs_torch_seed is False
    assert CustomSampler([SS_Temperature(0.7), SS_Sample()]).reqs_torch_seed is False
    assert CustomSampler([SS_Temperature(0.7), SS_Sample_mn()]).reqs_torch_seed is True
