#pragma once

#include <cstdint>
#include <string>

/*
Telemetria fina do passo de decode: anel em memória, despejo só quando dói.

Por que um anel e não log direto: o que se quer ver é o passo LENTO, e ele é
raro. Medido em 11/09/2026 na v6rj9 (1x H200, DSV4-Flash a 1M): de 101
requisições, 5 renderam metade da velocidade das outras 96. Logar os 96 casos
normais para pegar os 5 enche o disco e muda o que se está medindo — um passo
de decode inteiro dura ~40 ms, e escrever em disco no meio dele é ruído maior
que o sinal.

Então: os eventos entram num anel em memória (custo de um push), e só quando o
passo passa do limiar o anel é despejado. Os passos normais não deixam rastro.

Desligado por padrão. `EXL3_TEL=1` liga.

Variáveis:
  EXL3_TEL            1 liga (padrão 0)
  EXL3_TEL_RING       eventos guardados no anel (padrão 4096)
  EXL3_TEL_LIMIAR_MS  despeja o anel quando o passo passa disto (padrão 0 =
                      automático, 2x a média móvel dos últimos passos)
  EXL3_TEL_CONTEXTO   passos ANTERIORES ao lento que entram no despejo
                      (padrão 2; 0 com EXL3_TEL_MODULOS)
  EXL3_TEL_HISTOGRAMA de quantos em quantos passos o histograma sai (padrão 1000;
                      sai também a cada 30 s)
  EXL3_TEL_MODULOS    1 grava um marco por MÓDULO do forward no anel (padrão 0)

O histograma tem duas séries: a DURAÇÃO do passo e o INTERVALO do fim de um
passo ao início do seguinte. A segunda é o que o anel não vê por dentro —
sampler, servidor, event loop, e o forward do rascunho quando há um — e é a
que diz se o tempo está fora do forward.

O spdlog é opcional em tempo de COMPILAÇÃO. Sem o header, tudo aqui vira
no-op e a extensão compila igual: a telemetria é acessória, e uma máquina que
não compila por falta dela seria um preço absurdo. Ver `tel_disponivel()`.
*/

/*
spdlog é OPT-IN, e não detectado.

A detecção por `__has_include` parecia segura e quebrou o build inteiro numa
máquina de verdade (11/09/2026): o header do Ubuntu 24.04 existe — spdlog
1.12 — e não compila dentro desta extensão, porque o torch 2.9 expõe `fmt`
v11 no mesmo include path e o spdlog 1.12 foi escrito para fmt 9/10:

    /usr/include/spdlog/common.h:373: error:
        'basic_format_string' is not a member of 'fmt'

"Existe" e "compila junto com o torch" não são a mesma pergunta, e a detecção
só respondia a primeira. Agora o spdlog só entra com `EXL3_COM_SPDLOG=1`
passado ao build, por quem tiver uma combinação que fecha; o padrão é o anel
próprio abaixo, que faz o mesmo com `fprintf` e não depende de ninguém.
*/
#if defined(EXL3_COM_SPDLOG) && defined(__has_include)
    #if __has_include(<spdlog/spdlog.h>)
        #define EXL3_TEM_SPDLOG 1
    #endif
#endif

namespace exl3_tel
{
    /// Ligada por `EXL3_TEL=1`. Lida uma vez.
    bool ativa();

    /// `EXL3_TEL_MODULOS=1` com a telemetria ligada: o laço de módulos grava
    /// um marco por módulo no anel. Lida uma vez; o chamador guarda.
    bool marca_modulos();

    /// A extensão foi compilada COM spdlog? Serve para o Python não prometer
    /// traço que esta build não sabe produzir.
    bool compilada_com_spdlog();

    /// Abre um passo de decode. Zera o relógio; não zera o anel — o anel é
    /// circular de propósito, para que o despejo mostre também o que veio
    /// ANTES do passo lento.
    void passo_inicio();

    /// Fecha o passo. Se durou mais que o limiar, despeja o anel.
    /// Devolve a duração em milissegundos, ou -1 com a telemetria desligada.
    double passo_fim();

    /// Um evento dentro do passo. `nome` tem de sobreviver à chamada
    /// (literal de string, na prática): o anel guarda o ponteiro, não uma cópia.
    void evento(const char* nome);

    /**
     * Despeja o anel agora, sem esperar limiar. Para o `POST /traco` do hub.
     *
     * `duracao_ms` negativa marca o despejo como "sob demanda" no cabeçalho:
     * fora de um passo não há duração que signifique alguma coisa, e imprimir
     * a do passo anterior faria o hub gravar um passo lento que não existiu.
     */
    void despejar(double duracao_ms = -1.0);
}

/*
A macro é o que se usa nos kernels.

Com a telemetria desligada o custo é a leitura de um bool — o `if` vem ANTES da
chamada de propósito, para que o caso normal não pague nem o salto para a
função.
*/
#define TEL_EVENTO(nome)                          \
    do {                                          \
        if (exl3_tel::ativa()) exl3_tel::evento(nome); \
    } while (0)
