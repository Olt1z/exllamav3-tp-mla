/*
Autoteste da telemetria, fora do build da extensão.

Compila e roda sozinho:

    g++ -std=c++17 -DEXL3_TESTE_TELEMETRIA \
        exllamav3/exllamav3_ext/telemetria_autoteste.cpp \
        exllamav3/exllamav3_ext/telemetria.cpp -o /tmp/tel && /tmp/tel

Fica fora do `bindings.cpp` porque o `setup.py` varre a pasta e compila todo
`.cpp` que encontra: sem o `#ifdef`, este `main` entraria na extensão e brigaria
com o do Python.
*/

#ifdef EXL3_TESTE_TELEMETRIA

#include "telemetria.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <unistd.h>

static void dorme_ms(int ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main()
{
    const bool filho = std::getenv("EXL3_TESTE_FILHO") != nullptr;

    if (!filho)
    {
        /// Sem `EXL3_TEL` a telemetria fica desligada e tudo vira no-op: é o
        /// estado de produção, e o que não pode custar nada nem quebrar.
        unsetenv("EXL3_TEL");
        assert(!exl3_tel::ativa());
        exl3_tel::passo_inicio();
        TEL_EVENTO("nao deve ser gravado");
        assert(exl3_tel::passo_fim() < 0.0);
        std::printf("ok: desligada por padrão\n");

        if (!exl3_tel::compilada_com_spdlog())
        {
            std::printf("sem spdlog nesta build; o resto do teste não se aplica\n");
            return 0;
        }
    }

    /*
    Ligada, com limiar fixo de 50 ms.

    O `setenv` funciona porque `exl3_tel::ativa()` lê o ambiente na PRIMEIRA
    chamada e guarda a decisão — e a primeira chamada deste processo ainda não
    aconteceu quando a telemetria está desligada por ausência da variável.
    Como o bloco acima já chamou `ativa()`, o teste do caso ligado roda num
    processo filho.
    */
    if (!filho)
    {
        setenv("EXL3_TESTE_FILHO", "1", 1);
        setenv("EXL3_TEL", "1", 1);
        setenv("EXL3_TEL_LIMIAR_MS", "50", 1);
        char caminho[4096];
        ssize_t n = readlink("/proc/self/exe", caminho, sizeof(caminho) - 1);
        if (n <= 0) { std::printf("nao deu para reexecutar; pulando\n"); return 0; }
        caminho[n] = 0;
        int r = system(caminho);
        return r == 0 ? 0 : 1;
    }

    /// No filho: a telemetria está ligada.
    assert(exl3_tel::ativa());
    std::printf("ok: EXL3_TEL=1 liga\n");

    /*
    Dez passos rápidos NÃO despejam. É o ponto todo do anel: o caso normal não
    deixa rastro, senão pegar os 5 passos lentos de 101 custaria o log dos 96
    normais — e escrever em disco dentro de um passo de 40 ms muda o que se
    está medindo.
    */
    for (int i = 0; i < 10; i++)
    {
        exl3_tel::passo_inicio();
        TEL_EVENTO("rapido: projecao");
        TEL_EVENTO("rapido: atencao");
        double ms = exl3_tel::passo_fim();
        assert(ms >= 0.0 && ms < 50.0);
    }
    std::printf("ok: 10 passos rápidos, nenhum despejo (nada acima)\n");

    /// O passo lento despeja: as linhas `[exl3_tel]` abaixo são a prova.
    exl3_tel::passo_inicio();
    TEL_EVENTO("lento: projecao");
    dorme_ms(30);
    TEL_EVENTO("lento: atencao");
    dorme_ms(40);
    TEL_EVENTO("lento: moe");
    double ms = exl3_tel::passo_fim();
    assert(ms > 50.0);
    std::printf("ok: passo de %.1f ms passou do limiar e despejou o anel acima\n", ms);

    /// Despejado, o anel zera: o próximo estouro não repete o que já se viu.
    exl3_tel::passo_inicio();
    TEL_EVENTO("depois do despejo");
    dorme_ms(60);
    exl3_tel::passo_fim();
    std::printf("ok: segundo despejo traz só o evento novo\n");
    return 0;
}

#endif
