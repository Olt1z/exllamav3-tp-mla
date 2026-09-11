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
#include <set>
#include <string>
#include <unistd.h>

/// O despejo sai em `stderr`. Para AFIRMAR algo sobre ele — e não só imprimi-lo
/// e confiar no olho — o pai redireciona a saída do filho para um arquivo e lê.
static const char* SAIDA = "/tmp/exl3_tel_autoteste.txt";

static std::string ler(const char* caminho)
{
    std::string texto;
    FILE* f = std::fopen(caminho, "r");
    if (!f) return texto;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) texto.append(buf, n);
    std::fclose(f);
    return texto;
}

/**
 * Confere o despejo que o filho escreveu.
 *
 * Os dois erros que isto tranca já aconteceram de verdade neste arquivo:
 *
 *  - delta NEGATIVO, que aparecia na virada de passo porque o anel guarda os
 *    passos anteriores e `em_us` é relativo ao início de cada um. A maior
 *    fatia, que é a agulha que se procura, cairia num intervalo inexistente.
 *  - despejo SEM LIMITE, que carregava tudo desde o despejo anterior. O hub lê
 *    o log numa janela: um despejo maior que a janela nunca chega inteiro e o
 *    passo lento é descartado em silêncio do outro lado.
 */
static void conferir_despejo(const std::string& texto, int contexto)
{
    assert(texto.find("--- passo lento:") != std::string::npos);

    size_t negativos = 0;
    for (size_t i = texto.find("(+"); i != std::string::npos; i = texto.find("(+", i + 2))
    {
        size_t j = i + 2;
        while (j < texto.size() && texto[j] == ' ') j++;
        if (j < texto.size() && texto[j] == '-') negativos++;
    }
    assert(negativos == 0);

    /// Passos distintos do primeiro despejo: o que estourou mais o contexto.
    size_t inicio = texto.find("--- passo lento:");
    size_t fim = texto.find("--- passo lento:", inicio + 1);
    std::set<int> passos;
    for (size_t i = texto.find("  #", inicio); i != std::string::npos && i < fim; i = texto.find("  #", i + 3))
        passos.insert(std::atoi(texto.c_str() + i + 3));
    assert(!passos.empty());
    assert(static_cast<int>(passos.size()) <= contexto + 1);
    std::printf("ok: despejo com %zu passos (teto %d), nenhum delta negativo\n",
                passos.size(), contexto + 1);
}

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

        /// O anel funciona COM ou SEM spdlog — ele só muda por onde a linha
        /// sai. O teste roda igual nos dois casos, e é assim que se percebe se
        /// alguém amarrar de novo o mecanismo à biblioteca.
        std::printf("spdlog nesta build: %s\n",
                    exl3_tel::compilada_com_spdlog() ? "sim" : "não (saída por stderr)");
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
        setenv("EXL3_TEL_CONTEXTO", "2", 1);
        char caminho[4096];
        ssize_t n = readlink("/proc/self/exe", caminho, sizeof(caminho) - 1);
        if (n <= 0) { std::printf("nao deu para reexecutar; pulando\n"); return 0; }
        caminho[n] = 0;
        std::string comando = std::string(caminho) + " 2>" + SAIDA;
        int r = system(comando.c_str());
        if (r != 0) return 1;
        std::string despejo = ler(SAIDA);
        std::fputs(despejo.c_str(), stdout);
        conferir_despejo(despejo, 2);
        std::remove(SAIDA);
        return 0;
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
