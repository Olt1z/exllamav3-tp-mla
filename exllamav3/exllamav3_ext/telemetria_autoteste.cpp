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
/**
 * Confere o histograma da duração dos passos.
 *
 * O que ele existe para responder: "24 T/s" é 40 ms constante ou 30 ms com uma
 * cauda? A média não distingue os dois, e é a cauda que se está caçando.
 *
 * O teste faz 10 passos rápidos (microssegundos) e 3 lentos (dezenas de
 * milissegundos), então a linha tem de sair com o total certo e com baldes
 * DIFERENTES — um histograma que jogasse tudo num balde só passaria por
 * qualquer verificação de formato e não serviria para nada.
 */
/// Soma os pares `k:n` de [ini, fim) e guarda os baldes vistos.
static int somar_pares(const std::string& texto, size_t ini, size_t fim, std::set<int>& baldes)
{
    int total = 0;
    for (size_t j = ini; j < fim; )
    {
        size_t dp = texto.find(':', j);
        if (dp == std::string::npos || dp >= fim) break;
        baldes.insert(std::atoi(texto.c_str() + j));
        total += std::atoi(texto.c_str() + dp + 1);
        size_t esp = texto.find(' ', dp);
        if (esp == std::string::npos || esp >= fim) break;
        j = esp + 1;
    }
    return total;
}

static void conferir_histograma(const std::string& texto, int a_cada)
{
    std::set<int> baldes;
    std::set<int> baldes_intervalo;
    int linhas = 0;
    for (size_t i = texto.find("--- histograma: "); i != std::string::npos;
         i = texto.find("--- histograma: ", i + 1))
    {
        linhas++;
        /// Todo lote fecha com exatamente `a_cada` passos: o contador zera
        /// junto com os baldes, e uma linha com total diferente quer dizer que
        /// os dois deixaram de andar juntos.
        assert(std::atoi(texto.c_str() + i + 16) == a_cada);
        /// `..., duracao k:n k:n, intervalo(M) k:n k:n ---`
        size_t dur = texto.find(", duracao ", i);
        size_t itv = texto.find(", intervalo(", dur);
        size_t fim = texto.find(" ---", itv);
        assert(dur != std::string::npos && itv != std::string::npos && fim != std::string::npos);
        assert(somar_pares(texto, dur + 10, itv, baldes) == a_cada);
        int m = std::atoi(texto.c_str() + itv + 12);
        size_t pares = texto.find(") ", itv) + 2;
        /*
        A soma dos intervalos bate com o M anunciado, e M é `a_cada` ou um a
        menos: o primeiro passo do processo não tem "antes". Só o primeiro
        lote pode ficar um abaixo — a partir daí todo passo tem intervalo.
        */
        assert(somar_pares(texto, pares, fim, baldes_intervalo) == m);
        assert(m == a_cada || (linhas == 1 && m == a_cada - 1));
    }
    assert(linhas >= 1);
    /*
    Os intervalos são medidos de verdade, e não uma cópia da duração: o teste
    dorme 10 ms ENTRE os passos rápidos, e 10 ms = 10.000 µs cai no balde 13
    ([8,2 ms, 16,4 ms)), ou no 14 se o escalonador atrasar. Os passos rápidos em si duram microssegundos — balde
    de um dígito — então as duas séries têm de ser diferentes.
    */
    /// 13 ou 14: `sleep_for` só promete "pelo menos", e numa máquina carregada
    /// o escalonador devolve depois dos 16,4 ms que fecham o balde 13.
    assert(baldes_intervalo.count(13) == 1 || baldes_intervalo.count(14) == 1);
    assert(baldes.count(13) == 0 && baldes.count(14) == 0);
    /*
    Um balde CONHECIDO, e não só "dois baldes diferentes".

    O passo lento dorme 30 + 40 ms, então cai em [65,5 ms, 131,1 ms) = balde 16,
    e o balde é `floor(log2(microssegundos))`. Sem esta linha, trocar o
    deslocamento de 1 para 2 — log na base 4 — passava com a suíte inteira
    verde, e a tela rotularia um passo de 70 ms como "0,26 – 0,51 ms". É o
    "fator de dois de erro faria a tela mentir sobre a escala inteira" que o
    `histograma.ts` documenta, e este é o único lugar onde `balde()` existe.
    */
    assert(baldes.count(16) == 1);
    /*
    Baldes DIFERENTES entre os lotes.

    O teste faz passos de microssegundos e passos de dezenas de milissegundos.
    Um histograma que jogasse tudo num balde só passaria por qualquer conferência
    de formato e não distinguiria "40 ms constante" de "30 ms com cauda", que é a
    única coisa para a qual ele serve.
    */
    assert(baldes.size() >= 2);
    std::printf("ok: %d histogramas de %d passos, %zu baldes de duração (o lento no 16), intervalo de 10 ms no 13\n",
                linhas, a_cada, baldes.size());
}

static void conferir_despejo(const std::string& texto, int contexto)
{
    assert(texto.find("--- passo lento:") != std::string::npos);
    /// O cabeçalho do passo lento termina com o intervalo que o antecedeu: é
    /// o número que diz se o tempo daquele passo estava dentro ou fora dele.
    {
        size_t cab = texto.find("--- passo lento:");
        size_t fim = texto.find(" ---", cab);
        assert(texto.substr(cab, fim - cab).find(", intervalo ") != std::string::npos);
    }

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
        char caminho[4096];
        ssize_t n = readlink("/proc/self/exe", caminho, sizeof(caminho) - 1);
        if (n <= 0) { std::printf("nao deu para reexecutar; pulando\n"); return 0; }
        caminho[n] = 0;
        std::string comando = std::string(caminho) + " 2>" + SAIDA;

        setenv("EXL3_TESTE_FILHO", "1", 1);
        setenv("EXL3_TEL", "1", 1);
        setenv("EXL3_TEL_LIMIAR_MS", "50", 1);
        setenv("EXL3_TEL_CONTEXTO", "2", 1);
        /// 4 e não o padrão 1000: o teste faz 12 passos, e com o padrão o
        /// histograma nunca sairia — teste que não exercita não trava nada.
        setenv("EXL3_TEL_HISTOGRAMA", "4", 1);
        if (system(comando.c_str()) != 0) return 1;
        std::string despejo = ler(SAIDA);
        std::fputs(despejo.c_str(), stdout);
        conferir_despejo(despejo, 2);
        conferir_histograma(despejo, 4);
        std::remove(SAIDA);

        /*
        Segundo filho: `EXL3_TEL_MODULOS=1` SEM `EXL3_TEL_CONTEXTO`.

        O que se tranca: o contexto cai para 0 sozinho. São dezenas de marcos
        por passo, e o despejo com o padrão de 2 passos de contexto passaria de
        300 linhas — o tamanho que já expulsou as linhas `Metrics` da janela de
        log do hub (tarefa 4b). Um despejo com mais de um passo aqui é a
        regressão que faria o diagnóstico quebrar a medição.
        */
        setenv("EXL3_TESTE_FILHO", "2", 1);
        setenv("EXL3_TEL_MODULOS", "1", 1);
        unsetenv("EXL3_TEL_CONTEXTO");
        if (system(comando.c_str()) != 0) return 1;
        despejo = ler(SAIDA);
        assert(despejo.find("0 passos de contexto") != std::string::npos);
        assert(despejo.find("marcos por módulo") != std::string::npos);
        conferir_despejo(despejo, 0);
        std::remove(SAIDA);
        std::printf("ok: com EXL3_TEL_MODULOS o despejo traz só o passo que estourou\n");
        return 0;
    }

    /// No filho: a telemetria está ligada.
    assert(exl3_tel::ativa());
    const bool modulos = std::getenv("EXL3_TESTE_FILHO")[0] == '2';
    assert(exl3_tel::marca_modulos() == modulos);
    std::printf("ok: EXL3_TEL=1 liga%s\n", modulos ? ", com marcos por módulo" : "");

    /*
    Dez passos rápidos NÃO despejam. É o ponto todo do anel: o caso normal não
    deixa rastro, senão pegar os 5 passos lentos de 101 custaria o log dos 96
    normais — e escrever em disco dentro de um passo de 40 ms muda o que se
    está medindo.
    */
    for (int i = 0; i < 10; i++)
    {
        /// FORA do passo: é o intervalo que o histograma mede, e 10 ms é um
        /// balde conhecido (13). Dentro, mediria duração, que é outra série.
        dorme_ms(10);
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
