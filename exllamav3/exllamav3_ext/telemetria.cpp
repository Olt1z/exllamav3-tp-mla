#include "telemetria.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

/// O `setup.py` varre todo `.cpp` da pasta, então este arquivo tem de compilar
/// onde os vizinhos compilam. Mesmo guarda de `ngram.cu`.
#ifdef _WIN32
    #include <process.h>
    #define exl3_getpid _getpid
#else
    #include <unistd.h>
    #define exl3_getpid getpid
#endif

#ifdef EXL3_TEM_SPDLOG
    #include <spdlog/spdlog.h>
    #include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace
{
    using Relogio = std::chrono::steady_clock;

    int inteiro_do_ambiente(const char* nome, int padrao)
    {
        const char* v = std::getenv(nome);
        if (!v || !*v) return padrao;
        char* fim = nullptr;
        long n = std::strtol(v, &fim, 10);
        return (fim && *fim == 0 && n > 0) ? static_cast<int>(n) : padrao;
    }

    /**
     * A hora de PAREDE, em UTC, só para carimbar o despejo.
     *
     * O `Relogio` do anel é `steady_clock` e continua sendo: ele mede duração e
     * não anda para trás quando o NTP ajusta. Mas `steady_clock` não tem época
     * — o valor dele não significa nada fora do processo, e o que se quer aqui
     * é casar o passo lento com a série do sensor e com a linha `Metrics` do
     * mesmo segundo. Para isso, e só para isso, vale o `system_clock`.
     */
    std::string agora_iso8601()
    {
        auto agora = std::chrono::system_clock::now();
        /// `%` trunca para zero em C++: com relógio anterior a 1970 os ms
        /// saem negativos e o `%03d` imprime o sinal, o que produz um instante
        /// que o `new Date()` do hub lê como `NaN` — e um despejo com data
        /// inválida é um despejo perdido em silêncio.
        auto ms = ((std::chrono::duration_cast<std::chrono::milliseconds>(agora.time_since_epoch()) % 1000) + std::chrono::milliseconds(1000)) % 1000;
        std::time_t t = std::chrono::system_clock::to_time_t(agora);
        std::tm tm{};
#ifdef _WIN32
        gmtime_s(&tm, &t);
#else
        gmtime_r(&t, &tm);
#endif
        char base[32];
        std::strftime(base, sizeof(base), "%Y-%m-%dT%H:%M:%S", &tm);
        char saida[64];
        std::snprintf(saida, sizeof(saida), "%s.%03dZ", base, static_cast<int>(ms.count()));
        return saida;
    }

    struct Evento
    {
        /**
         * Cópia, e não ponteiro.
         *
         * Guardar `const char*` funcionava para literal de C++ e explodia para
         * quem chama do Python: o pybind entrega o ponteiro de uma `str`
         * temporária, que morre quando a chamada volta, e o despejo leria
         * memória liberada. A cópia só acontece com a telemetria LIGADA, que é
         * modo de diagnóstico — no caminho normal a macro nem chega aqui.
         */
        std::string nome;
        /// Microssegundos desde o início do passo.
        long long em_us;
        /**
         * Qual passo gravou este evento.
         *
         * Sem ele o despejo mente. O anel NÃO é zerado a cada passo — é
         * circular de propósito, para mostrar o que veio antes do passo lento —
         * e `em_us` é relativo ao início do passo de CADA um. Na virada de um
         * passo para o outro a subtração dá delta negativo (visto no autoteste:
         * `+  -0.001 ms`), e a maior fatia, que é a agulha que se procura,
         * cairia num intervalo que nunca existiu. Com o número do passo o
         * despejo reinicia a conta na fronteira, e quem lê sabe quais eventos
         * são do passo que estourou: os do maior número.
         */
        int passo;
    };

    /**
     * O estado da telemetria.
     *
     * Uma instância só, sem trava. O passo de decode do ExLlamaV3 a lote 1 é
     * lançado por UMA thread — é justamente esse o gargalo que isto existe para
     * medir, e foi medido: CPU do motor cravada em 100,0% de um núcleo enquanto
     * a GPU ficava em 48,9% (v6rj9, 11/09/2026). Pôr um mutex aqui custaria
     * tempo no caminho quente para proteger contra uma concorrência que não
     * acontece.
     *
     * Sob tensor parallel, só o processo PRINCIPAL despeja. Os workers são
     * processos `spawn` de `mp_model_worker` e nunca abrem passo, então o anel
     * deles fica vazio e `despejar` sai cedo. A consequência para quem lê o
     * despejo: a parte distribuída do passo aparece como UM evento opaco
     * (`forward_tp`), e os marcos de dentro dos workers não existem. Ver
     * dentro do TP é trabalho do NVTX, não deste anel.
     */
    struct Estado
    {
        bool ligada = false;
        bool decidida = false;

        std::vector<Evento> anel;
        size_t proximo = 0;
        size_t gravados = 0;

        Relogio::time_point inicio_do_passo;
        bool passo_aberto = false;

        /// Quantos passos ANTERIORES ao que estourou entram no despejo.
        int contexto = 2;
        /// Limiar fixo em ms; 0 = automático (ver `limiar_atual`).
        double limiar_ms = 0.0;
        /// Média móvel das durações, para o limiar automático.
        double media_ms = 0.0;
        int passos = 0;

#ifdef EXL3_TEM_SPDLOG
        std::shared_ptr<spdlog::logger> log;
#endif
    };

    Estado& estado()
    {
        static Estado e;
        return e;
    }

    /**
     * A saída da telemetria, com ou sem spdlog.
     *
     * O anel, o limiar e o despejo são o valor real disto; o spdlog é só por
     * onde a linha sai. Amarrar um ao outro fez o build inteiro cair quando o
     * spdlog não compilou — ver a nota no `telemetria.h`. Aqui a linha sai de
     * qualquer jeito, e quem tiver spdlog ganha o sink dele de graça.
     */
    void escrever(const char* formato, ...)
    {
        char corpo[1024];
        va_list args;
        va_start(args, formato);
        std::vsnprintf(corpo, sizeof(corpo), formato, args);
        va_end(args);
#ifdef EXL3_TEM_SPDLOG
        Estado& e = estado();
        if (e.log) { e.log->warn("{}", corpo); return; }
#endif
        /**
         * O PID vai no prefixo de TODA linha, e a linha sai num `fwrite` só.
         *
         * O PID identifica o processo que despejou — é metade da chave de
         * deduplicação do hub, que relê o log inteiro a cada coleta, e separa
         * motores distintos escrevendo no mesmo log.
         *
         * O `fwrite` de um buffer pronto, e não `fprintf` com formato, porque
         * `stderr` ser "sem buffer" NÃO quer dizer uma chamada, um `write`: o
         * `vfprintf` do glibc >= 2.37 usa um estágio de 128 bytes para stream
         * sem buffer e, acima disso, emite vários `write()`. Medido, não
         * deduzido — dois processos no mesmo pipe, 40 mil linhas: 0 rasgadas
         * com 128 bytes, 29.318 com 133, 37.804 com 1027. Em glibc anterior a
         * 2.37 é pior, um `write()` por conversão.
         *
         * Importa porque o hub conta as linhas do despejo contra o número que o
         * cabeçalho promete: uma linha rasgada não casa com o padrão, a conta
         * nunca fecha, e o passo lento some inteiro sem aviso nenhum. E duas
         * metades podem colar formando uma linha que casa e mistura dois
         * processos. A linha fica abaixo de PIPE_BUF (4096), que é o tamanho
         * até o qual o núcleo garante a escrita indivisível.
         */
        char linha[1152];
        int n = std::snprintf(linha, sizeof(linha), "[exl3_tel %d] %s\n",
                              static_cast<int>(exl3_getpid()), corpo);
        if (n > 0)
        {
            size_t bytes = static_cast<size_t>(n) < sizeof(linha) ? static_cast<size_t>(n) : sizeof(linha) - 1;
            std::fwrite(linha, 1, bytes, stderr);
        }
    }


    void garantir_iniciada()
    {
        Estado& e = estado();
        if (e.decidida) return;
        e.decidida = true;

        const char* v = std::getenv("EXL3_TEL");
        e.ligada = v && *v && std::string(v) != "0";
        if (!e.ligada) return;

        e.anel.resize(static_cast<size_t>(inteiro_do_ambiente("EXL3_TEL_RING", 4096)));
        e.limiar_ms = static_cast<double>(inteiro_do_ambiente("EXL3_TEL_LIMIAR_MS", 0));
        e.contexto = inteiro_do_ambiente("EXL3_TEL_CONTEXTO", 2);

#ifdef EXL3_TEM_SPDLOG
        e.log = spdlog::get("exl3_tel");
        if (!e.log) e.log = spdlog::stderr_color_mt("exl3_tel");
        e.log->set_pattern("[exl3_tel %P] %v");
#endif
        escrever("anel de %zu eventos, limiar %.0f ms%s, %d passos de contexto",
                 e.anel.size(),
                 e.limiar_ms,
                 e.limiar_ms == 0.0 ? " (automático: 2x a média móvel)" : "",
                 e.contexto);
    }

    double limiar_atual()
    {
        const Estado& e = estado();
        if (e.limiar_ms > 0.0) return e.limiar_ms;
        /**
         * Automático: 2x a média móvel, e só depois de 20 passos.
         *
         * Antes disso não há do que a média ser média, e um limiar tirado de
         * dois passos despejaria o anel em todo passo do aquecimento — que é
         * quando TUDO é lento, e portanto quando o despejo não diz nada.
         */
        if (e.passos < 20) return 1e9;
        return e.media_ms * 2.0;
    }
}

namespace exl3_tel
{
    bool compilada_com_spdlog()
    {
#ifdef EXL3_TEM_SPDLOG
        return true;
#else
        return false;
#endif
    }

    bool ativa()
    {
        garantir_iniciada();
        return estado().ligada;
    }

    void evento(const char* nome)
    {
        Estado& e = estado();
        if (!e.ligada || e.anel.empty() || !e.passo_aberto) return;
        auto agora = Relogio::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(agora - e.inicio_do_passo).count();
        /// Nome vazio vira "?" porque o cabeçalho promete uma contagem e o
        /// hub confere: um evento que o despejo pula faz a conta nunca fechar,
        /// e o despejo inteiro é descartado do outro lado.
        e.anel[e.proximo] = Evento{std::string(nome && *nome ? nome : "?"), us, e.passos};
        e.proximo = (e.proximo + 1) % e.anel.size();
        if (e.gravados < e.anel.size()) e.gravados++;
    }

    void passo_inicio()
    {
        if (!ativa()) return;
        Estado& e = estado();
        /**
         * Abrir um passo já aberto é ignorado, e não rezera o relógio.
         *
         * Rezerar traria de volta, invisível, o delta negativo que o número do
         * passo conserta: `e.passos` só anda em `passo_fim`, então os dois
         * trechos levariam o MESMO rótulo, e a fronteira — que é detectada pela
         * troca de rótulo — não seria vista. O contextmanager do Python fecha
         * no `finally` e não é reentrante, mas `tel_passo_inicio` está exposto
         * cru no pybind e chamar duas vezes é um erro fácil de cometer.
         */
        if (e.passo_aberto) return;
        e.inicio_do_passo = Relogio::now();
        e.passo_aberto = true;
        /**
         * O anel NÃO é zerado aqui.
         *
         * É circular de propósito: quando um passo estoura o limiar, o que
         * interessa é também o que veio antes dele. Zerar a cada passo daria
         * só o passo lento, sem o contexto que explica por que ele foi lento.
         */
    }

    double passo_fim()
    {
        if (!ativa()) return -1.0;
        Estado& e = estado();
        if (!e.passo_aberto) return -1.0;
        e.passo_aberto = false;

        auto us = std::chrono::duration_cast<std::chrono::microseconds>(Relogio::now() - e.inicio_do_passo).count();
        double ms = static_cast<double>(us) / 1000.0;

        double limiar = limiar_atual();

        /// A média entra DEPOIS da comparação: senão o próprio passo lento
        /// levantaria o limiar que ele acabou de ter de passar.
        e.passos++;
        e.media_ms += (ms - e.media_ms) / static_cast<double>(e.passos < 100 ? e.passos : 100);

        if (ms > limiar) despejar(ms);
        return ms;
    }

    void despejar(double duracao_ms)
    {
        Estado& e = estado();
        if (!e.ligada || e.gravados == 0) return;

        /**
         * O cabeçalho carrega hora, PID e duração porque o despejo sai em
         * `stderr` cru, no meio do log do contêiner — sem carimbo ele não casa
         * com nada. A hora põe o passo na mesma régua da série do sensor e das
         * requisições, e é a chave de deduplicação do hub, que relê o log
         * inteiro a cada coleta; a duração é o que fez o despejo acontecer. O
         * PID não vem aqui: ele está no prefixo de toda linha, ver `escrever`.
         */
        /// Do mais velho para o mais novo, que é a ordem em que aconteceram.
        size_t inicio = (e.gravados == e.anel.size()) ? e.proximo : 0;

        /**
         * O despejo é LIMITADO aos últimos `contexto + 1` passos.
         *
         * Sem isto o despejo carrega tudo o que entrou no anel desde o despejo
         * anterior — o anel não é zerado a cada passo, e `gravados` só volta a
         * zero aqui. Com o limiar automático despejando uma fração dos passos,
         * isso são dezenas de passos e centenas de linhas por despejo, com teto
         * em `EXL3_TEL_RING` (4096).
         *
         * Dois estragos, ambos silenciosos. O hub lê o log numa JANELA
         * (`docker logs --tail N`): um despejo maior que a janela nunca chega
         * inteiro, a contagem do cabeçalho não fecha e o passo lento é
         * descartado — quanto mais contexto o anel juntou, mais garantido o
         * descarte. E a mesma janela é a que alimenta as medidas de velocidade:
         * inundada de despejo, ela deixa de conter as linhas `Metrics`, e ligar
         * o diagnóstico quebraria a medição que já funcionava.
         *
         * Contexto existe para explicar o passo lento, não para narrar o
         * minuto: dois passos antes bastam para ver se o anterior já vinha
         * ruim.
         */
        size_t primeiro = e.gravados;
        {
            int distintos = 0;
            int anterior = -1;
            for (size_t i = e.gravados; i > 0; i--)
            {
                const Evento& ev = e.anel[(inicio + i - 1) % e.anel.size()];
                if (ev.passo != anterior) { distintos++; anterior = ev.passo; }
                if (distintos > e.contexto + 1) break;
                primeiro = i - 1;
            }
        }

        escrever("--- %s: %.3f ms, %zu eventos, %s ---",
                 duracao_ms >= 0.0 ? "passo lento" : "sob demanda",
                 duracao_ms >= 0.0 ? duracao_ms : 0.0,
                 e.gravados - primeiro,
                 agora_iso8601().c_str());
        long long anterior_us = -1;
        int passo_anterior = -1;
        for (size_t i = primeiro; i < e.gravados; i++)
        {
            const Evento& ev = e.anel[(inicio + i) % e.anel.size()];
            /// Fronteira de passo: a conta recomeça, senão o delta atravessa
            /// dois relógios diferentes e sai negativo.
            if (ev.passo != passo_anterior) { anterior_us = -1; passo_anterior = ev.passo; }
            /// O delta é o que importa: onde o tempo FOI, não que horas eram.
            long long delta = anterior_us < 0 ? 0 : ev.em_us - anterior_us;
            escrever("  #%d  %10.3f ms  (+%8.3f ms)  %s",
                     ev.passo,
                     static_cast<double>(ev.em_us) / 1000.0,
                     static_cast<double>(delta) / 1000.0,
                     ev.nome.c_str());
            anterior_us = ev.em_us;
        }
        e.gravados = 0;
        e.proximo = 0;
    }
}
