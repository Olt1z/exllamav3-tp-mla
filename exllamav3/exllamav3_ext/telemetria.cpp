#include "telemetria.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

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
     * Sob tensor parallel cada rank é um PROCESSO, com seu próprio anel: os
     * despejos saem separados e identificados pelo PID do logger, que é o que
     * se quer para comparar ranks.
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
        char linha[1024];
        va_list args;
        va_start(args, formato);
        std::vsnprintf(linha, sizeof(linha), formato, args);
        va_end(args);
#ifdef EXL3_TEM_SPDLOG
        Estado& e = estado();
        if (e.log) { e.log->warn("{}", linha); return; }
#endif
        std::fprintf(stderr, "[exl3_tel] %s\n", linha);
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

#ifdef EXL3_TEM_SPDLOG
        e.log = spdlog::get("exl3_tel");
        if (!e.log) e.log = spdlog::stderr_color_mt("exl3_tel");
        e.log->set_pattern("[exl3_tel] %v");
#endif
        escrever("anel de %zu eventos, limiar %.0f ms%s",
                 e.anel.size(),
                 e.limiar_ms,
                 e.limiar_ms == 0.0 ? " (automático: 2x a média móvel)" : "");
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
        e.anel[e.proximo] = Evento{std::string(nome ? nome : "?"), us};
        e.proximo = (e.proximo + 1) % e.anel.size();
        if (e.gravados < e.anel.size()) e.gravados++;
    }

    void passo_inicio()
    {
        if (!ativa()) return;
        Estado& e = estado();
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

        if (ms > limiar) despejar();
        return ms;
    }

    void despejar()
    {
        Estado& e = estado();
        if (!e.ligada || e.gravados == 0) return;

        escrever("--- passo acima do limiar: %zu eventos no anel ---", e.gravados);
        /// Do mais velho para o mais novo, que é a ordem em que aconteceram.
        size_t inicio = (e.gravados == e.anel.size()) ? e.proximo : 0;
        long long anterior_us = -1;
        for (size_t i = 0; i < e.gravados; i++)
        {
            const Evento& ev = e.anel[(inicio + i) % e.anel.size()];
            if (ev.nome.empty()) continue;
            /// O delta é o que importa: onde o tempo FOI, não que horas eram.
            long long delta = anterior_us < 0 ? 0 : ev.em_us - anterior_us;
            escrever("  %10.3f ms  (+%8.3f ms)  %s",
                     static_cast<double>(ev.em_us) / 1000.0,
                     static_cast<double>(delta) / 1000.0,
                     ev.nome.c_str());
            anterior_us = ev.em_us;
        }
        e.gravados = 0;
        e.proximo = 0;
    }
}
