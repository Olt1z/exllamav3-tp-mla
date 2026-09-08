#include <cuda_fp16.h>
#include "all_reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "ll.cuh"
#include "barrier_inner.cuh"

#define MAX_NUM_THREADS 1024
#define BATCH_STAGE 2

__global__ __launch_bounds__(MAX_NUM_THREADS)
void pg_all_reduce_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ shbuf_ptr,
    const size_t data_size,
    const size_t shbuf_size,
    uint32_t* abort_flag,
    const int phase
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();

    __shared__ bool r;
    int dir = blockIdx.x;

    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    uint8_t* data_end = data_ptr + data_size;
    const size_t reduce_stage_size = blockDim.x * sizeof(uint4);

    // Divide shared buffer among ranks
    size_t rank_shbuf_size = shbuf_size / num_ranks / reduce_stage_size * reduce_stage_size;

    // Divide each rank into segments divisible into stages, last segment may need padding
    size_t segment_size = CEIL_DIVIDE(data_size, num_ranks);
    segment_size = CEIL_DIVIDE(segment_size, reduce_stage_size) * reduce_stage_size;

    // Divide each workload and buffer into stages
    int num_stages = segment_size / reduce_stage_size;
    int num_buf_stages = rank_shbuf_size / reduce_stage_size;

    // O anel tem duas metades e cada uma é uma primitiva completa: as primeiras (num_ranks - 1)
    // iterações acumulam, o que É reduce-scatter, e as últimas (num_ranks - 1) copiam, o que É
    // all-gather. phase seleciona qual rodar. O context parallel precisa das duas separadas
    // (prova 19: só AG(lse) + reduce-scatter cabe no passo de decode), e elas já estavam aqui.
    //   PG_RING_ALL_REDUCE = 0  as duas metades, o comportamento de sempre
    //   PG_RING_REDUCE_SCATTER = 1  só a primeira: o rank r sai com o segmento r somado
    //   PG_RING_ALL_GATHER = 2  só a segunda: exige o segmento r já preenchido pelo rank r
    int iter_beg = (phase == 2) ? (num_ranks - 1) : 0;
    int iter_end = (phase == 1) ? (num_ranks - 1) : (num_ranks - 1) * 2;
    int num_iters = iter_end - iter_beg;
    bool no_overflow = num_stages * num_iters < num_buf_stages - 2;

    // Indexing
    auto shbuf_stage_ptr = [&] (int rank, int stage_idx)
    {
        return shbuf_ptr +
               rank * rank_shbuf_size +
               (stage_idx % num_buf_stages) * reduce_stage_size;
    };

    auto data_stage_ptr = [&] (int segment_idx, int stage_idx)
    {
        return data_ptr +
               segment_idx * segment_size +
               (stage_idx % num_stages) * reduce_stage_size;
    };

    // Sync
    auto wait_min_stage = [&] (uint32_t* stage_ptr, int min_stage, uint64_t deadline)
    {
        if (t == 0)
        {
            uint32_t sleep = SYNC_MIN_SLEEP;
            while ((int) ldg_acquire_sys_u32(stage_ptr) < min_stage)
            {
                __nanosleep(sleep);
                if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                else *abort_flag = check_timeout(ctx, deadline, "all_reduce");
                if (*abort_flag) break;
            }
        }
        __syncthreads();
    };

    auto stage_ready = [&] (uint32_t* stage_ptr, int min_stage)
    {
        if (t == 0)
            r = ((int) ldg_acquire_sys_u32(stage_ptr) >= min_stage);
        __syncthreads();
        return r;
    };

    // Send to next rank, receive from previous rank
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));
    int dst_rank = (this_rank + 1) % num_ranks;
    int src_rank = (this_rank + num_ranks - 1) % num_ranks;

    // Loop around ring
    for (int iter = iter_beg; iter < iter_end; ++iter)
    {
        uint64_t deadline = sync_deadline(ctx);

        // Outgoing segment to (rank+1)%num_ranks is (rank+iter)%num_iters
        // Incoming segment from (rank-1)%num_ranks is (rank+iter-1)%num_iters
        //
        // Deslocado de -1 em relação ao upstream para que o rank r termine a fase de acumulação
        // dono do segmento r, e não do (r+1)%num_ranks. Para o all-reduce inteiro a permutação
        // dos segmentos é indiferente (todo rank acaba com tudo), mas para as metades soltas é o
        // que faz a semântica bater com a do NCCL sem uma rotação do buffer por camada. O anel
        // continua consistente: o rank r manda send_seg(r, iter) e o r+1 recebe
        // recv_seg(r+1, iter), e as duas expressões são a mesma.
        int send_seg = (this_rank + num_ranks * 2 - 1 - iter) % num_ranks;
        int recv_seg = (this_rank + num_ranks * 2 - 2 - iter) % num_ranks;

        // Numeração de estágio relativa ao início da FASE, não do anel inteiro. Os contadores
        // nascem em zero e o teste de espaço do produtor é
        // `consumed >= stage_send - num_buf_stages + 1 + BATCH_STAGE`: começar a all-gather em
        // (num_ranks-1)*num_stages deixaria esse limiar positivo com os contadores ainda em zero,
        // e o produtor esperaria um consumo que só o próprio produtor pode destravar.
        int stage_beg = (iter - iter_beg) * num_stages;
        int stage_end = stage_beg + num_stages;
        int stage_send = stage_beg;
        int stage_recv = stage_beg;

        uint32_t sleep = SYNC_MIN_SLEEP;

        if (dir == 0)
        {
            while (stage_recv < stage_end)
            {
                __shared__ uint32_t sr;
                if (t == 0)
                    sr = (int) ldg_acquire_sys_u32(ctx->reduce_stage_produced + src_rank);
                __syncthreads();
                uint32_t stage_ready = sr;

                if (stage_recv < stage_ready)
                {
                    while (stage_recv < stage_ready)
                    {
                        sleep = SYNC_MIN_SLEEP;

                        // First num_ranks - 1 iterations: accumulate
                        if (iter < num_ranks - 1)
                        {
                            float4* src = (float4*) shbuf_stage_ptr(this_rank, stage_recv);
                            float4* dst = (float4*) data_stage_ptr(recv_seg, stage_recv);
                            if (dst + t < (float4*) data_end)
                            {
                                float4 a = dst[t];
                                float4 b = src[t];
                                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                                dst[t] = a;
                            }
                        }

                        // Last num_ranks - 1 iterations: copy
                        else
                        {
                            uint4* src = (uint4*) shbuf_stage_ptr(this_rank, stage_recv);
                            uint4* dst = (uint4*) data_stage_ptr(recv_seg, stage_recv);
                            if (dst + t < (uint4*) data_end) dst[t] = src[t];
                        }

                        // Advance
                        stage_recv++;
                    }

                    // All warps must finish reading their slots before thread 0 frees them for the producer
                    __syncthreads();

                    if (t == 0)
                    {
                        stg_release_sys_u32(ctx->reduce_stage_consumed + this_rank, stage_recv);
                    }
                }
                else
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "all_reduce (1)");
                    if (*abort_flag) break;
                }
            }
        }

        // Send
        if (dir == 1)
        {
            while (stage_send < stage_end)
            {
                bool ready_send = stage_send < stage_end &&
                                  (no_overflow || stage_ready(ctx->reduce_stage_consumed + dst_rank, stage_send - num_buf_stages + 1 + BATCH_STAGE));
                if (ready_send)
                {
                    for (int i = 0; i < BATCH_STAGE && stage_send < stage_end; ++i)
                    {
                        sleep = SYNC_MIN_SLEEP;
                        uint4* src = (uint4*) data_stage_ptr(send_seg, stage_send);
                        uint4* dst = (uint4*) shbuf_stage_ptr(dst_rank, stage_send);
                        if (src + t < (uint4*) data_end) dst[t] = src[t];

                        // Advance
                        stage_send++;
                    }

                    // All warps must finish staging before thread 0 publishes the counter; the release store only
                    // orders thread 0's own prior writes
                    __syncthreads();

                    if (t == 0)
                    {
                        stg_release_sys_u32(ctx->reduce_stage_produced + this_rank, stage_send);
                    }
                }
                else
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "all_reduce (2)");
                    if (*abort_flag) break;
                }
            }

            // Wait for destination to finish receiving
            wait_min_stage(ctx->reduce_stage_consumed + dst_rank, stage_end, deadline);
        }

        // Every block must arrive at the barrier in every round: a block that breaks out on its own
        // timeout while the other block is already parked in grid.sync() would leave that block
        // waiting forever. Sync first, then read the (possibly other block's) abort flag through a
        // volatile load so a cached copy can't hide it
        grid.sync();
        if (*((volatile uint32_t*) abort_flag)) break;
    }

    // Finished. Reset counters for next kernel
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);

    if (t == 0)
    {
        ctx->reduce_stage_consumed[dst_rank] = 0;
        ctx->reduce_stage_produced[this_rank] = 0;
        __threadfence_system();
    }
}


static void pg_ring
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag,
    int phase
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* shbuf_ptr = (uint8_t*) shbuf_dev;
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 16 == 0, "data_size must be multiple of 16");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    long num_ranks = devices.size();

    int threads = (int) CEIL_DIVIDE(CEIL_DIVIDE(data_size / 16ll, num_ranks), 32ll) * 32ll;
    threads = MIN(threads, MAX_NUM_THREADS);

    // As metades soltas exigem que o segmento do anel COINCIDA com a fatia lógica do chamador:
    // reduce-scatter entrega ao rank r o segmento r, e all-gather espera encontrar ali a
    // contribuição dele. No all-reduce inteiro a última fatia pode sobrar do fim do tensor sem
    // problema (o kernel guarda por data_end e todo rank termina com tudo), mas aqui o
    // arredondamento moveria a fronteira e cortaria dados em silêncio.
    //
    // O segmento é `threads * 16` B, então threads tem de DIVIDIR a fatia em unidades de 16 B, e
    // não é o que a conta do all-reduce dá: o lse do context parallel são 256 B por rank, que a
    // conta original arredondaria para um estágio de 512 B. Pega-se o maior divisor até o teto de
    // bloco -- para fatia potência de dois isso é só o teto, e o laço nunca passa de 1024 voltas.
    if (phase != 0)
    {
        TORCH_CHECK(data_size % (size_t) num_ranks == 0,
                    "pg_ring: reduce_scatter/all_gather exigem tensor divisivel por num_ranks");
        size_t chunk = data_size / (size_t) num_ranks;
        TORCH_CHECK(chunk % 16 == 0,
                    "pg_ring: a fatia por rank (", chunk, " B) tem de ser multipla de 16");
        long chunk16 = (long) (chunk / 16);
        long t = MIN((long) MAX_NUM_THREADS, chunk16);
        while (t > 1 && chunk16 % t) t--;
        threads = (int) t;
    }

    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    void* kernelArgs[] =
    {
        (void*)& ctx_dev,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& master_device,
        (void*)& data_ptr,
        (void*)& shbuf_ptr,
        (void*)& data_size,
        (void*)& shbuf_size,
        (void*)& abort_flag_ptr,
        (void*)& phase
    };

    dim3 block_grid(2);
    dim3 block_dim(threads);

    cudaLaunchCooperativeKernel
    (
        (void*)pg_all_reduce_kernel,
        block_grid,
        block_dim,
        kernelArgs,
        0,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}


void pg_all_reduce
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    pg_ring(ctx, ctx_dev, devices, this_device, master_device, tensor,
            shbuf_dev, shbuf_size, abort_flag, 0);
}


void pg_reduce_scatter
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    // No lugar: entra o buffer inteiro (num_ranks fatias), sai com a fatia do proprio rank somada
    // sobre todos. As outras fatias ficam com lixo parcial, e o chamador nao deve le-las.
    pg_ring(ctx, ctx_dev, devices, this_device, master_device, tensor,
            shbuf_dev, shbuf_size, abort_flag, 1);
}


void pg_all_gather
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    // No lugar: o chamador ja escreveu a contribuicao dele na fatia do proprio rank; sai com as
    // num_ranks fatias preenchidas em todos.
    pg_ring(ctx, ctx_dev, devices, this_device, master_device, tensor,
            shbuf_dev, shbuf_size, abort_flag, 2);
}