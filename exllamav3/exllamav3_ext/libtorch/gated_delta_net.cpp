#include <Python.h>
#include "gated_delta_net.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../gdn.cuh"
#include "../add.cuh"

using namespace torch::indexing;

at::Tensor BC_GatedDeltaNet::run_bsz1_a
(
    const at::Tensor& x
)
{
    py::gil_scoped_release _;

    qkvz_proj->run(x, qkvz);
    ba_proj->run(x, ba);

    gated_delta_net_fused_op
    (
        qkvz, ba,
        dt_bias, a_log,
        mixed_qkv, z, beta, g,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        beta_scale
    );

    return mixed_qkv;
}

void BC_GatedDeltaNet::run_bsz1_b
(
    at::Tensor& mixed_qkv,
    at::Tensor& y,
    at::Tensor& recurrent_state
)
{
    cuda_recurrent_gated_delta_rule
    (
        mixed_qkv.transpose(1, 2),
        g,
        beta,
        recurrent_state,
        core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        c10::nullopt,
        false
    );

    norm->run(core_attn_out, core_attn_out_f, z);
    o_proj->run(core_attn_out_f, y);
}

bool BC_GatedDeltaNetSplit::needs_configure(int bsz, int seqlen, bool history)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= seqlen && seqlen <= MAX_QLEN,
                "BC_GatedDeltaNetSplit: shape out of range");
    return !slot(bsz, seqlen, history).configured;
}

void BC_GatedDeltaNetSplit::configure_slot
(
    int bsz,
    int seqlen,
    bool history,
    at::Tensor qkv,
    at::Tensor z,
    at::Tensor ba,
    at::Tensor beta,
    at::Tensor g,
    at::Tensor mixed_qkv,
    at::Tensor conv_out,
    at::Tensor core_attn_out,
    at::Tensor core_attn_out_f,
    at::Tensor qkv_xh,
    at::Tensor z_xh,
    at::Tensor o_xh
)
{
    Slot& s = slot(bsz, seqlen, history);

    s.qkv             = std::move(qkv);
    s.z               = std::move(z);
    s.ba              = std::move(ba);
    s.beta            = std::move(beta);
    s.g               = std::move(g);
    s.mixed_qkv       = std::move(mixed_qkv);
    s.conv_out        = std::move(conv_out);
    s.core_attn_out   = std::move(core_attn_out);
    s.core_attn_out_f = std::move(core_attn_out_f);
    s.qkv_xh          = std::move(qkv_xh);
    s.z_xh            = std::move(z_xh);
    s.o_xh            = std::move(o_xh);
    s.z_flat = s.z.view({bsz, seqlen, -1});

    TORCH_CHECK(s.qkv.is_contiguous() && s.z.is_contiguous() && s.ba.is_contiguous() &&
                s.beta.is_contiguous() && s.g.is_contiguous() && s.mixed_qkv.is_contiguous() &&
                s.conv_out.is_contiguous() && s.core_attn_out.is_contiguous() &&
                s.core_attn_out_f.is_contiguous(),
                "BC_GatedDeltaNetSplit: statics must be contiguous");

    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_GatedDeltaNetSplit::configure_slot_kda
(
    int bsz,
    int seqlen,
    bool history,
    at::Tensor qkv,
    at::Tensor z,
    at::Tensor b_out,
    at::Tensor fa_out,
    at::Tensor fb_out,
    at::Tensor ga_out,
    at::Tensor beta,
    at::Tensor g,
    at::Tensor mixed_qkv,
    at::Tensor conv_out,
    at::Tensor core_attn_out,
    at::Tensor core_attn_out_f,
    at::Tensor qkv_xh,
    at::Tensor o_xh,
    c10::optional<at::Tensor> xp,
    c10::optional<at::Tensor> yp
)
{
    Slot& s = slot(bsz, seqlen, history);
    int R = bsz * seqlen;

    // fp16 qkv_proj: qkv arrives as the padded 2D buffer (R_pad, F) together with xp; the
    // consumers keep reading the exact (bsz, seqlen, F) view of its first R rows
    if (qkv_proj_fp16)
    {
        int64_t hidden = qkv_proj_fp16->weight.size(0);
        int64_t f = qkv_proj_fp16->weight.size(1);
        TORCH_CHECK(xp.has_value(), "BC_GatedDeltaNetSplit (KDA): fp16 qkv_proj needs xp");
        TORCH_CHECK(xp->dim() == 2 && xp->dtype() == at::kHalf && xp->is_contiguous() &&
                    xp->size(0) >= R && xp->size(1) == hidden,
                    "BC_GatedDeltaNetSplit (KDA): xp must be (R_pad >= R, hidden) half");
        TORCH_CHECK(qkv.dim() == 2 && qkv.dtype() == at::kFloat && qkv.is_contiguous() &&
                    qkv.size(0) == xp->size(0) && qkv.size(1) == f,
                    "BC_GatedDeltaNetSplit (KDA): qkv must be (R_pad, F) float for fp16 qkv_proj");
        s.xp = std::move(xp.value());
        s.qkv_pad = std::move(qkv);
        s.qkv = s.qkv_pad.narrow(0, 0, R).view({bsz, seqlen, f});
    }
    else
    {
        TORCH_CHECK(!xp.has_value(), "BC_GatedDeltaNetSplit (KDA): xp only applies to fp16 qkv_proj");
        s.qkv = std::move(qkv);
    }

    // fp16 o_proj: core_attn_out_f arrives as the padded 2D buffer (R_pad, Nv*Hv) together with
    // yp; the norm writes the exact view, the GEMM reads/writes the padded buffers
    if (o_proj_fp16)
    {
        int64_t vdim = o_proj_fp16->weight.size(0);
        int64_t hidden = o_proj_fp16->weight.size(1);
        TORCH_CHECK(yp.has_value(), "BC_GatedDeltaNetSplit (KDA): fp16 o_proj needs yp");
        TORCH_CHECK(core_attn_out_f.dim() == 2 && core_attn_out_f.is_contiguous() &&
                    core_attn_out_f.size(0) >= R && core_attn_out_f.size(1) == vdim,
                    "BC_GatedDeltaNetSplit (KDA): core_attn_out_f must be (R_pad >= R, Nv*Hv) for fp16 o_proj");
        TORCH_CHECK(yp->dim() == 2 && yp->is_contiguous() &&
                    (yp->dtype() == at::kHalf || yp->dtype() == at::kFloat) &&
                    yp->size(0) == core_attn_out_f.size(0) && yp->size(1) == hidden,
                    "BC_GatedDeltaNetSplit (KDA): yp must be (R_pad, hidden) half/float");
        s.yp = std::move(yp.value());
        s.caof_pad = std::move(core_attn_out_f);
        s.core_attn_out_f = s.caof_pad.narrow(0, 0, R).view({bsz, seqlen, vdim});
    }
    else
    {
        TORCH_CHECK(!yp.has_value(), "BC_GatedDeltaNetSplit (KDA): yp only applies to fp16 o_proj");
        s.core_attn_out_f = std::move(core_attn_out_f);
    }

    s.z               = std::move(z);
    s.b_out           = std::move(b_out);
    s.fa_out          = std::move(fa_out);
    s.fb_out          = std::move(fb_out);
    s.ga_out          = std::move(ga_out);
    s.beta            = std::move(beta);
    s.g               = std::move(g);
    s.mixed_qkv       = std::move(mixed_qkv);
    s.conv_out        = std::move(conv_out);
    s.core_attn_out   = std::move(core_attn_out);
    s.qkv_xh          = std::move(qkv_xh);
    s.o_xh            = std::move(o_xh);
    s.z_flat = s.z.view({bsz, seqlen, -1});

    TORCH_CHECK(s.qkv.is_contiguous() && s.z.is_contiguous() && s.b_out.is_contiguous() &&
                s.fa_out.is_contiguous() && s.fb_out.is_contiguous() && s.ga_out.is_contiguous() &&
                s.beta.is_contiguous() && s.g.is_contiguous() && s.mixed_qkv.is_contiguous() &&
                s.conv_out.is_contiguous() && s.core_attn_out.is_contiguous() &&
                s.core_attn_out_f.is_contiguous(),
                "BC_GatedDeltaNetSplit (KDA): statics must be contiguous");

    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_GatedDeltaNetSplit::run_bszN_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history,
    Slot& s,
    Graph* graph
)
{
    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    int R = bsz * seqlen;

    // qkv/z projections: linear_gr bypasses BC_LinearEXL3::run_gr, which hard-refuses graph
    // capture above 1 row, and calls exl3_gemm_gr with this slot's own xh scratch instead, exactly
    // like BC_GatedMLP::run_bszN_gr does for shared-expert projections. An fp16 qkv_proj is a
    // cuBLAS node with no patchable sites: x is copied into the static xp at the graph head
    // (patched GP_copy2d_src) and the GEMM runs over all R_pad rows of the statics
    if (qkv_proj_fp16)
    {
        TORCH_CHECK(s.xp.defined() && x.size(2) == s.xp.size(1),
                    "BC_GatedDeltaNetSplit: slot not configured for fp16 qkv_proj");
        at::Tensor x2 = x.view({R, -1});
        at::Tensor xp2 = s.xp.narrow(0, 0, R);
        copy2d_gr(x2, xp2, graph);
        linear_gr(nullptr, qkv_proj_fp16, s.xp, s.qkv_pad, at::Tensor(), graph);
    }
    else
        linear_gr(qkv_proj, nullptr, x, s.qkv, s.qkv_xh, graph);

    if (kda)
    {
        // KDA: three fp16 GEMVs off x (patched inputs), low-rank second stages and the gate op
        // run entirely on graph statics. z (the sigmoid norm gate) is the g_b output
        gdn_ba_gemv_gr(x, b_weight_t, {}, s.b_out, graph);
        gdn_ba_gemv_gr(x, f_a_weight_t, {}, s.fa_out, graph);
        gdn_ba_gemv_gr(x, g_a_weight_t, {}, s.ga_out, graph);
        gdn_lowrank_gemv_f_gr(s.fa_out, f_b_weight_t, s.fb_out, graph);
        gdn_lowrank_gemv_f_gr(s.ga_out, g_b_weight_t, s.z_flat, graph);
        kda_gate_op_gr
        (
            s.qkv, s.b_out, s.fb_out,
            dt_bias, a_log,
            s.mixed_qkv, s.beta, s.g,
            lower_bound, beta_scale,
            graph
        );
    }
    else
    {
        exl3_gemm_gr(x, z_proj->trellis, s.z_flat, z_proj->suh, s.z_xh, z_proj->svh, -1, z_proj->mcg, z_proj->mul1, 0, graph);
        if (z_proj->bias)
            add_gr(s.z_flat, z_proj->bias.value(), s.z_flat, graph);

        gdn_ba_gemv_gr(x, ba_weight_t, ba_bias, s.ba, graph);

        gated_delta_net_fused_op_3_gr
        (
            s.qkv, s.ba,
            dt_bias, a_log,
            s.mixed_qkv, s.beta, s.g,
            beta_scale,
            graph
        );
    }

    cuda_causal_conv1d_update_gr
    (
        s.mixed_qkv,
        conv_state,
        slots,
        conv1d_weight,
        conv1d_bias,
        s.conv_out,
        true,
        history,
        graph
    );

    cuda_recurrent_gated_delta_rule_gr
    (
        s.conv_out,
        s.g,
        s.beta,
        recurrent_state,
        s.core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        history,
        graph
    );

    norm->run_gr(s.core_attn_out, s.core_attn_out_f, s.z, graph);

    // fp16 o_proj: same staging in reverse, the GEMM writes the static yp and the exact rows copy
    // out to y at the graph tail (patched GP_copy2d_dst)
    if (o_proj_fp16)
    {
        TORCH_CHECK(s.yp.defined() && y.size(2) == s.yp.size(1) && y.dtype() == s.yp.dtype(),
                    "BC_GatedDeltaNetSplit: slot not configured for fp16 o_proj / y dtype mismatch");
        linear_gr(nullptr, o_proj_fp16, s.caof_pad, s.yp, at::Tensor(), graph);
        at::Tensor yp2 = s.yp.narrow(0, 0, R);
        at::Tensor y2 = y.view({R, -1});
        copy2d_gr(yp2, y2, graph);
    }
    else
        linear_gr(o_proj, nullptr, s.core_attn_out_f, y, s.o_xh, graph);
}

void BC_GatedDeltaNetSplit::run_bszN
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    TORCH_CHECK(bsz >= 1 && bsz <= MAX_BSZ && seqlen >= 1 && seqlen <= MAX_QLEN,
                "BC_GatedDeltaNetSplit::run_bszN: shape out of range");
    Slot& s = slot(bsz, seqlen, history);
    TORCH_CHECK(s.configured, "BC_GatedDeltaNetSplit::run_bszN: slot not configured");

    // EXL3_BC_GDN_EAGER=1: run the fused path without ever capturing (A/B of the graph replay
    // against the same kernels launched eagerly)
    static const bool force_eager = [](){ const char* e = getenv("EXL3_BC_GDN_EAGER"); return e && *e == '1'; }();
    if (force_eager || s.graph->disabled || (!s.graph->ready && !s.graph->ready_to_record))
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        if (force_eager) return;
        s.graph->ready_to_record = true;
        s.graph_state_size = (int) conv_state.size(2);
        s.graph_hist_stride = (int) recurrent_state.size(1);
        return;
    }

    // The captured graph bakes in the state-buffer geometry (scalar kernel args can't be patched),
    // so a cache with different dimensions falls back to the eager path. The snapshot is per slot:
    // another slot's eager run against a second cache must not re-arm this slot's replay
    if ((int) conv_state.size(2) != s.graph_state_size ||
        (int) recurrent_state.size(1) != s.graph_hist_stride)
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, s.graph.get());
        s.graph->capture_end();
    }

    std::vector<PPTR> args;
    if (kda)
    {
        // Sites in emission order (see run_bszN_gr). An fp16 projection contributes no gemm
        // site; its staging copy is patched instead
        args.reserve(9);
        if (qkv_proj_fp16)
            args.emplace_back(GP_copy2d_src, (void*) x.data_ptr());     // x -> xp
        else
            args.emplace_back(GP_gemm_A,     (void*) x.data_ptr());     // qkv_proj input
        args.emplace_back(GP_gdn_ba_x,       (void*) x.data_ptr());     // b_proj input
        args.emplace_back(GP_gdn_ba_x,       (void*) x.data_ptr());     // f_a input
        args.emplace_back(GP_gdn_ba_x,       (void*) x.data_ptr());     // g_a input
        args.emplace_back(GP_conv1d_state,   (void*) conv_state.data_ptr());
        args.emplace_back(GP_conv1d_slots,   (void*) slots.data_ptr());
        args.emplace_back(GP_gdn_rule_state, (void*) recurrent_state.data_ptr());
        args.emplace_back(GP_gdn_rule_slots, (void*) slots.data_ptr());
        if (o_proj_fp16)
            args.emplace_back(GP_copy2d_dst, (void*) y.data_ptr());     // yp -> y
        else
            args.emplace_back(GP_gemm_C,     (void*) y.data_ptr());     // o_proj output
    }
    else
        args = std::vector<PPTR>
        {
            PPTR(GP_gemm_A,         (void*) x.data_ptr()),          // qkv_proj input
            PPTR(GP_gemm_A,         (void*) x.data_ptr()),          // z_proj input
            PPTR(GP_gdn_ba_x,       (void*) x.data_ptr()),
            PPTR(GP_conv1d_state,   (void*) conv_state.data_ptr()),
            PPTR(GP_conv1d_slots,   (void*) slots.data_ptr()),
            PPTR(GP_gdn_rule_state, (void*) recurrent_state.data_ptr()),
            PPTR(GP_gdn_rule_slots, (void*) slots.data_ptr()),
            PPTR(GP_gemm_C,         (void*) y.data_ptr())           // o_proj output
        };
    s.graph->launch(args, stream);
}

bool BC_Mamba2::needs_configure(int bsz, int seqlen, bool history)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= seqlen && seqlen <= MAX_QLEN,
                "BC_Mamba2: shape out of range");
    return !slot(bsz, seqlen, history).configured;
}

void BC_Mamba2::configure_slot
(
    int bsz,
    int seqlen,
    bool history,
    c10::optional<at::Tensor> xp,
    at::Tensor proj,
    at::Tensor mixed_xbc,
    at::Tensor dt,
    at::Tensor g,
    at::Tensor z_gate,
    at::Tensor conv_out,
    at::Tensor core_attn_out,
    at::Tensor core_attn_out_f,
    c10::optional<at::Tensor> yp,
    at::Tensor in_xh,
    at::Tensor o_xh
)
{
    Slot& s = slot(bsz, seqlen, history);

    s.xp              = std::move(xp);
    s.proj            = std::move(proj);
    s.mixed_xbc       = std::move(mixed_xbc);
    s.dt              = std::move(dt);
    s.g               = std::move(g);
    s.conv_out        = std::move(conv_out);
    s.core_attn_out   = std::move(core_attn_out);
    s.core_attn_out_f = std::move(core_attn_out_f);
    s.yp              = std::move(yp);
    s.in_xh           = std::move(in_xh);
    s.o_xh            = std::move(o_xh);

    TORCH_CHECK(padded_in == s.xp.has_value(), "BC_Mamba2: xp presence must match padded_in");
    TORCH_CHECK(padded_out == s.yp.has_value(), "BC_Mamba2: yp presence must match padded_out");

    TORCH_CHECK(s.proj.is_contiguous() && s.mixed_xbc.is_contiguous() && s.dt.is_contiguous() &&
                s.g.is_contiguous() && s.conv_out.is_contiguous() && s.core_attn_out.is_contiguous() &&
                s.core_attn_out_f.is_contiguous() && z_gate.is_contiguous(),
                "BC_Mamba2: statics must be contiguous");

    int gs = v_dim / num_k_heads;
    s.z_gate = z_gate.view({bsz, seqlen, num_k_heads, gs});
    s.core_g = s.core_attn_out.view({bsz, seqlen, num_k_heads, gs});
    s.core_f_g = s.core_attn_out_f.view({bsz, seqlen, num_k_heads, gs});

    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_Mamba2::run_bszN_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history,
    Slot& s,
    Graph* graph
)
{
    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    int R = bsz * seqlen;

    // Padded in_proj K: stage x through the zero-padded static (pad columns are zeroed at
    // configure time and only the exact width is ever copied in)
    at::Tensor x_in = x;
    if (s.xp)
    {
        at::Tensor x2 = x.reshape({R, hidden_size});
        at::Tensor xp2 = s.xp.value();
        copy2d_gr(x2, xp2, graph);
        x_in = xp2.view({bsz, seqlen, -1});
    }

    // in_proj: bypass BC_LinearEXL3::run_gr for the same reason as BC_GatedDeltaNetSplit above
    exl3_gemm_gr(x_in, in_proj->trellis, s.proj, in_proj->suh, s.in_xh, in_proj->svh, -1, in_proj->mcg, in_proj->mul1, 0, graph);
    if (in_proj->bias)
        add_gr(s.proj, in_proj->bias.value(), s.proj, graph);

    mamba2_fused_op_gr(s.proj, s.mixed_xbc, s.dt, s.g, s.z_gate, dt_bias, a_log, v_dim, dt_first, dt_min, dt_max, graph);

    cuda_causal_conv1d_update_gr
    (
        s.mixed_xbc,
        conv_state,
        slots,
        conv1d_weight,
        conv1d_bias,
        s.conv_out,
        true,
        history,
        graph
    );

    cuda_recurrent_mamba2_gr
    (
        s.conv_out,
        s.g,
        s.dt,
        d_skip,
        recurrent_state,
        s.core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        history,
        graph
    );

    norm->run_gr(s.core_g, s.core_f_g, s.z_gate, graph);

    // Padded o_proj N: the GEMM writes the padded static, then the exact width copies out to y
    if (s.yp)
    {
        at::Tensor yp3 = s.yp.value().view({bsz, seqlen, -1});
        exl3_gemm_gr(s.core_attn_out_f, o_proj->trellis, yp3, o_proj->suh, s.o_xh, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
        if (o_proj->bias)
            add_gr(yp3, o_proj->bias.value(), yp3, graph);
        at::Tensor y2 = y.reshape({R, hidden_size});
        at::Tensor yp2 = s.yp.value();
        copy2d_gr(yp2, y2, graph);
    }
    else
    {
        exl3_gemm_gr(s.core_attn_out_f, o_proj->trellis, y, o_proj->suh, s.o_xh, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
        if (o_proj->bias)
            add_gr(y, o_proj->bias.value(), y, graph);
    }
}

void BC_Mamba2::run_bszN
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    TORCH_CHECK(bsz >= 1 && bsz <= MAX_BSZ && seqlen >= 1 && seqlen <= MAX_QLEN,
                "BC_Mamba2::run_bszN: shape out of range");
    Slot& s = slot(bsz, seqlen, history);
    TORCH_CHECK(s.configured, "BC_Mamba2::run_bszN: slot not configured");

    if (s.graph->disabled || (!s.graph->ready && !s.graph->ready_to_record))
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        s.graph->ready_to_record = true;
        s.graph_state_size = (int) conv_state.size(2);
        s.graph_hist_stride = (int) recurrent_state.size(1);
        return;
    }

    // The captured graph bakes in the state-buffer geometry (scalar kernel args can't be patched),
    // so a cache with different dimensions falls back to the eager path. The snapshot is per slot:
    // another slot's eager run against a second cache must not re-arm this slot's replay
    if ((int) conv_state.size(2) != s.graph_state_size ||
        (int) recurrent_state.size(1) != s.graph_hist_stride)
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, s.graph.get());
        s.graph->capture_end();
    }

    std::vector<PPTR> args;
    args.reserve(8);
    if (s.xp)
        args.emplace_back(GP_copy2d_src, (void*) x.data_ptr());
    else
        args.emplace_back(GP_gemm_A, (void*) x.data_ptr());     // in_proj input
    args.emplace_back(GP_conv1d_state,   (void*) conv_state.data_ptr());
    args.emplace_back(GP_conv1d_slots,   (void*) slots.data_ptr());
    args.emplace_back(GP_gdn_rule_state, (void*) recurrent_state.data_ptr());
    args.emplace_back(GP_gdn_rule_slots, (void*) slots.data_ptr());
    if (s.yp)
        args.emplace_back(GP_copy2d_dst, (void*) y.data_ptr());
    else
        args.emplace_back(GP_gemm_C, (void*) y.data_ptr());     // o_proj output
    s.graph->launch(args, stream);
}
