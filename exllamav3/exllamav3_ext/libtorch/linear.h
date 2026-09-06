#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "../graph.cuh"

struct BC_LinearFP16
{
    at::Tensor weight;
    c10::optional<at::Tensor> bias;

    BC_LinearFP16
    (
        at::Tensor _weight,
        c10::optional<at::Tensor> _bias
    ) :
        weight(std::move(_weight)),
        bias(std::move(_bias))
    {}

    void run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph);
    void run(const at::Tensor& x, at::Tensor& y);
    // void run_cublas(const at::Tensor& x, at::Tensor& y);
};

struct BC_LinearEXL3
{
    at::Tensor trellis;
    at::Tensor suh;
    at::Tensor svh;
    int K;
    c10::optional<at::Tensor> bias;
    bool mcg;
    bool mul1;
    at::Tensor xh;

    BC_LinearEXL3
    (
        at::Tensor _trellis,
        at::Tensor _suh,
        at::Tensor _svh,
        int _K,
        c10::optional<at::Tensor> _bias,
        bool _mcg,
        bool _mul1,
        at::Tensor _xh
    ) :
        trellis(std::move(_trellis)),
        suh(std::move(_suh)),
        svh(std::move(_svh)),
        K(_K),
        bias(std::move(_bias)),
        mcg(_mcg),
        mul1(_mul1),
        xh(std::move(_xh))
    {}

    void run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph);
    void run(const at::Tensor& x, at::Tensor& y);
    at::Tensor run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32);
};

// One projection inside a captured graph, EXL3 or fp16 (exactly one of the two is set).
// EXL3 is a kernel node with patchable A/C sites (the caller may pass a per-step x or y and patch
// GP_gemm_A / GP_gemm_C at replay); fp16 is a cuBLAS node with NO patchable sites, so x and y must
// both be static buffers that the caller stages with copy2d_gr (bc_attn / bc_dsa pattern). Bias is
// applied in both cases. xh is the Hadamard scratch for EXL3 (ignored for fp16, may be empty).
void linear_gr
(
    const std::shared_ptr<BC_LinearEXL3>& exl3,
    const std::shared_ptr<BC_LinearFP16>& fp16,
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& xh,
    Graph* graph
);
