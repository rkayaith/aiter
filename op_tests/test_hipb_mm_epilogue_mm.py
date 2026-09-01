# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

from aiter.ops.gemm_rmsnorm_gemm import (
    Requant,
    RequantScaleGranularity,
    ResidualAdd,
    RMSNorm,
    hipb_mm_epilogue_mm,
)


def test_hipb_mm_epilogue_mm_bf16_model_sequence_cudagraph():
    eps = 1e-6
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.ones(m, n, dtype=torch.bfloat16, device="cuda")
    residual_out = torch.full_like(residual, 123.0)
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.eye(n, dtype=torch.bfloat16, device="cuda")
    gemm_out = torch.empty(n, m, dtype=torch.bfloat16, device="cuda").T

    def run():
        return hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(
                ResidualAdd(residual=residual, residual_out=residual_out),
                RMSNorm(gamma=gamma, eps=eps),
            ),
            gemm_out=gemm_out,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    torch.cuda.synchronize()

    expected_residual = torch.ones_like(residual)
    expected_output = torch.full_like(gemm_out, 1.0 / math.sqrt(1.0 + eps))
    assert output is gemm_out
    torch.testing.assert_close(gemm_out, expected_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual_out, expected_residual)


def test_hipb_mm_epilogue_mm_mxfp8_model_sequence():
    eps = 1e-6
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    input_scale = torch.full((32, 8), 127, dtype=torch.uint8, device="cuda")
    weight0 = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight0_scale = torch.full((256, 8), 127, dtype=torch.uint8, device="cuda")
    residual = torch.ones(m, n, dtype=torch.bfloat16, device="cuda")
    residual_out = torch.full_like(residual, 123.0)
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.eye(n, dtype=torch.float8_e4m3fn, device="cuda")
    weight1_scale = torch.full((256, 8), 127, dtype=torch.uint8, device="cuda")
    requant_scale_out = torch.empty(32, 8, dtype=torch.uint8, device="cuda")
    gemm_out = torch.empty(n, m, dtype=torch.bfloat16, device="cuda").T

    output = hipb_mm_epilogue_mm(
        input,
        weight0,
        weight1,
        stages=(
            ResidualAdd(residual=residual, residual_out=residual_out),
            RMSNorm(gamma=gamma, eps=eps),
            Requant(
                scale_out=requant_scale_out,
                granularity=RequantScaleGranularity.PER_BLOCK_MX,
            ),
        ),
        producer_out_dtype=torch.float8_e4m3fn,
        scaleA=input_scale,
        scaleB1=weight0_scale,
        scaleB2=weight1_scale,
        gemm_out=gemm_out,
    )
    torch.cuda.synchronize()

    expected_residual = torch.ones_like(residual)
    expected_output = torch.full_like(gemm_out, 1.0 / math.sqrt(1.0 + eps))
    # UE8M0 encodes ceil(log2(1 / 448)) as 119. These are the gfx950-swizzled
    # offsets for the eight logical scale values in the first row.
    first_logical_row_offsets = [0, 64, 128, 192, 2, 66, 130, 194]

    assert output is gemm_out
    torch.testing.assert_close(residual_out, expected_residual)
    assert torch.all(
        requant_scale_out.flatten()[first_logical_row_offsets] == 119
    )
    torch.testing.assert_close(gemm_out, expected_output, rtol=1e-2, atol=1e-2)


def test_hipb_mm_epilogue_mm_rejects_unsupported_producer_storage_dtype():
    input, weight0, gamma, weight1 = _bf16_operands()

    with pytest.raises(
        RuntimeError,
        match="producer_out_dtype must be bfloat16 or float8_e4m3fn",
    ):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(RMSNorm(gamma=gamma, eps=1e-6),),
            producer_out_dtype=torch.float32,
        )


def test_hipb_mm_epilogue_mm_rejects_oversized_rmsnorm_handoff():
    m, k, n = 16385, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.eye(n, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match="RMSNorm handoff buffer is too small"):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(RMSNorm(gamma=gamma, eps=1e-6),),
        )


def test_hipb_mm_epilogue_mm_rejects_undersized_mx_scale_buffer():
    input, weight0, gamma, weight1 = _bf16_operands()
    input = input.to(torch.float8_e4m3fn)
    weight0 = weight0.to(torch.float8_e4m3fn)
    weight1 = weight1.to(torch.float8_e4m3fn)
    requant_scale_out = torch.empty(32, 8, dtype=torch.uint8, device="cuda")
    undersized_input_scale = torch.empty(1, dtype=torch.uint8, device="cuda")

    with pytest.raises(RuntimeError, match="scaleA has the wrong size for its operand"):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(
                RMSNorm(gamma=gamma, eps=1e-6),
                Requant(
                    scale_out=requant_scale_out,
                    granularity=RequantScaleGranularity.PER_BLOCK_MX,
                ),
            ),
            producer_out_dtype=torch.float8_e4m3fn,
            scaleA=undersized_input_scale,
        )


def test_hipb_mm_epilogue_mm_defers_missing_residual_to_hipblaslt():
    input, weight0, gamma, weight1 = _bf16_operands()

    with pytest.raises(RuntimeError, match="hipblaslt error at"):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(
                ResidualAdd(residual=None),
                RMSNorm(gamma=gamma, eps=1e-6),
            ),
        )


def test_hipb_mm_epilogue_mm_defers_missing_requant_scale_to_hipblaslt():
    input, weight0, gamma, weight1 = _bf16_operands()

    with pytest.raises(RuntimeError, match="hipblaslt error at"):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(
                RMSNorm(gamma=gamma, eps=1e-6),
                Requant(
                    scale_out=None,
                    granularity=RequantScaleGranularity.PER_BLOCK_MX,
                ),
            ),
            producer_out_dtype=torch.float8_e4m3fn,
        )


def _bf16_operands():
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.eye(n, dtype=torch.bfloat16, device="cuda")
    return input, weight0, gamma, weight1
