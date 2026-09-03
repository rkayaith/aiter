# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from collections.abc import Iterator

import pytest
import torch

from aiter import (
    HipblasLtMatmulScaleMode,
    Requant,
    ResidualAdd,
    RMSNorm,
    hipb_create_extension,
    hipb_destroy_extension,
    hipb_mm_epilogue_mm,
)


@pytest.fixture(scope="module", autouse=True)
def _initialized_hipblaslt_extension() -> Iterator[None]:
    hipb_create_extension()
    yield
    hipb_destroy_extension()


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
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            out=gemm_out,
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


def test_hipb_mm_epilogue_mm_bf16_multiple_rows():
    eps = 1e-6
    m, k, n = 2, 256, 256
    input = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.eye(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.arange(m * n, dtype=torch.float32, device="cuda")
    residual = (residual.reshape(m, n) % 17 + 1).to(torch.bfloat16)
    residual_before = residual.clone()
    residual_out = torch.empty_like(residual)
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.eye(n, dtype=torch.bfloat16, device="cuda")

    def run(input, weight0, weight1, residual, residual_out, gamma):
        output = hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(
                ResidualAdd(residual=residual, residual_out=residual_out),
                RMSNorm(gamma=gamma, eps=eps),
            ),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )
        return output, residual_out

    compiled_run = torch.compile(run, fullgraph=True)
    output, residual_result = compiled_run(
        input, weight0, weight1, residual, residual_out, gamma
    )
    torch.cuda.synchronize()

    expected = residual_before.float() + 1
    expected *= torch.rsqrt(expected.square().mean(dim=-1, keepdim=True) + eps)
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)
    assert residual_result is residual_out
    torch.testing.assert_close(residual, residual_before)
    torch.testing.assert_close(residual_out.float(), residual_before.float() + 1)


def test_hipb_mm_epilogue_mm_requires_residual_out():
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.ones(m, n, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.eye(n, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(
        RuntimeError, match="ResidualAdd requires a residual_out tensor"
    ):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(
                ResidualAdd(residual=residual, residual_out=None),  # type: ignore[arg-type]
            ),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
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
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )


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
                scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
            ),
        ),
        intermediate_dtype=torch.float8_e4m3fn,
        dtype=torch.bfloat16,
        input_scale=input_scale,
        input_scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
        weight1_scale=weight0_scale,
        weight1_scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
        weight2_scale=weight1_scale,
        weight2_scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
        out=gemm_out,
    )
    torch.cuda.synchronize()

    expected_residual = torch.ones_like(residual)
    expected_output = torch.full_like(gemm_out, 1.0 / math.sqrt(1.0 + eps))
    # UE8M0 encodes ceil(log2(1 / 448)) as 119. These are the gfx950-swizzled
    # offsets for the eight logical scale values in the first row.
    first_logical_row_offsets = [0, 64, 128, 192, 2, 66, 130, 194]

    assert output is gemm_out
    torch.testing.assert_close(residual_out, expected_residual)
    assert torch.all(requant_scale_out.flatten()[first_logical_row_offsets] == 119)
    torch.testing.assert_close(gemm_out, expected_output, rtol=1e-2, atol=1e-2)


def test_hipb_mm_epilogue_mm_rejects_unsupported_requant_scale_mode():
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    weight2 = torch.eye(n, dtype=torch.bfloat16, device="cuda")
    requant_scale_out = torch.empty(1, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="Requant does not support scale_mode 0"):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(
                Requant(
                    scale_out=requant_scale_out,
                    scale_mode=HipblasLtMatmulScaleMode.SCALAR_32F,
                ),
            ),
            intermediate_dtype=torch.float8_e4m3fn,
            dtype=torch.bfloat16,
        )


def test_hipb_mm_epilogue_mm_validates_requant_scale_shape():
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    weight2 = torch.eye(n, dtype=torch.float8_e4m3fn, device="cuda")
    requant_scale_out = torch.empty(1, dtype=torch.uint8, device="cuda")

    with pytest.raises(
        RuntimeError, match=r"Requant.scale_out must have shape .*\[32, 8\]"
    ):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(
                Requant(
                    scale_out=requant_scale_out,
                    scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
                ),
            ),
            intermediate_dtype=torch.float8_e4m3fn,
            dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    ("scale_name", "bad_shape", "expected_shape"),
    (
        ("input_scale", (32, 32), "[64, 16]"),
        ("weight1_scale", (32, 32), "[64, 16]"),
        ("weight2_scale", (16, 16), "[32, 8]"),
    ),
)
def test_hipb_mm_epilogue_mm_validates_mxfp8_scale_shape(
    scale_name: str,
    bad_shape: tuple[int, int],
    expected_shape: str,
):
    m, k, n_hidden, n_out = 33, 300, 50, 17
    input = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight1 = torch.zeros(n_hidden, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight2 = torch.zeros(n_out, n_hidden, dtype=torch.float8_e4m3fn, device="cuda")
    scales: dict[str, torch.Tensor | None] = {
        "input_scale": None,
        "weight1_scale": None,
        "weight2_scale": None,
    }
    scales[scale_name] = torch.ones(bad_shape, dtype=torch.uint8, device="cuda")

    with pytest.raises(
        RuntimeError,
        match=rf"{scale_name} must have shape .*{expected_shape}",
    ):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(),
            intermediate_dtype=torch.float8_e4m3fn,
            dtype=torch.bfloat16,
            input_scale=scales["input_scale"],
            input_scale_mode=(
                HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
                if scale_name == "input_scale"
                else None
            ),
            weight1_scale=scales["weight1_scale"],
            weight1_scale_mode=(
                HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
                if scale_name == "weight1_scale"
                else None
            ),
            weight2_scale=scales["weight2_scale"],
            weight2_scale_mode=(
                HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
                if scale_name == "weight2_scale"
                else None
            ),
        )


@pytest.mark.parametrize(
    "scale_mode",
    (None, HipblasLtMatmulScaleMode.SCALAR_32F),
)
@pytest.mark.parametrize("bad_shape", ((2,), (1, 1)))
def test_hipb_mm_epilogue_mm_validates_scalar_scale_shape(
    scale_mode: HipblasLtMatmulScaleMode | None,
    bad_shape: tuple[int, ...],
):
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight1 = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight2 = torch.zeros(n, n, dtype=torch.bfloat16, device="cuda")
    input_scale = torch.ones(bad_shape, dtype=torch.float32, device="cuda")

    with pytest.raises(
        RuntimeError, match="input_scale must contain exactly one element"
    ):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            input_scale=input_scale,
            input_scale_mode=scale_mode,
        )


def test_hipb_mm_epilogue_mm_launch_schema_marks_mutations():
    schema = str(torch.ops.aiter.hipb_mm_epilogue_mm_launch.default._schema)

    assert "Tensor input" in schema
    assert "Tensor weight1" in schema
    assert "Tensor weight2" in schema
    assert "Tensor? residual" in schema
    assert "Tensor(a5!)? residual_out" in schema
    assert "Tensor(a8!)? requant_scale_out" in schema
    assert "SymInt requant_scale_mode" in schema
    assert "Tensor(a10!) intermediate" in schema
    assert "Tensor(a11!) out" in schema
    assert "SymInt? input_scale_mode" in schema
    assert "SymInt? weight1_scale_mode" in schema
    assert "SymInt? weight2_scale_mode" in schema
    assert schema.count("!") == 4
    assert schema.endswith("-> ()")
