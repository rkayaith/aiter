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
from aiter.jit.core import get_module
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="fused hipBLASLt kernels require gfx950",
)


@pytest.fixture(scope="module", autouse=True)
def _initialized_hipblaslt_extension() -> Iterator[None]:
    hipb_create_extension()
    try:
        if get_module("module_hipbsolgemm").hipblaslt_build_version < (1, 5, 0):
            pytest.skip("requires hipBLASLt >= 1.5.0")
        yield
    finally:
        torch.cuda.synchronize()
        hipb_destroy_extension()


def _as_column_major(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mT.contiguous().mT


def test_hipb_mm_epilogue_mm_uses_pytorch_matrix_layouts():
    eps = 1e-6
    tokens, input_size, hidden_size, output_size = 1024, 64, 1024, 64
    input = (
        torch.arange(tokens * input_size, dtype=torch.float32, device="cuda")
        .reshape(tokens, input_size)
        .remainder(13)
        .add(1)
        .to(torch.bfloat16)
    )
    weight1 = _as_column_major(
        torch.eye(input_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    )
    weight2 = _as_column_major(
        torch.eye(
            hidden_size,
            output_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * torch.linspace(
            0.5,
            1.5,
            output_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
    )
    residual = torch.arange(tokens * hidden_size, dtype=torch.float32, device="cuda")
    residual = residual.reshape(tokens, hidden_size).remainder(7).to(torch.bfloat16)
    residual_out = torch.empty_like(residual)
    gamma = torch.linspace(
        0.75,
        1.25,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output_buffer = torch.empty(
        tokens, output_size, dtype=torch.bfloat16, device="cuda"
    )

    output = hipb_mm_epilogue_mm(
        input,
        weight1,
        weight2,
        stages=(
            ResidualAdd(residual=residual, residual_out=residual_out),
            RMSNorm(gamma=gamma, eps=eps),
        ),
        intermediate_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        out=output_buffer,
    )
    torch.cuda.synchronize()

    expected_residual = input.float() @ weight1.float() + residual.float()
    expected = expected_residual * torch.rsqrt(
        expected_residual.square().mean(dim=-1, keepdim=True) + eps
    )
    expected = (expected * gamma.float()) @ weight2.float()

    assert input.shape == (tokens, input_size)
    assert input.stride() == (input_size, 1)
    assert weight1.shape == (input_size, hidden_size)
    assert weight1.stride() == (1, input_size)
    assert weight2.shape == (hidden_size, output_size)
    assert weight2.stride() == (1, hidden_size)
    assert output is output_buffer
    assert output.shape == (tokens, output_size)
    assert output.stride() == (output_size, 1)
    torch.testing.assert_close(
        residual_out.float(), expected_residual, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


def test_hipb_mm_epilogue_mm_rejects_row_major_weights():
    size = 256
    input = torch.zeros(1, size, dtype=torch.bfloat16, device="cuda")
    weight1 = torch.zeros(size, size, dtype=torch.bfloat16, device="cuda")
    weight2 = _as_column_major(
        torch.zeros(size, size, dtype=torch.bfloat16, device="cuda")
    )

    with pytest.raises(RuntimeError, match="mat2 must use column-major layout"):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )


def test_hipb_mm_epilogue_mm_bf16_model_sequence_cudagraph():
    eps = 1e-6
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = _as_column_major(torch.zeros(k, n, dtype=torch.bfloat16, device="cuda"))
    residual = torch.ones(m, n, dtype=torch.bfloat16, device="cuda")
    residual_out = torch.full_like(residual, 123.0)
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = _as_column_major(torch.eye(n, dtype=torch.bfloat16, device="cuda"))
    gemm_out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

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
    weight0 = _as_column_major(torch.eye(k, n, dtype=torch.bfloat16, device="cuda"))
    residual = torch.arange(m * n, dtype=torch.float32, device="cuda")
    residual = (residual.reshape(m, n) % 17 + 1).to(torch.bfloat16)
    residual_before = residual.clone()
    residual_out = torch.empty_like(residual)
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = _as_column_major(torch.eye(n, dtype=torch.bfloat16, device="cuda"))

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
    assert output.is_contiguous()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)
    assert residual_result is residual_out
    torch.testing.assert_close(residual, residual_before)
    torch.testing.assert_close(residual_out.float(), residual_before.float() + 1)


def test_hipb_mm_epilogue_mm_rejects_oversized_rmsnorm_handoff():
    """Reject row counts that exceed the fixed 64 KiB per-row stats buffer."""
    m, k, n = 16385, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight0 = _as_column_major(torch.zeros(k, n, dtype=torch.bfloat16, device="cuda"))
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = _as_column_major(torch.eye(n, dtype=torch.bfloat16, device="cuda"))

    with pytest.raises(RuntimeError, match="RMSNorm handoff buffer is too small"):
        hipb_mm_epilogue_mm(
            input,
            weight0,
            weight1,
            stages=(RMSNorm(gamma=gamma, eps=1e-6),),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("scale_dtype", (torch.uint8, torch.float8_e8m0fnu))
def test_hipb_mm_epilogue_mm_mxfp8_model_sequence(scale_dtype):
    eps = 1e-6
    m, k, n = 32, 256, 256
    input = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    input_scale = torch.full((64, 8), 127, dtype=torch.uint8, device="cuda").view(
        scale_dtype
    )
    weight0 = _as_column_major(
        torch.zeros(k, n, dtype=torch.float8_e4m3fn, device="cuda")
    )
    weight0_scale = torch.full((288, 8), 127, dtype=torch.uint8, device="cuda").view(
        scale_dtype
    )
    residual = torch.ones(m, n, dtype=torch.bfloat16, device="cuda")
    residual_out = torch.full_like(residual, 123.0)
    gamma = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    weight1 = _as_column_major(torch.eye(n, dtype=torch.float8_e4m3fn, device="cuda"))
    weight1_scale = torch.full((288, 8), 127, dtype=torch.uint8, device="cuda").view(
        scale_dtype
    )
    requant_scale_out = torch.empty(32, 8, dtype=scale_dtype, device="cuda")
    gemm_out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

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
        mat1_scale=input_scale,
        mat1_scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
        mat2_scale=weight0_scale,
        mat2_scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
        mat3_scale=weight1_scale,
        mat3_scale_mode=HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
        out=gemm_out,
    )
    torch.cuda.synchronize()

    expected_residual = torch.ones_like(residual)
    expected_output = torch.full_like(gemm_out, 1.0 / math.sqrt(1.0 + eps))
    # UE8M0 encodes ceil(log2(1 / 448)) as 119.

    assert output is gemm_out
    torch.testing.assert_close(residual_out, expected_residual)
    assert torch.all(requant_scale_out.view(torch.uint8) == 119)
    torch.testing.assert_close(gemm_out, expected_output, rtol=1e-2, atol=1e-2)


def test_hipb_mm_epilogue_mm_rejects_unsupported_requant_scale_mode():
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight1 = _as_column_major(torch.zeros(k, n, dtype=torch.bfloat16, device="cuda"))
    weight2 = _as_column_major(torch.eye(n, dtype=torch.bfloat16, device="cuda"))
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
    weight1 = _as_column_major(torch.zeros(k, n, dtype=torch.bfloat16, device="cuda"))
    weight2 = _as_column_major(torch.eye(n, dtype=torch.float8_e4m3fn, device="cuda"))
    requant_scale_out = torch.empty(1, dtype=torch.uint8, device="cuda")

    with pytest.raises(
        RuntimeError, match=r"Requant.scale_out must have shape \[at least 32, 8\]"
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
        ("mat1_scale", (32, 32), (64, 16)),
        ("mat2_scale", (32, 32), (64, 16)),
        ("mat3_scale", (16, 16), (32, 8)),
    ),
)
def test_hipb_mm_epilogue_mm_validates_mxfp8_scale_shape(
    scale_name: str,
    bad_shape: tuple[int, int],
    expected_shape: tuple[int, int],
):
    m, k, n_hidden, n_out = 33, 300, 50, 17
    input = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight1 = _as_column_major(
        torch.zeros(k, n_hidden, dtype=torch.float8_e4m3fn, device="cuda")
    )
    weight2 = _as_column_major(
        torch.zeros(n_hidden, n_out, dtype=torch.float8_e4m3fn, device="cuda")
    )
    scales: dict[str, torch.Tensor | None] = {
        "mat1_scale": None,
        "mat2_scale": None,
        "mat3_scale": None,
    }
    scales[scale_name] = torch.ones(bad_shape, dtype=torch.uint8, device="cuda")

    with pytest.raises(
        RuntimeError,
        match=rf"{scale_name} must have shape \[at least {expected_shape[0]}, {expected_shape[1]}\]",
    ):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(),
            intermediate_dtype=torch.float8_e4m3fn,
            dtype=torch.bfloat16,
            mat1_scale=scales["mat1_scale"],
            mat1_scale_mode=(
                HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
                if scale_name == "mat1_scale"
                else None
            ),
            mat2_scale=scales["mat2_scale"],
            mat2_scale_mode=(
                HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
                if scale_name == "mat2_scale"
                else None
            ),
            mat3_scale=scales["mat3_scale"],
            mat3_scale_mode=(
                HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT
                if scale_name == "mat3_scale"
                else None
            ),
        )


@pytest.mark.parametrize(
    "scale_mode",
    (None, HipblasLtMatmulScaleMode.SCALAR_32F),
)
@pytest.mark.parametrize("bad_shape", ((2,), (1, 2)))
def test_hipb_mm_epilogue_mm_validates_scalar_scale_shape(
    scale_mode: HipblasLtMatmulScaleMode | None,
    bad_shape: tuple[int, ...],
):
    m, k, n = 1, 256, 256
    input = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    weight1 = _as_column_major(
        torch.zeros(k, n, dtype=torch.float8_e4m3fn, device="cuda")
    )
    weight2 = _as_column_major(torch.zeros(n, n, dtype=torch.bfloat16, device="cuda"))
    input_scale = torch.ones(bad_shape, dtype=torch.float32, device="cuda")

    with pytest.raises(
        RuntimeError, match="mat1_scale must contain exactly one element"
    ):
        hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            mat1_scale=input_scale,
            mat1_scale_mode=scale_mode,
        )


@pytest.mark.parametrize(
    "scale_mode",
    (
        HipblasLtMatmulScaleMode.SCALAR_32F,
        HipblasLtMatmulScaleMode.OUTER_VEC_32F,
        HipblasLtMatmulScaleMode.VEC32_UE8M0,
        HipblasLtMatmulScaleMode.BLK32_UE8M0_32_8_EXT,
    ),
)
def test_hipb_mm_epilogue_mm_requires_scale_for_explicit_mode(scale_mode):
    input = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
    weight = _as_column_major(
        torch.zeros(256, 256, dtype=torch.bfloat16, device="cuda")
    )
    with pytest.raises(RuntimeError, match="mat1_scale mode requires a scale tensor"):
        hipb_mm_epilogue_mm(
            input,
            weight,
            weight,
            stages=(),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            mat1_scale_mode=scale_mode,
        )


def test_hipb_mm_epilogue_mm_survives_destroy_reinit():
    size = 256
    input = torch.zeros(1, size, dtype=torch.bfloat16, device="cuda")
    weight1 = _as_column_major(
        torch.zeros(size, size, dtype=torch.bfloat16, device="cuda")
    )
    weight2 = _as_column_major(torch.eye(size, dtype=torch.bfloat16, device="cuda"))
    residual = torch.ones_like(input)
    residual_out = torch.empty_like(input)
    gamma = torch.ones(size, dtype=torch.bfloat16, device="cuda")

    for _ in range(3):
        torch.cuda.synchronize()
        hipb_destroy_extension()
        hipb_destroy_extension()
        hipb_create_extension()
        hipb_create_extension()
        output = hipb_mm_epilogue_mm(
            input,
            weight1,
            weight2,
            stages=(
                ResidualAdd(residual=residual, residual_out=residual_out),
                RMSNorm(gamma=gamma, eps=1e-6),
            ),
            intermediate_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(output, residual, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(residual_out, residual)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
def test_hipb_mm_epilogue_mm_rejects_other_device():
    owner = torch.cuda.current_device()
    other = (owner + 1) % torch.cuda.device_count()
    owner_input = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
    owner_weight = _as_column_major(torch.eye(256, dtype=torch.bfloat16, device="cuda"))
    owner_residual = torch.ones_like(owner_input)
    owner_residual_out = torch.empty_like(owner_input)
    owner_gamma = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.device(other):
        # The caller's stream belongs to the other GPU; launch on the operand stream.
        with torch.cuda.stream(torch.cuda.Stream()):
            owner_output = hipb_mm_epilogue_mm(
                owner_input,
                owner_weight,
                owner_weight,
                stages=(
                    ResidualAdd(owner_residual, owner_residual_out),
                    RMSNorm(owner_gamma, 1e-6),
                ),
                intermediate_dtype=torch.bfloat16,
                dtype=torch.bfloat16,
            )
            assert torch.cuda.current_device() == other
        torch.cuda.synchronize(owner)
        torch.testing.assert_close(owner_output, owner_residual, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(owner_residual_out, owner_residual)
        input = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
        weight = _as_column_major(
            torch.zeros(256, 256, dtype=torch.bfloat16, device="cuda")
        )
        with pytest.raises(
            RuntimeError, match="destroy it before initializing on another device"
        ):
            hipb_mm_epilogue_mm(
                input,
                weight,
                weight,
                stages=(),
                intermediate_dtype=torch.bfloat16,
                dtype=torch.bfloat16,
            )

        # Destruction must free the old owner's buffer and preserve this device.
        hipb_destroy_extension()
        assert torch.cuda.current_device() == other
        hipb_create_extension()
        try:
            residual = torch.ones_like(input)
            residual_out = torch.empty_like(input)
            gamma = torch.ones(256, dtype=torch.bfloat16, device="cuda")
            output = hipb_mm_epilogue_mm(
                input,
                weight,
                _as_column_major(torch.eye(256, dtype=torch.bfloat16, device="cuda")),
                stages=(
                    ResidualAdd(residual=residual, residual_out=residual_out),
                    RMSNorm(gamma=gamma, eps=1e-6),
                ),
                intermediate_dtype=torch.bfloat16,
                dtype=torch.bfloat16,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(output, residual, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(residual_out, residual)
        finally:
            torch.cuda.synchronize()
            hipb_destroy_extension()
    hipb_create_extension()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
