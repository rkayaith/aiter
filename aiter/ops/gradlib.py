# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops


@compile_ops("module_hipbsolgemm")
def hipb_create_extension() -> None: ...


@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension() -> None: ...


class HipblasLtMatmulScaleMode(IntEnum):
    SCALAR_32F = 0
    BLK32_UE8M0_32_8_EXT = 1001

    # TODO: Expose additional hipblasLtMatmulMatrixScale_t modes when the
    # corresponding operand-scale or requant producer formats are supported.


@dataclass(frozen=True)
class ResidualAdd:
    """Add ``residual`` to the first GEMM result and store the sum in ``residual_out``."""

    residual: torch.Tensor
    residual_out: torch.Tensor


@dataclass(frozen=True)
class RMSNorm:
    gamma: torch.Tensor
    eps: float


@dataclass(frozen=True)
class Requant:
    scale_out: torch.Tensor
    scale_mode: HipblasLtMatmulScaleMode


FusedEpilogueStage: TypeAlias = ResidualAdd | RMSNorm | Requant

# Values from hipblasLtFuseableEpilogue_t.
_HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD = 0
_HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS = 2
_HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT = 5


def gen_hipb_mm_fake_tensor(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
):
    mat1_sizes = mat1.size()
    mat2_sizes = mat2.size()
    in_dtype = mat1.dtype
    out_dtype = out_dtype if out_dtype is not None else in_dtype
    result = torch.empty(
        (mat1_sizes[0], mat2_sizes[1]), dtype=out_dtype, device=mat1.device
    )

    return result


# torch-free kernel entry: writes into caller-allocated `result` (the de-torched C++
# TU can no longer torch::empty). outDtype is read from result.dtype() on the C++ side.
@compile_ops("module_hipbsolgemm", fc_name="hipb_mm", develop=True)
def _hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
) -> None: ...


def hipb_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    solution_index: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleOut: torch.Tensor | None = None,
    bpreshuffle: bool | None = None,
    use_gelu: bool | None = None,
) -> torch.Tensor:
    out_dtype = out_dtype if out_dtype is not None else mat1.dtype
    result = torch.empty(
        (mat1.size(0), mat2.size(1)), dtype=out_dtype, device=mat1.device
    )
    _hipb_mm(
        mat1,
        mat2,
        solution_index,
        result,
        bias,
        scaleA,
        scaleB,
        scaleOut,
        bpreshuffle,
        use_gelu,
    )
    return result


direct_register_custom_op(
    "hipb_mm",
    hipb_mm,
    [],
    fake_impl=gen_hipb_mm_fake_tensor,
)


@compile_ops("module_hipbsolgemm", fc_name="hipb_findallsols", develop=True)
def _hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    result: torch.Tensor,
    bias: torch.Tensor | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleC: torch.Tensor | None = None,
    bpreshuffle: bool = False,
    use_gelu: bool = False,
) -> list[int]: ...


def hipb_findallsols(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scaleA: torch.Tensor | None = None,
    scaleB: torch.Tensor | None = None,
    scaleC: torch.Tensor | None = None,
    bpreshuffle: bool = False,
    use_gelu: bool = False,
) -> list[int]:
    out_dtype = out_dtype if out_dtype is not None else mat1.dtype
    result = torch.empty(
        (mat1.size(0), mat2.size(1)), dtype=out_dtype, device=mat1.device
    )
    return _hipb_findallsols(
        mat1, mat2, result, bias, scaleA, scaleB, scaleC, bpreshuffle, use_gelu
    )


@compile_ops(
    "module_hipbsolgemm",
    fc_name="hipb_mm_epilogue_mm",
    ffi_type="pybind",
    develop=True,
)
def _hipb_mm_epilogue_mm(
    A: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    stage_kinds: list[int],
    residual: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    gamma: torch.Tensor | None,
    eps: float,
    requant_scale_out: torch.Tensor | None,
    requant_scale_mode: int,
    intermediate: torch.Tensor,
    gemm_out: torch.Tensor,
    scaleA: torch.Tensor | None,
    scaleB1: torch.Tensor | None,
    scaleB2: torch.Tensor | None,
    scaleA_mode: int | None,
    scaleB1_mode: int | None,
    scaleB2_mode: int | None,
    solution_index1: int | None,
    solution_index2: int | None,
) -> None: ...


def _flatten_epilogue_stages(
    stages: tuple[FusedEpilogueStage, ...],
) -> tuple[
    list[int],
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    float,
    torch.Tensor | None,
    int,
]:
    stage_kinds: list[int] = []
    residual = None
    residual_out = None
    gamma: torch.Tensor | None = None
    eps = 0.0
    requant_scale_out = None
    requant_scale_mode = -1

    for stage in stages:
        if isinstance(stage, ResidualAdd):
            stage_kinds.append(_HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD)
            residual = stage.residual
            residual_out = stage.residual_out
        elif isinstance(stage, RMSNorm):
            stage_kinds.append(_HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS)
            gamma = stage.gamma
            eps = stage.eps
        elif isinstance(stage, Requant):
            stage_kinds.append(_HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT)
            requant_scale_out = stage.scale_out
            requant_scale_mode = int(stage.scale_mode)
        else:
            raise TypeError(f"unsupported fused epilogue stage: {type(stage).__name__}")

    return (
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_scale_mode,
    )


@torch.library.custom_op(
    "aiter::hipb_mm_epilogue_mm_launch",
    mutates_args=(
        "residual_out",
        "requant_scale_out",
        "intermediate",
        "out",
    ),
)
def _hipb_mm_epilogue_mm_launch(
    input: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    stage_kinds: list[int],
    residual: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    gamma: torch.Tensor | None,
    eps: float,
    requant_scale_out: torch.Tensor | None,
    requant_scale_mode: int,
    intermediate: torch.Tensor,
    out: torch.Tensor,
    input_scale: torch.Tensor | None,
    weight1_scale: torch.Tensor | None,
    weight2_scale: torch.Tensor | None,
    input_scale_mode: int | None,
    weight1_scale_mode: int | None,
    weight2_scale_mode: int | None,
    solution_index1: int | None,
    solution_index2: int | None,
) -> None:
    _hipb_mm_epilogue_mm(
        input,
        weight1,
        weight2,
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_scale_mode,
        intermediate,
        out,
        input_scale,
        weight1_scale,
        weight2_scale,
        input_scale_mode,
        weight1_scale_mode,
        weight2_scale_mode,
        solution_index1,
        solution_index2,
    )


@_hipb_mm_epilogue_mm_launch.register_fake
def _hipb_mm_epilogue_mm_launch_fake(
    input: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    stage_kinds: list[int],
    residual: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    gamma: torch.Tensor | None,
    eps: float,
    requant_scale_out: torch.Tensor | None,
    requant_scale_mode: int,
    intermediate: torch.Tensor,
    out: torch.Tensor,
    input_scale: torch.Tensor | None,
    weight1_scale: torch.Tensor | None,
    weight2_scale: torch.Tensor | None,
    input_scale_mode: int | None,
    weight1_scale_mode: int | None,
    weight2_scale_mode: int | None,
    solution_index1: int | None,
    solution_index2: int | None,
) -> None:
    del input, weight1, weight2, stage_kinds
    del residual, residual_out, gamma, eps
    del requant_scale_out, requant_scale_mode
    del intermediate, out, input_scale, weight1_scale, weight2_scale
    del input_scale_mode, weight1_scale_mode, weight2_scale_mode
    del solution_index1, solution_index2


def hipb_mm_epilogue_mm(
    input: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    *,
    stages: tuple[FusedEpilogueStage, ...],
    intermediate_dtype: torch.dtype,
    dtype: torch.dtype,
    input_scale: torch.Tensor | None = None,
    weight1_scale: torch.Tensor | None = None,
    weight2_scale: torch.Tensor | None = None,
    input_scale_mode: HipblasLtMatmulScaleMode | None = None,
    weight1_scale_mode: HipblasLtMatmulScaleMode | None = None,
    weight2_scale_mode: HipblasLtMatmulScaleMode | None = None,
    out: torch.Tensor | None = None,
    solution_index1: int | None = None,
    solution_index2: int | None = None,
) -> torch.Tensor:
    """Run two GEMMs with the requested fused epilogue stages between them.

    Scale modes are optional. When a scale tensor is provided without a mode,
    hipBLASLt's default scalar FP32 mode is used.
    """

    (
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_scale_mode,
    ) = _flatten_epilogue_stages(stages)

    m = input.size(0)
    n_hidden = weight1.size(0)
    n_out = weight2.size(0)
    intermediate = torch.empty(
        (m, n_hidden),
        dtype=intermediate_dtype,
        device=input.device,
    )
    out = (
        out
        if out is not None
        else torch.empty((n_out, m), dtype=dtype, device=input.device).T
    )
    _hipb_mm_epilogue_mm_launch(
        input,
        weight1,
        weight2,
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_scale_mode,
        intermediate,
        out,
        input_scale,
        weight1_scale,
        weight2_scale,
        int(input_scale_mode) if input_scale_mode is not None else None,
        int(weight1_scale_mode) if weight1_scale_mode is not None else None,
        int(weight2_scale_mode) if weight2_scale_mode is not None else None,
        solution_index1,
        solution_index2,
    )
    return out


@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_create_extension() -> None: ...


@compile_ops("module_rocsolgemm")
def rocb_destroy_extension() -> None: ...


def gen_rocb_mm_fake_tensor(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: int
) -> torch.Tensor:
    # gemm out = (M, N) = (arg0.size(0), arg1.size(1)).
    return torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )


# torch-free kernel entry: writes into caller-allocated `result`.
@compile_ops("module_rocsolgemm", fc_name="rocb_mm", develop=True)
def _rocb_mm(
    arg0: torch.Tensor, arg1: torch.Tensor, result: torch.Tensor, arg2: int
) -> None: ...


def rocb_mm(arg0: torch.Tensor, arg1: torch.Tensor, arg2: int) -> torch.Tensor:
    result = torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )
    _rocb_mm(arg0, arg1, result, arg2)
    return result


direct_register_custom_op(
    "rocb_mm",
    rocb_mm,
    [],
    fake_impl=gen_rocb_mm_fake_tensor,
)


@compile_ops("module_rocsolgemm", fc_name="rocb_findallsols", develop=True)
def _rocb_findallsols(
    arg0: torch.Tensor, arg1: torch.Tensor, result: torch.Tensor
) -> list[int]: ...


def rocb_findallsols(arg0: torch.Tensor, arg1: torch.Tensor) -> list[int]:
    result = torch.empty(
        (arg0.size(0), arg1.size(1)), dtype=arg0.dtype, device=arg0.device
    )
    return _rocb_findallsols(arg0, arg1, result)
