# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops


@compile_ops("module_hipbsolgemm")
def hipb_create_extension() -> None:
    """Initialize hipBLASLt resources for the current device.

    Repeated calls on that device are no-ops. To change devices, synchronize
    outstanding work and destroy the existing extension before initializing
    again. Only one device's resources can be active in the module at a time.
    """


@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension() -> None:
    """Release hipBLASLt resources, allowing a later initialization.

    Synchronize outstanding GEMMs before calling. Destruction uses the owning
    device and restores the caller's current device. Without an active
    extension this is a no-op, including after a previous destruction.
    """


class HipblasLtMatmulScaleMode(IntEnum):
    SCALAR_32F = 0
    VEC32_UE8M0 = 2
    OUTER_VEC_32F = 3
    BLK32_UE8M0_32_8_EXT = 1001


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
    out: torch.Tensor | None = None,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
):
    return torch.empty(
        (mat1.size(0), mat2.size(1)),
        dtype=_hipb_mm_out_dtype(mat1, out_dtype, out),
        device=mat1.device,
    )


def _hipb_mm_out_dtype(
    mat1: torch.Tensor, out_dtype: torch.dtype | None, out: torch.Tensor | None
) -> torch.dtype:
    """The result dtype: ``out``'s when given, else ``out_dtype``, else the input's."""
    if out is None:
        return out_dtype if out_dtype is not None else mat1.dtype
    if out_dtype is not None and out_dtype != out.dtype:
        raise ValueError(f"hipb_mm: out_dtype={out_dtype} but out is {out.dtype}")
    return out.dtype


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
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
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
    out: torch.Tensor | None = None,
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
) -> torch.Tensor:
    """``out``: the [M, N] tensor to write instead of allocating; returned."""
    dtype = _hipb_mm_out_dtype(mat1, out_dtype, out)
    result = (
        torch.empty((mat1.size(0), mat2.size(1)), dtype=dtype, device=mat1.device)
        if out is None
        else out
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
        scaleA_mode,
        scaleB_mode,
    )
    return result


direct_register_custom_op(
    "hipb_mm",
    hipb_mm,
    ["out"],
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
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
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
    scaleA_mode: int | None = None,
    scaleB_mode: int | None = None,
) -> list[int]:
    out_dtype = out_dtype if out_dtype is not None else mat1.dtype
    result = torch.empty(
        (mat1.size(0), mat2.size(1)), dtype=out_dtype, device=mat1.device
    )
    return _hipb_findallsols(
        mat1,
        mat2,
        result,
        bias,
        scaleA,
        scaleB,
        scaleC,
        bpreshuffle,
        use_gelu,
        scaleA_mode,
        scaleB_mode,
    )


@compile_ops(
    "module_hipbsolgemm",
    fc_name="hipb_mm_epilogue_mm",
    ffi_type="pybind",
    develop=True,
)
def _hipb_mm_epilogue_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    mat3: torch.Tensor,
    stage_kinds: list[int],
    residual: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    gamma: torch.Tensor | None,
    eps: float,
    requant_scale_out: torch.Tensor | None,
    requant_scale_mode: int,
    intermediate: torch.Tensor,
    gemm_out: torch.Tensor,
    mat1_scale: torch.Tensor | None,
    mat2_scale: torch.Tensor | None,
    mat3_scale: torch.Tensor | None,
    mat1_scale_mode: int | None,
    mat2_scale_mode: int | None,
    mat3_scale_mode: int | None,
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
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    mat3: torch.Tensor,
    stage_kinds: list[int],
    residual: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    gamma: torch.Tensor | None,
    eps: float,
    requant_scale_out: torch.Tensor | None,
    requant_scale_mode: int,
    intermediate: torch.Tensor,
    out: torch.Tensor,
    mat1_scale: torch.Tensor | None,
    mat2_scale: torch.Tensor | None,
    mat3_scale: torch.Tensor | None,
    mat1_scale_mode: int | None,
    mat2_scale_mode: int | None,
    mat3_scale_mode: int | None,
    solution_index1: int | None,
    solution_index2: int | None,
) -> None:
    # The pybind wrapper selects the current device stream before entering C++.
    with torch.cuda.device(mat1.device):
        _hipb_mm_epilogue_mm(
            mat1,
            mat2,
            mat3,
            stage_kinds,
            residual,
            residual_out,
            gamma,
            eps,
            requant_scale_out,
            requant_scale_mode,
            intermediate,
            out,
            mat1_scale,
            mat2_scale,
            mat3_scale,
            mat1_scale_mode,
            mat2_scale_mode,
            mat3_scale_mode,
            solution_index1,
            solution_index2,
        )


@_hipb_mm_epilogue_mm_launch.register_fake
def _hipb_mm_epilogue_mm_launch_fake(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    mat3: torch.Tensor,
    stage_kinds: list[int],
    residual: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    gamma: torch.Tensor | None,
    eps: float,
    requant_scale_out: torch.Tensor | None,
    requant_scale_mode: int,
    intermediate: torch.Tensor,
    out: torch.Tensor,
    mat1_scale: torch.Tensor | None,
    mat2_scale: torch.Tensor | None,
    mat3_scale: torch.Tensor | None,
    mat1_scale_mode: int | None,
    mat2_scale_mode: int | None,
    mat3_scale_mode: int | None,
    solution_index1: int | None,
    solution_index2: int | None,
) -> None:
    del mat1, mat2, mat3, stage_kinds
    del residual, residual_out, gamma, eps
    del requant_scale_out, requant_scale_mode
    del intermediate, out, mat1_scale, mat2_scale, mat3_scale
    del mat1_scale_mode, mat2_scale_mode, mat3_scale_mode
    del solution_index1, solution_index2


def hipb_mm_epilogue_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    mat3: torch.Tensor,
    *,
    stages: tuple[FusedEpilogueStage, ...],
    intermediate_dtype: torch.dtype,
    dtype: torch.dtype,
    mat1_scale: torch.Tensor | None = None,
    mat2_scale: torch.Tensor | None = None,
    mat3_scale: torch.Tensor | None = None,
    mat1_scale_mode: HipblasLtMatmulScaleMode | None = None,
    mat2_scale_mode: HipblasLtMatmulScaleMode | None = None,
    mat3_scale_mode: HipblasLtMatmulScaleMode | None = None,
    out: torch.Tensor | None = None,
    solution_index1: int | None = None,
    solution_index2: int | None = None,
) -> torch.Tensor:
    """Run ``(mat1 @ mat2) @ mat3`` with fused intermediate stages.

    ``mat1`` has shape ``[M, K1]`` in row-major layout. ``mat2`` and ``mat3``
    have shapes ``[K1, K2]`` and ``[K2, N]`` in column-major layout. The
    returned tensor has shape ``[M, N]`` in row-major layout.

    Scale modes are optional. When a scale tensor is provided without a mode,
    hipBLASLt's default scalar FP32 mode is used. Operand scales follow the
    ``hipb_mm`` scale contract, including padded rows and uint8 or native E8M0
    block scales. Requant supports only ``BLK32_UE8M0_32_8_EXT``.

    Requires a hipBLASLt build with the fused epilogue API. Initialize the
    extension on the operand device before calling. Synchronize outstanding
    GEMMs before destroying the extension or changing its device.
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

    m = mat1.size(0)
    k2 = mat2.size(1)
    n = mat3.size(1)
    intermediate = torch.empty(
        (m, k2),
        dtype=intermediate_dtype,
        device=mat1.device,
    )
    if out is None:
        out = torch.empty((m, n), dtype=dtype, device=mat1.device)
    _hipb_mm_epilogue_mm_launch(
        mat1,
        mat2,
        mat3,
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_scale_mode,
        intermediate,
        out,
        mat1_scale,
        mat2_scale,
        mat3_scale,
        int(mat1_scale_mode) if mat1_scale_mode is not None else None,
        int(mat2_scale_mode) if mat2_scale_mode is not None else None,
        int(mat3_scale_mode) if mat3_scale_mode is not None else None,
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
