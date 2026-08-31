# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import torch
from torch import Tensor

from ..jit.core import compile_ops, get_gfx

__all__ = [
    "RMSNorm",
    "Requant",
    "RequantScaleGranularity",
    "ResidualAdd",
    "hipb_mm_epilogue_mm",
]


class RequantScaleGranularity(IntEnum):
    PER_BLOCK_MX = 2


@dataclass(frozen=True)
class ResidualAdd:
    residual: Tensor
    # TODO: Clarify whether None requests hipBLASLt's documented in-place
    # residual update or disables residual writeback, then add tests for the
    # selected behavior.
    residual_out: Tensor | None = None


@dataclass(frozen=True)
class RMSNorm:
    gamma: Tensor
    eps: float


@dataclass(frozen=True)
class Requant:
    scale_out: Tensor
    granularity: RequantScaleGranularity


FusedEpilogueStage: TypeAlias = ResidualAdd | RMSNorm | Requant

_RESIDUAL_ADD_STAGE = 0
_RMSNORM_STAGE = 1
_REQUANT_STAGE = 2


def _require_gfx950():
    arch = get_gfx()
    if arch != "gfx950":
        raise NotImplementedError(
            f"hipb_mm_epilogue_mm requires gfx950; current arch is {arch!r}"
        )


@compile_ops(
    "module_gemm_rmsnorm_gemm",
    fc_name="hipb_mm_epilogue_mm",
    ffi_type="pybind",
)
def _hipb_mm_epilogue_mm(
    A: Tensor,
    B1: Tensor,
    B2: Tensor,
    stage_kinds: list[int],
    residual: Tensor | None,
    residual_out: Tensor | None,
    gamma: Tensor,
    eps: float,
    requant_scale_out: Tensor | None,
    requant_granularity: int,
    producer_out_dtype: torch.dtype,
    gemm_out_dtype: torch.dtype,
    scaleA: Tensor | None,
    scaleB1: Tensor | None,
    scaleB2: Tensor | None,
    gemm_out: Tensor | None,
    solution_index1: int | None,
    solution_index2: int | None,
) -> Tensor: ...


def _flatten_epilogue_stages(
    stages: tuple[FusedEpilogueStage, ...],
) -> tuple[
    list[int],
    Tensor | None,
    Tensor | None,
    Tensor,
    float,
    Tensor | None,
    int,
]:
    stage_kinds: list[int] = []
    residual = None
    residual_out = None
    gamma = None
    eps = 0.0
    requant_scale_out = None
    requant_granularity = -1

    for stage in stages:
        if isinstance(stage, ResidualAdd):
            stage_kinds.append(_RESIDUAL_ADD_STAGE)
            residual = stage.residual
            residual_out = stage.residual_out
        elif isinstance(stage, RMSNorm):
            stage_kinds.append(_RMSNORM_STAGE)
            gamma = stage.gamma
            eps = stage.eps
        elif isinstance(stage, Requant):
            stage_kinds.append(_REQUANT_STAGE)
            requant_scale_out = stage.scale_out
            requant_granularity = int(stage.granularity)
        else:
            raise TypeError(f"unsupported fused epilogue stage: {type(stage).__name__}")

    if gamma is None:
        raise ValueError(
            "stages must include RMSNorm because the second GEMM uses "
            "RMSNormScaleApply"
        )
    return (
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_granularity,
    )


def hipb_mm_epilogue_mm(
    A: Tensor,
    B1: Tensor,
    B2: Tensor,
    *,
    stages: tuple[FusedEpilogueStage, ...],
    producer_out_dtype: torch.dtype = torch.bfloat16,
    gemm_out_dtype: torch.dtype = torch.bfloat16,
    scaleA: Tensor | None = None,
    scaleB1: Tensor | None = None,
    scaleB2: Tensor | None = None,
    gemm_out: Tensor | None = None,
    solution_index1: int | None = None,
    solution_index2: int | None = None,
) -> Tensor:
    """Run a managed GEMM -> split RMSNorm stages -> GEMM pair."""
    _require_gfx950()
    (
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_granularity,
    ) = _flatten_epilogue_stages(stages)
    return _hipb_mm_epilogue_mm(
        A,
        B1,
        B2,
        stage_kinds,
        residual,
        residual_out,
        gamma,
        eps,
        requant_scale_out,
        requant_granularity,
        producer_out_dtype,
        gemm_out_dtype,
        scaleA,
        scaleB1,
        scaleB2,
        gemm_out,
        solution_index1,
        solution_index2,
    )
