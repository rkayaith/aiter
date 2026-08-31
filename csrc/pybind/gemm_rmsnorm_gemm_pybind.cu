// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <pybind11/pybind11.h>
#include <torch/all.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

#define CHECK_HIPBLAS(expr)                                                             \
    do {                                                                                \
        hipblasStatus_t status_ = (expr);                                               \
        if (status_ != HIPBLAS_STATUS_SUCCESS)                                          \
            throw std::runtime_error(std::string("hipblaslt error at " #expr " = ") +  \
                                     std::to_string(static_cast<int>(status_)));        \
    } while (0)

// PyTorch loads system libhipblaslt.so.1 first; that version pre-dates the
// fused-epilogue API and the SONAME cache prevents the custom build from
// replacing it via RTLD_GLOBAL.  Opening the custom build by its ABSOLUTE PATH
// bypasses the SONAME cache and lets us retrieve each symbol via dlsym.
//
// AITER_HIPBLASLT_PATH (absolute path to the custom libhipblaslt.so.1) drives
// BOTH the compile-time flags (-I, -L, -rpath, -DCUSTOM_HIPBLASLT_PATH,
// injected by optCompilerConfig.json) AND the runtime dlopen path used by
// getHipblasltPath below.
#ifndef CUSTOM_HIPBLASLT_PATH
#  define CUSTOM_HIPBLASLT_PATH \
    "libhipblaslt.so.1"
#endif

static std::string getHipblasltPath()
{
    const char* env = std::getenv("AITER_HIPBLASLT_PATH");
    return env ? std::string(env) : std::string(CUSTOM_HIPBLASLT_PATH);
}

// ─── function pointer types ───────────────────────────────────────────────────
using pfnCreate =
    hipblasStatus_t (*)(hipblasLtHandle_t*);
using pfnMatrixLayoutCreate =
    hipblasStatus_t (*)(hipblasLtMatrixLayout_t*, hipDataType, uint64_t, uint64_t, int64_t);
using pfnMatrixLayoutDestroy =
    hipblasStatus_t (*)(const hipblasLtMatrixLayout_t);
using pfnMatmulDescCreate =
    hipblasStatus_t (*)(hipblasLtMatmulDesc_t*, hipblasComputeType_t, hipDataType);
using pfnMatmulDescSetAttribute =
    hipblasStatus_t (*)(hipblasLtMatmulDesc_t, hipblasLtMatmulDescAttributes_t,
                        const void*, size_t);
using pfnMatmulDescDestroy =
    hipblasStatus_t (*)(const hipblasLtMatmulDesc_t);
using pfnMatmulPreferenceCreate =
    hipblasStatus_t (*)(hipblasLtMatmulPreference_t*);
using pfnMatmulPreferenceSetAttribute =
    hipblasStatus_t (*)(hipblasLtMatmulPreference_t,
                        hipblasLtMatmulPreferenceAttributes_t, const void*, size_t);
using pfnMatmulPreferenceDestroy =
    hipblasStatus_t (*)(const hipblasLtMatmulPreference_t);
using pfnMatmulAlgoGetHeuristic =
    hipblasStatus_t (*)(hipblasLtHandle_t, hipblasLtMatmulDesc_t,
                        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
                        hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t,
                        hipblasLtMatmulPreference_t, int,
                        hipblasLtMatmulHeuristicResult_t*, int*);
using pfnMatmul =
    hipblasStatus_t (*)(hipblasLtHandle_t, hipblasLtMatmulDesc_t,
                        const void*, const void*, hipblasLtMatrixLayout_t,
                        const void*, hipblasLtMatrixLayout_t,
                        const void*, const void*, hipblasLtMatrixLayout_t,
                        void*, hipblasLtMatrixLayout_t,
                        const hipblasLtMatmulAlgo_t*, void*, size_t, hipStream_t);
using pfnFusedEpilogueCreate =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t*);
using pfnFusedEpilogueAdd =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t,
                        hipblasLtFuseableEpilogue_t);
using pfnFusedEpilogueSetAttribute =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t,
                        hipblasLtFusedEpilogueAttribute_t, const void*, size_t);
using pfnFusedEpilogueDestroy =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueDescriptor_t);
using pfnFusedEpilogueRMSNormDescriptorCreate =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueRMSNormDescriptor_t*);
using pfnFusedEpilogueRMSNormDescriptorDestroy =
    hipblasStatus_t (*)(hipblasLtFusedEpilogueRMSNormDescriptor_t);

// ─── vtable ──────────────────────────────────────────────────────────────────
// All calls go through this table so they always use the custom library,
// regardless of which hipblaslt version the SONAME cache returns.
struct HbltVtable {
    void*                                    lib;
    pfnCreate                                create;
    pfnMatrixLayoutCreate                    matrixLayoutCreate;
    pfnMatrixLayoutDestroy                   matrixLayoutDestroy;
    pfnMatmulDescCreate                      matmulDescCreate;
    pfnMatmulDescSetAttribute                matmulDescSetAttribute;
    pfnMatmulDescDestroy                     matmulDescDestroy;
    pfnMatmulPreferenceCreate                matmulPreferenceCreate;
    pfnMatmulPreferenceSetAttribute          matmulPreferenceSetAttribute;
    pfnMatmulPreferenceDestroy               matmulPreferenceDestroy;
    pfnMatmulAlgoGetHeuristic                matmulAlgoGetHeuristic;
    pfnMatmul                                matmul;
    pfnFusedEpilogueCreate                   fusedEpilogueCreate;
    pfnFusedEpilogueAdd                      fusedEpilogueAdd;
    pfnFusedEpilogueSetAttribute             fusedEpilogueSetAttribute;
    pfnFusedEpilogueDestroy                  fusedEpilogueDestroy;
    pfnFusedEpilogueRMSNormDescriptorCreate  rmsNormDescCreate;
    pfnFusedEpilogueRMSNormDescriptorDestroy rmsNormDescDestroy;

    template<typename F>
    F loadSym(const char* name) const
    {
        void* sym = dlsym(lib, name);
        if (!sym)
            throw std::runtime_error(std::string("hipblaslt symbol not found: ") + name);
        return reinterpret_cast<F>(sym);
    }

    HbltVtable()
    {
        lib = dlopen(getHipblasltPath().c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (!lib)
            throw std::runtime_error(
                std::string("cannot dlopen custom hipblaslt: ") + dlerror());
        create                       = loadSym<pfnCreate>("hipblasLtCreate");
        matrixLayoutCreate           = loadSym<pfnMatrixLayoutCreate>("hipblasLtMatrixLayoutCreate");
        matrixLayoutDestroy          = loadSym<pfnMatrixLayoutDestroy>("hipblasLtMatrixLayoutDestroy");
        matmulDescCreate             = loadSym<pfnMatmulDescCreate>("hipblasLtMatmulDescCreate");
        matmulDescSetAttribute       = loadSym<pfnMatmulDescSetAttribute>("hipblasLtMatmulDescSetAttribute");
        matmulDescDestroy            = loadSym<pfnMatmulDescDestroy>("hipblasLtMatmulDescDestroy");
        matmulPreferenceCreate       = loadSym<pfnMatmulPreferenceCreate>("hipblasLtMatmulPreferenceCreate");
        matmulPreferenceSetAttribute = loadSym<pfnMatmulPreferenceSetAttribute>(
            "hipblasLtMatmulPreferenceSetAttribute");
        matmulPreferenceDestroy      = loadSym<pfnMatmulPreferenceDestroy>("hipblasLtMatmulPreferenceDestroy");
        matmulAlgoGetHeuristic       = loadSym<pfnMatmulAlgoGetHeuristic>("hipblasLtMatmulAlgoGetHeuristic");
        matmul                       = loadSym<pfnMatmul>("hipblasLtMatmul");
        fusedEpilogueCreate          = loadSym<pfnFusedEpilogueCreate>("hipblasLtFusedEpilogueCreate");
        fusedEpilogueAdd             = loadSym<pfnFusedEpilogueAdd>("hipblasLtFusedEpilogueAdd");
        fusedEpilogueSetAttribute    = loadSym<pfnFusedEpilogueSetAttribute>(
            "hipblasLtFusedEpilogueSetAttribute");
        fusedEpilogueDestroy         = loadSym<pfnFusedEpilogueDestroy>("hipblasLtFusedEpilogueDestroy");
        rmsNormDescCreate            = loadSym<pfnFusedEpilogueRMSNormDescriptorCreate>(
            "hipblasLtFusedEpilogueRMSNormDescriptorCreate");
        rmsNormDescDestroy           = loadSym<pfnFusedEpilogueRMSNormDescriptorDestroy>(
            "hipblasLtFusedEpilogueRMSNormDescriptorDestroy");
    }

    // Run custom-hipblaslt cleanup callbacks at process exit.
    ~HbltVtable()
    {
        if (lib)
            dlclose(lib);
    }
    HbltVtable(const HbltVtable&)            = delete;
    HbltVtable& operator=(const HbltVtable&) = delete;

    static const HbltVtable& get()
    {
        static HbltVtable instance;
        return instance;
    }
};

// RAII guards that destroy hipblaslt objects via the loaded vtable on scope exit.
struct MatrixLayoutGuard {
    hipblasLtMatrixLayout_t handle = nullptr;
    MatrixLayoutGuard()                                        = default;
    MatrixLayoutGuard(const MatrixLayoutGuard&)                = delete;
    MatrixLayoutGuard& operator=(const MatrixLayoutGuard&)     = delete;
    ~MatrixLayoutGuard() { if (handle) HbltVtable::get().matrixLayoutDestroy(handle); }
};
struct MatmulDescGuard {
    hipblasLtMatmulDesc_t handle = nullptr;
    MatmulDescGuard()                                      = default;
    MatmulDescGuard(const MatmulDescGuard&)                = delete;
    MatmulDescGuard& operator=(const MatmulDescGuard&)     = delete;
    ~MatmulDescGuard() { if (handle) HbltVtable::get().matmulDescDestroy(handle); }
};
struct MatmulPreferenceGuard {
    hipblasLtMatmulPreference_t handle = nullptr;
    MatmulPreferenceGuard()                                          = default;
    MatmulPreferenceGuard(const MatmulPreferenceGuard&)              = delete;
    MatmulPreferenceGuard& operator=(const MatmulPreferenceGuard&)   = delete;
    ~MatmulPreferenceGuard() { if (handle) HbltVtable::get().matmulPreferenceDestroy(handle); }
};
struct FusedEpilogueGuard {
    hipblasLtFusedEpilogueDescriptor_t handle = nullptr;
    FusedEpilogueGuard()                                       = default;
    FusedEpilogueGuard(const FusedEpilogueGuard&)              = delete;
    FusedEpilogueGuard& operator=(const FusedEpilogueGuard&)   = delete;
    ~FusedEpilogueGuard() { if (handle) HbltVtable::get().fusedEpilogueDestroy(handle); }
};
struct RmsNormStatsGuard {
    hipblasLtFusedEpilogueRMSNormDescriptor_t handle = nullptr;
    RmsNormStatsGuard()                                        = default;
    RmsNormStatsGuard(const RmsNormStatsGuard&)                = delete;
    RmsNormStatsGuard& operator=(const RmsNormStatsGuard&)     = delete;
    ~RmsNormStatsGuard() { if (handle) HbltVtable::get().rmsNormDescDestroy(handle); }
};

static hipblasLtHandle_t getHandle(int deviceIndex)
{
    static std::mutex mtx;
    static std::unordered_map<int, hipblasLtHandle_t> handles;
    std::lock_guard<std::mutex> lock(mtx);
    hipblasLtHandle_t& handle = handles[deviceIndex];
    if (!handle) {
        if (HbltVtable::get().create(&handle) != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error("failed to create hipblaslt handle");
    }
    return handle;
}

// Returns a pointer to a persistent, lazily-grown device workspace.
// The buffer lives for the lifetime of the process and is never freed, so
// hipblaslt can reuse it across calls without hitting the PyTorch allocator.
static std::pair<void*, size_t> getWorkspace(at::Device device)
{
    // 256 MB is the ceiling hipblaslt currently needs for fused-epilogue solutions.
    static constexpr size_t kWsSize = static_cast<size_t>(256) * 1024 * 1024;
    static std::mutex mtx;
    static std::unordered_map<int, torch::Tensor> workspaces;
    std::lock_guard<std::mutex> lock(mtx);
    const int deviceIndex = static_cast<int>(device.index());
    torch::Tensor& ws = workspaces[deviceIndex];
    if (!ws.defined())
        ws = torch::empty({static_cast<int64_t>(kWsSize)},
                          torch::TensorOptions().dtype(torch::kUInt8).device(device));
    return {ws.data_ptr(), kWsSize};
}

// Generalized configureMatmulLayouts with typed A/B/CD element types.
static void configureMatmulLayoutsTyped(const HbltVtable& v,
                                        hipDataType aType, hipDataType bType, hipDataType cdType,
                                        int64_t m, int64_t n, int64_t k, int64_t lda,
                                        hipblasLtFusedEpilogueDescriptor_t fused,
                                        MatrixLayoutGuard& gLayA, MatrixLayoutGuard& gLayB,
                                        MatrixLayoutGuard& gLayC, MatrixLayoutGuard& gLayD,
                                        MatmulDescGuard& gMm)
{
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayA.handle, aType,  k, m, lda));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayB.handle, bType,  k, n, k));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayC.handle, cdType, m, n, m));
    CHECK_HIPBLAS(v.matrixLayoutCreate(&gLayD.handle, cdType, m, n, m));
    CHECK_HIPBLAS(v.matmulDescCreate(&gMm.handle, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    CHECK_HIPBLAS(v.matmulDescSetAttribute(gMm.handle, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(gMm.handle, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(
        gMm.handle, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused)));
}

static void setScaleAttrib(hipblasLtMatmulDesc_t mm,
                           hipblasLtMatmulDescAttributes_t modeAttr,
                           hipblasLtMatmulDescAttributes_t ptrAttr,
                           const void* scale,
                           hipblasLtMatmulMatrixScale_t mode)
{
    if (!scale)
        return;
    const HbltVtable& v = HbltVtable::get();
    CHECK_HIPBLAS(v.matmulDescSetAttribute(mm, modeAttr, &mode, sizeof(mode)));
    CHECK_HIPBLAS(v.matmulDescSetAttribute(mm, ptrAttr, &scale, sizeof(scale)));
}

// Generalized TN fused matmul with typed A/B/CD types and optional input scales.
static void runTnFused(hipblasLtHandle_t                  handle,
                       hipDataType                        aType,
                       hipDataType                        bType,
                       hipDataType                        cdType,
                       int64_t                            m,
                       int64_t                            n,
                       int64_t                            k,
                       const void*                        dA,
                       int64_t                            lda,
                       const void*                        dScaleA,
                       const void*                        dB,
                       const void*                        dScaleB,
                       void*                              dC,
                       void*                              dD,
                       hipblasLtFusedEpilogueDescriptor_t fused,
                       void*                              dWs,
                       size_t                             wsSize,
                       hipStream_t                        stream,
                       hipblasLtMatmulMatrixScale_t       scaleAMode
                           = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT,
                       hipblasLtMatmulMatrixScale_t       scaleBMode
                           = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT)
{
    const HbltVtable& v = HbltVtable::get();
    MatrixLayoutGuard gLayA, gLayB, gLayC, gLayD;
    MatmulDescGuard gMm;
    configureMatmulLayoutsTyped(v, aType, bType, cdType, m, n, k, lda, fused,
                                gLayA, gLayB, gLayC, gLayD, gMm);
    setScaleAttrib(gMm.handle,
                   HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                   HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                   dScaleA,
                   scaleAMode);
    setScaleAttrib(gMm.handle,
                   HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                   HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                   dScaleB,
                   scaleBMode);
    MatmulPreferenceGuard gPref;
    CHECK_HIPBLAS(v.matmulPreferenceCreate(&gPref.handle));
    CHECK_HIPBLAS(v.matmulPreferenceSetAttribute(
        gPref.handle, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
    hipblasLtMatmulHeuristicResult_t heur[1];
    int algoCount = 0;
    CHECK_HIPBLAS(v.matmulAlgoGetHeuristic(
        handle, gMm.handle, gLayA.handle, gLayB.handle, gLayC.handle, gLayD.handle,
        gPref.handle, 1, heur, &algoCount));
    if (algoCount <= 0)
        throw std::runtime_error("no hipblaslt fused solution selected (algoCount==0)");
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_HIPBLAS(v.matmul(handle, gMm.handle, &alpha, dA, gLayA.handle, dB, gLayB.handle,
                           &beta, dC, gLayC.handle, dD, gLayD.handle,
                           &heur[0].algo, dWs, wsSize, stream));
}

// Builds the consumer fused epilogue: RMSNORM_SCALE_APPLY reading from stats.
static void buildConsumerEpilogue(const HbltVtable& v,
                                  hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                  FusedEpilogueGuard& cons)
{
    CHECK_HIPBLAS(v.fusedEpilogueCreate(&cons.handle));
    CHECK_HIPBLAS(v.fusedEpilogueAdd(cons.handle, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY));
    CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
        cons.handle, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)));
}

// Returns the flat size of the MX scale byte buffer for a [mTok,nHid] tile.
static int64_t mxfp8ScaleBufferSize(int64_t mTok, int64_t nHid)
{
    const int64_t paddedRows = ((mTok + 31) / 32) * 32;
    const int64_t nTiles     = (nHid + 31) / 32;
    const int64_t paddedCols = ((nTiles + 7) / 8) * 8;
    return paddedRows * paddedCols;
}

static hipDataType chainOperandType(const torch::Tensor& operand, const char* name)
{
    if (operand.scalar_type() == at::kBFloat16)
        return HIP_R_16BF;
    if (operand.scalar_type() == at::kFloat8_e4m3fn)
        return HIP_R_8F_E4M3;
    TORCH_CHECK(false, name, " must be bf16 or fp8 e4m3");
    return HIP_R_16BF;
}

static void validateOptionalMxScale(const std::optional<torch::Tensor>& scale,
                                    const torch::Tensor& operand,
                                    const char* name)
{
    if (!scale)
        return;
    TORCH_CHECK(scale->is_cuda(), name, " must be cuda");
    TORCH_CHECK(scale->device() == operand.device(), name, " must be on the operand device");
    TORCH_CHECK(scale->scalar_type() == at::kByte, name, " must be uint8");
    TORCH_CHECK(scale->numel()
                    == mxfp8ScaleBufferSize(operand.size(0), operand.size(1)),
                name,
                " has the wrong size for its operand");
}

static void buildPairProducerEpilogue(
    const HbltVtable& v,
    const std::vector<int64_t>& stageKinds,
    void* residualPtr,
    void* residualOutputPtr,
    void* gammaPtr,
    float eps,
    void* requantScaleOutPtr,
    int64_t requantGranularity,
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
    FusedEpilogueGuard& producer)
{
    CHECK_HIPBLAS(v.fusedEpilogueCreate(&producer.handle));
    const bool hasRequant
        = std::find(stageKinds.begin(), stageKinds.end(), 2) != stageKinds.end();
    const bool mxRequant
        = hasRequant && requantGranularity == HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;

    for (const int64_t stageKind : stageKinds) {
        if (stageKind == 0) {
            CHECK_HIPBLAS(v.fusedEpilogueAdd(
                producer.handle, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD));
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
                &residualPtr,
                sizeof(residualPtr)));
            if (residualOutputPtr) {
                const hipblasLtFusedEpilogueAttribute_t outputAttribute
                    = mxRequant
                          ? HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER
                          : HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER;
                CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                    producer.handle,
                    outputAttribute,
                    &residualOutputPtr,
                    sizeof(residualOutputPtr)));
            }
            continue;
        }
        if (stageKind == 1) {
            CHECK_HIPBLAS(v.fusedEpilogueAdd(
                producer.handle, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS));
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                &gammaPtr,
                sizeof(gammaPtr)));
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                &eps,
                sizeof(eps)));
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS,
                &stats,
                sizeof(stats)));
            continue;
        }

        CHECK_HIPBLAS(v.fusedEpilogueAdd(
            producer.handle, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT));
        const hipblasLtRequantScaleComputeMode_t computeMode
            = HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX;
        const auto granularity
            = static_cast<hipblasLtRequantScaleGranularity_t>(requantGranularity);
        CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
            producer.handle,
            HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE,
            &computeMode,
            sizeof(computeMode)));
        CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
            producer.handle,
            HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
            &granularity,
            sizeof(granularity)));
        if (mxRequant) {
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER,
                &requantScaleOutPtr,
                sizeof(requantScaleOutPtr)));
            const int32_t blockSize = 32;
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE,
                &blockSize,
                sizeof(blockSize)));
            const hipDataType outputType = HIP_R_8F_E4M3;
            CHECK_HIPBLAS(v.fusedEpilogueSetAttribute(
                producer.handle,
                HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE,
                &outputType,
                sizeof(outputType)));
        }
    }
}

static void validatePairOutput(const torch::Tensor& output,
                               int64_t m,
                               int64_t n,
                               at::ScalarType dtype,
                               const at::Device& device)
{
    TORCH_CHECK(output.is_cuda(), "gemm_out must be cuda");
    TORCH_CHECK(output.device() == device, "gemm_out must be on the input device");
    TORCH_CHECK(output.scalar_type() == dtype, "gemm_out dtype must match gemm_out_dtype");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == m && output.size(1) == n,
                "gemm_out must have shape [M, Nout]");
    TORCH_CHECK(output.stride(1) == m && (m == 1 || output.stride(0) == 1),
                "gemm_out must use the native column-major [M, N] layout");
}

torch::Tensor hipb_mm_epilogue_mm(
    torch::Tensor A,
    torch::Tensor B1,
    torch::Tensor B2,
    std::vector<int64_t> stageKinds,
    std::optional<torch::Tensor> residual,
    std::optional<torch::Tensor> residualOut,
    torch::Tensor gamma,
    double eps,
    std::optional<torch::Tensor> requantScaleOut,
    int64_t requantGranularity,
    c10::ScalarType producerOutDtype,
    c10::ScalarType gemmOutDtype,
    std::optional<torch::Tensor> scaleA,
    std::optional<torch::Tensor> scaleB1,
    std::optional<torch::Tensor> scaleB2,
    std::optional<torch::Tensor> gemmOut,
    std::optional<int64_t> solutionIndex1,
    std::optional<int64_t> solutionIndex2)
{
    TORCH_CHECK(!solutionIndex1 && !solutionIndex2,
                "explicit solution indices are not supported by the custom hipBLASLt loader");
    TORCH_CHECK(A.is_cuda() && B1.is_cuda() && B2.is_cuda() && gamma.is_cuda(),
                "A, B1, B2, and gamma must be cuda tensors");
    TORCH_CHECK(A.device() == B1.device() && A.device() == B2.device()
                    && A.device() == gamma.device(),
                "A, B1, B2, and gamma must be on the same device");
    TORCH_CHECK(A.dim() == 2 && B1.dim() == 2 && B2.dim() == 2 && gamma.dim() == 1,
                "A, B1, B2, and gamma must be two-, two-, two-, and one-dimensional");

    const int64_t mTok = A.size(0);
    const int64_t k1 = A.size(1);
    const int64_t nHid = B1.size(0);
    const int64_t nOut = B2.size(0);
    TORCH_CHECK(B1.size(1) == k1, "B1 columns must equal A columns");
    TORCH_CHECK(B2.size(1) == nHid, "B2 columns must equal B1 rows");
    TORCH_CHECK(gamma.size(0) == nHid, "gamma length must equal B1 rows");
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16, "gamma must be bfloat16");

    const hipDataType aType = chainOperandType(A, "A");
    const hipDataType b1Type = chainOperandType(B1, "B1");
    const hipDataType b2Type = chainOperandType(B2, "B2");
    const bool hasRequant
        = std::find(stageKinds.begin(), stageKinds.end(), 2) != stageKinds.end();
    TORCH_CHECK(producerOutDtype == at::kBFloat16
                    || producerOutDtype == at::kFloat8_e4m3fn,
                "producer_out_dtype must be bfloat16 or float8_e4m3fn");
    TORCH_CHECK(gemmOutDtype == at::kBFloat16,
                "the fused pair currently supports only bfloat16 final output");

    validateOptionalMxScale(scaleA, A, "scaleA");
    validateOptionalMxScale(scaleB1, B1, "scaleB1");
    validateOptionalMxScale(scaleB2, B2, "scaleB2");

    if (residual) {
        TORCH_CHECK(residual->is_cuda() && residual->device() == A.device(),
                    "residual must be a cuda tensor on the input device");
        TORCH_CHECK(residual->scalar_type() == at::kBFloat16, "residual must be bfloat16");
        TORCH_CHECK(residual->dim() == 2 && residual->size(0) == mTok
                        && residual->size(1) == nHid,
                    "residual must have shape [M, Nhidden]");
        TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    }
    if (residualOut) {
        TORCH_CHECK(residualOut->is_cuda() && residualOut->device() == A.device(),
                    "residual_out must be a cuda tensor on the input device");
        TORCH_CHECK(residualOut->scalar_type() == at::kBFloat16,
                    "residual_out must be bfloat16");
        TORCH_CHECK(residualOut->dim() == 2 && residualOut->size(0) == mTok
                        && residualOut->size(1) == nHid,
                    "residual_out must have shape [M, Nhidden]");
        TORCH_CHECK(residualOut->is_contiguous(), "residual_out must be contiguous");
    }
    if (requantScaleOut) {
        TORCH_CHECK(requantScaleOut->is_cuda() && requantScaleOut->device() == A.device(),
                    "Requant.scale_out must be a cuda tensor on the input device");
        TORCH_CHECK(requantScaleOut->is_contiguous(), "Requant.scale_out must be contiguous");
        TORCH_CHECK(requantScaleOut->scalar_type() == at::kByte,
                    "MX Requant.scale_out must be uint8");
        TORCH_CHECK(requantScaleOut->numel() == mxfp8ScaleBufferSize(mTok, nHid),
                    "MX Requant.scale_out has the wrong size");
    }

    torch::Tensor aC = A.contiguous();
    torch::Tensor b1C = B1.contiguous();
    torch::Tensor b2C = B2.contiguous();
    torch::Tensor gammaC = gamma.contiguous();
    std::optional<torch::Tensor> scaleAC;
    std::optional<torch::Tensor> scaleB1C;
    std::optional<torch::Tensor> scaleB2C;
    if (scaleA)
        scaleAC = scaleA->contiguous();
    if (scaleB1)
        scaleB1C = scaleB1->contiguous();
    if (scaleB2)
        scaleB2C = scaleB2->contiguous();

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA deviceGuard{A.device()};
    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    auto intermediateOptions = torch::TensorOptions().dtype(producerOutDtype).device(A.device());
    torch::Tensor intermediate = torch::empty({mTok, nHid}, intermediateOptions);

    torch::Tensor output;
    if (gemmOut) {
        validatePairOutput(*gemmOut, mTok, nOut, gemmOutDtype, A.device());
        output = *gemmOut;
    } else {
        auto outputOptions = torch::TensorOptions().dtype(gemmOutDtype).device(A.device());
        output = torch::empty({nOut, mTok}, outputOptions).transpose(0, 1);
    }

    auto [workspace, workspaceSize] = getWorkspace(A.device());
    const HbltVtable& v = HbltVtable::get();
    hipblasLtHandle_t handle = getHandle(static_cast<int>(A.device().index()));
    RmsNormStatsGuard stats;
    CHECK_HIPBLAS(v.rmsNormDescCreate(&stats.handle));

    void* residualPtr = residual ? residual->data_ptr() : nullptr;
    void* residualOutputPtr = residualOut ? residualOut->data_ptr() : nullptr;
    void* requantScaleOutPtr
        = requantScaleOut ? requantScaleOut->data_ptr() : nullptr;
    FusedEpilogueGuard producer;
    buildPairProducerEpilogue(v,
                              stageKinds,
                              residualPtr,
                              residualOutputPtr,
                              gammaC.data_ptr(),
                              static_cast<float>(eps),
                              requantScaleOutPtr,
                              requantGranularity,
                              stats.handle,
                              producer);
    runTnFused(handle,
               aType,
               b1Type,
               producerOutDtype == at::kBFloat16 ? HIP_R_16BF : HIP_R_8F_E4M3,
               mTok,
               nHid,
               k1,
               aC.data_ptr(),
               k1,
               scaleAC ? scaleAC->data_ptr() : nullptr,
               b1C.data_ptr(),
               scaleB1C ? scaleB1C->data_ptr() : nullptr,
               intermediate.data_ptr(),
               intermediate.data_ptr(),
               producer.handle,
               workspace,
               workspaceSize,
               stream);

    FusedEpilogueGuard consumer;
    buildConsumerEpilogue(v, stats.handle, consumer);
    const void* consumerScale = hasRequant ? requantScaleOutPtr : nullptr;
    const auto consumerScaleMode
        = hasRequant ? HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT
                     : HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    runTnFused(handle,
               producerOutDtype == at::kBFloat16 ? HIP_R_16BF : HIP_R_8F_E4M3,
               b2Type,
               HIP_R_16BF,
               mTok,
               nOut,
               nHid,
               intermediate.data_ptr(),
               nHid,
               consumerScale,
               b2C.data_ptr(),
               scaleB2C ? scaleB2C->data_ptr() : nullptr,
               output.data_ptr(),
               output.data_ptr(),
               consumer.handle,
               workspace,
               workspaceSize,
               stream,
               consumerScaleMode);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hipb_mm_epilogue_mm",
          &hipb_mm_epilogue_mm,
          "Managed fused GEMM, split RMSNorm stages, and GEMM pair",
          py::arg("A"),
          py::arg("B1"),
          py::arg("B2"),
          py::arg("stage_kinds"),
          py::arg("residual"),
          py::arg("residual_out"),
          py::arg("gamma"),
          py::arg("eps"),
          py::arg("requant_scale_out"),
          py::arg("requant_granularity"),
          py::arg("producer_out_dtype"),
          py::arg("gemm_out_dtype"),
          py::arg("scaleA"),
          py::arg("scaleB1"),
          py::arg("scaleB2"),
          py::arg("gemm_out"),
          py::arg("solution_index1"),
          py::arg("solution_index2"));
}
