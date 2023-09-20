/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
#  include "cutlass/util/cublas_wrappers.hpp"
#endif
#include "cutlass/util/helper_cuda.hpp"

// Extra includes
/// collective builder
// #include "cutlass/util/command_line.h"
// #include "cutlass/kernel_hardware_info.hpp"

// #include "cutlass/cutlass.h"
// #include "cutlass/tensor_ref.h"
// #include "cutlass/epilogue/collective/default_epilogue.hpp"
// #include "cutlass/epilogue/thread/linear_combination.h"
// #include "cutlass/gemm/dispatch_policy.hpp"
// #include "cutlass/gemm/collective/collective_builder.hpp"
// #include "cutlass/epilogue/collective/collective_builder.hpp"
// #include "cutlass/gemm/device/gemm_universal_adapter.h"
// #include "cutlass/gemm/kernel/gemm_universal.hpp"
// #include "cutlass/gemm/kernel/tile_scheduler.hpp"

// #include "cutlass/util/command_line.h"
// #include "cutlass/util/distribution.h"
// #include "cutlass/util/host_tensor.h"
// #include "cutlass/util/packed_stride.hpp"
// #include "cutlass/util/tensor_view_io.h"
// #include "cutlass/util/reference/device/gemm_complex.h"
// #include "cutlass/util/reference/device/tensor_compare.h"
// #include "cutlass/util/reference/device/tensor_fill.h"
//// from example 47
#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
// Extra includes end


template <class MShape, class NShape, class KShape,
          class TA, class AStride, class ABlockLayout, class AThreadLayout,
          class TB, class BStride, class BBlockLayout, class BThreadLayout,
          class TC, class CStride, class CBlockLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(MShape M, NShape N, KShape K,
            TA const* A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
            TB const* B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
            TC      * C, CStride dC, CBlockLayout       , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  using namespace cute;
  using X = Underscore;

  // Preconditions
  CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
  CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

  CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
  CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
  CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

  //CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));      // BLK_M
  //CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));      // BLK_N
  CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB));        // BLK_K

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ABlockLayout>];
  __shared__ TB smemB[cosize_v<BBlockLayout>];
  auto sA = make_tensor(make_smem_ptr(smemA), blockA);               // (BLK_M,BLK_K)
  auto sB = make_tensor(make_smem_ptr(smemB), blockB);               // (BLK_N,BLK_K)

  // Represent the full tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);      // (M,K)
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);      // (N,K)
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block --
  // potential for thread block locality
  auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));// (BLK_M,BLK_N,BLK_K)
  auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);            // (m,n,k)

  auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  auto gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of simple partitioning of A|B tiles over tA|tB
  //   Default is a raked partition, but can be changed with Step<X,Y> parameter

  auto tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  auto tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  auto tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  auto tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  //
  // Define C accumulators and A/B partitioning
  //

  // TUTORIAL: Example of partitioning via projections of tC

  // Partition sA (M,K) by the rows of tC
  auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (N,K) by the cols of tC
  auto tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same size as the projected data
  auto tCrC = make_fragment_like(tCgC);                              // (THR_M,THR_N)

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("mA\n");
    print(mA.shape()); print("\n"); print(mA.stride());
    print("\n\ngA\n");
    print(gA.shape()); print("\n"); print(gA.stride());
    print("\n\ntAgA\n");
    print(tAgA.shape()); print("\n"); print(tAgA.stride());
    print("\n\nsA\n");
    print(sA.shape()); print("\n"); print(sA.stride());
    print("\n\ntAsA\n");
    print(tAsA.shape()); print("\n"); print(tAsA.stride());
    print("\n\n");
  }
#endif

#if 0
  if(thread0()) {
    print("mB\n");
    print(mB.shape()); print("\n"); print(mB.stride());
    print("\n\ngB\n");
    print(gB.shape()); print("\n"); print(gB.stride());
    print("\n\ntBgB\n");
    print(tBgB.shape()); print("\n"); print(tBgB.stride());
    print("\n\nsB\n");
    print(sB.shape()); print("\n"); print(sB.stride());
    print("\n\ntBsB\n");
    print(tBsB.shape()); print("\n"); print(tBsB.stride());
    print("\n\n");
  }
#endif

#if 0
  if(thread0()) {
    print("mC\n");
    print(mC.shape()); print("\n"); print(mC.stride());
    print("\n\ngC\n");
    print(gC.shape()); print("\n"); print(gC.stride());
    print("\n\ntCsA\n");
    print(tCsA.shape()); print("\n"); print(tCsA.stride());
    print("\n\ntCsB\n");
    print(tCsB.shape()); print("\n"); print(tCsB.stride());
    print("\n\ntCgC\n");
    print(tCgC.shape()); print("\n"); print(tCgC.stride());
    print("\n\ntCrC\n");
    print(tCrC.shape()); print("\n"); print(tCrC.stride());
    print("\n\n");
  }
#endif

#if 1

  // TUTORIAL: Example of a very simple compute loop
  //   Data is read from global to shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared memory directly via the tC partitioning

  auto k_max = size<2>(tAgA);

  for (int k = 0; k < k_max; ++k)
  {
    // Copy gmem to smem
    copy(tAgA(_,_,k), tAsA);
    copy(tBgB(_,_,k), tBsB);

    // In case copy uses cp.async, make sure that the cp.async
    // instructions are ordered with respect to other cp.async
    // instructions (fence), then wait on all the outstanding copy
    // operations (wait<0>()).  __syncthreads() alone does not do
    // this.
    //
    // NOTE: cp_async_wait<0>() currently issues cp.async.wait_all.
    // This is equivalent to cp.async.commit_group followed by
    // cp.async_wait_group 0.  This should make the first
    // cp_async_fence() (which also issues cp.async.commit_group)
    // redundant.  The tutorial works as-is, so we'll leave the
    // redundant fence in for now and study its removal later.
    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();

    // Compute gemm on smem
    gemm(tCsA, tCsB, tCrC);

    __syncthreads();
  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}


template <typename TA, typename TB, typename TC,
          typename Alpha, typename Beta>
void
gemm(int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  // Define strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  // Define block sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};

  // Define the block layouts (static)
  auto sA = make_layout(make_shape(bM,bK));
  auto sB = make_layout(make_shape(bN,bK));
  auto sC = make_layout(make_shape(bM,bN));

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC));
  dim3 dimGrid(ceil_div(size(M), size(bM)),
               ceil_div(size(N), size(bN)));
  gemm_device
      <<< dimGrid, dimBlock, 0, stream >>>
      (M,  N,  K,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

#include <cstdlib>
#include <cstdio>
#include <cassert>

void test_gemm(int m, int n, int k)
{
  cute::device_init(0);

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  TI alpha = 1.0;
  TI beta  = 0.0;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  //
  // cuBLas
  //

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Run once
  d_C = h_C;
  blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                     m, n, k,
                     &alpha,
                     d_A.data().get(), m,
                     d_B.data().get(), n,
                     &beta,
                     d_C.data().get(), m);
  CUTE_CHECK_LAST();

  thrust::host_vector<TC> cublas_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    blam::cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       m, n, k,
                       &alpha,
                       d_A.data().get(), m,
                       d_B.data().get(), n,
                       &beta,
                       d_C.data().get(), m);
  }
  double cublas_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUBLAS_GEMM:   [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cublas_time, cublas_time*1000);

#else

  std::cout << "Verification by comparison with cuBLAS is disabled, "
    "either because the CMake option CUTLASS_ENABLE_CUBLAS "
    "was explicitly set to OFF, or because CMake could not find cuBLAS.  "
    "If you would like to enable verification with cuBLAS, "
    "please set the CMake option CUTLASS_ENABLE_CUBLAS to ON, "
    "rerun CMake, and recompile this example.\n";

#endif // CUTLASS_ENABLE_CUBLAS

  //
  // CuTe
  //

  // Run once (and check)
  d_C = h_C;
  gemm(m, n, k,
       alpha,
       d_A.data().get(), m,
       d_B.data().get(), n,
       beta,
       d_C.data().get(), m);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(m, n, k,
         alpha,
         d_A.data().get(), m,
         d_B.data().get(), n,
         beta,
         d_C.data().get(), m);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

#if defined(CUTLASS_ENABLE_CUBLAS) && CUTLASS_ENABLE_CUBLAS != 0
  printf("Empirical Perf: %.1f%%\n", (cublas_time / cute_time) * 100);

  auto host_matrix_to_const_column_major_cute_tensor =
    [](const auto& X, int num_rows, int num_cols, int LDX) {
      const auto shape = cute::Shape<int, int>{num_rows, num_cols};
      const auto strides = cute::Stride<int, int>{1, LDX};
      return cute::make_tensor(X.data(), cute::make_layout(shape, strides));
    };

  const auto A_view = host_matrix_to_const_column_major_cute_tensor(h_A, m, k, m);
  // B^T is k x n, so B is n x k.
  const auto B_view = host_matrix_to_const_column_major_cute_tensor(h_B, n, k, n);
  const auto C_computed_view = host_matrix_to_const_column_major_cute_tensor(cute_result, m, n, m);
  const auto C_expected_view = host_matrix_to_const_column_major_cute_tensor(cublas_result, m, n, m);
  print_matrix_multiply_mollified_relative_error("float", A_view, B_view, C_computed_view, C_expected_view);

#endif // CUTLASS_ENABLE_CUBLAS
}



// TODO: use collective builder
/*
// template<
// class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
// class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
// class StageCountType = cutlass::gemm::collective::StageCountAuto,
// class TileSchedulerType = cutlass::gemm::PersistentScheduler
// >
// struct GemmRunner {
//   using LayoutA = cutlass::layout::RowMajor;
//   using LayoutB = cutlass::layout::ColumnMajor;
//   using LayoutC = cutlass::layout::ColumnMajor;
//   using LayoutD = cutlass::layout::ColumnMajor;

//   using ElementA = cutlass::half_t;
//   using ElementB = cutlass::half_t;
//   using ElementC = cutlass::half_t;
//   using ElementD = cutlass::half_t;
//   using ElementAccumulator = float;
//   using ElementCompute = float;
//   using ElementScalar = float;

//   // 16B align ony for TMA?
//   static constexpr int AlignmentA = 16 / sizeof(ElementA);
//   static constexpr int AlignmentB = 16 / sizeof(ElementB);
//   static constexpr int AlignmentC = 16 / sizeof(ElementC);
//   static constexpr int AlignmentD = 16 / sizeof(ElementD);

//   // Epilogue collective has sm80?
//   using CollectiveEpilogue = typename cutlass::epilogue::collective::DefaultEpilogue

// }
*/


// A matrix
using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
// B matrix
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
// C matrix
using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int NumStages = 3;

// Epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
  ElementC,
  AlignmentC,
  ElementAccumulator,
  ElementAccumulator>;

using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;
// Matching profiler generated GEMM
// cutlass::arch::OpMultiplyAdd?
// typename cutlass::gemm::kernel::DefaultGemmUniversal
using DeviceGemmTest = cutlass::gemm::device::GemmUniversal<
ElementA, LayoutA,
ElementB, LayoutB,
ElementC, LayoutC,
ElementAccumulator,
OperatorClass,
ArchTag,
ThreadblockShape,
WarpShape,
InstructionShape,
EpilogueOp,
cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
NumStages,
AlignmentA,
AlignmentB,
cutlass::arch::OpMultiplyAdd>;

struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(true)
  {}

};


/// Command line options parsing
struct Options
{
  std::string               command_name;
  bool                      help;
  cutlass::gemm::GemmCoord  problem_size;
  float                     alpha;
  float                     beta;
  int                       split_k_factor;
  int                       avail_sms;
  bool                      reference_check;
  int                       iterations;

  cutlass::HostTensor<ElementA, LayoutA> tensor_a;
  cutlass::HostTensor<ElementB, LayoutB> tensor_b;
  cutlass::HostTensor<ElementC, LayoutC> tensor_c;
  cutlass::HostTensor<ElementC, LayoutC> tensor_d;
  cutlass::HostTensor<ElementC, LayoutC> tensor_ref_d;

  Options(std::string command_name) :
    command_name(command_name),
    help(false),
    problem_size({512, 6144, 12288}),
    alpha(1.0f),
    beta(0.0f),
    split_k_factor(1),
    avail_sms(-1),              // Number of device SMs to use is unlimited
    reference_check(true),
    iterations(10000)
  {}

  bool valid() const
  {
    return true;
  }

  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("split", split_k_factor);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const
  {
    out
      << "Performs a GEMM computation.\n"
      << "\n"
      << "Options:\n"
      << "\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --split=<int>               Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << command_name << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    return 2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s;
  }
};


typename DeviceGemmTest::Arguments args_from_options(
  const DeviceGemmTest &device_gemm,
  const Options & options,
  cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
  cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
  cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
  cutlass::HostTensor<ElementC, LayoutC> &tensor_d) {
  return typename DeviceGemmTest::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0));             // stride_d
}

template <typename DeviceGemmT>
Result run(std::string description, Options &options){
  // Display test description
  std::cout << std::endl << description << std::endl;
  // Zero-initialize test output matrix D
  cutlass::reference::host::TensorFill(options.tensor_d.host_view());
  options.tensor_d.sync_device();
  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmT device_gemm;
  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  // Check the problem size is supported or not
  CUTLASS_CHECK(device_gemm.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
  // Correctness / Warmup iteration
  CUTLASS_CHECK(device_gemm());
  // Copy output data from CUTLASS and reference kernel to host for comparison
  options.tensor_d.sync_host();
  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = cutlass::reference::host::TensorEquals(
    options.tensor_d.host_view(),
    options.tensor_ref_d.host_view());

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(device_gemm());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPs: " << result.gflops << std::endl;
  }

  if (!result.passed) {
    exit(-1);
  }

  return result;
}




int main(int argc, char const **argv)
{
  // int m = 5120;
  // if (argc >= 2)
  //   sscanf(argv[1], "%d", &m);

  // int n = 5120;
  // if (argc >= 3)
  //   sscanf(argv[2], "%d", &n);

  // int k = 4096;
  // if (argc >= 4)
  //   sscanf(argv[3], "%d", &k);

  // test_gemm(m, n, k);

  // Current device must must have compute capability at least 80
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (!((props.major * 10 + props.minor) >= 80))
  {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;

    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#if 1
  std::cout << "gpu major:" << props.major << std::endl;
#endif

  // Parse options
  Options options("ampere_gemm");
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  std::cout <<
    options.iterations << " timing iterations of " <<
    options.problem_size.m() << " x " <<
    options.problem_size.n() << " x " <<
    options.problem_size.k() << " matrix-matrix multiply" << std::endl;

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  #if 1
  std::cout << hw_info.sm_count << std::endl;
  #endif

  // Build up gemm and run

  // Initialize tensors using CUTLASS helper functions
  options.tensor_a.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
  options.tensor_b.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
  options.tensor_c.resize(options.problem_size.mn());       // <- Create matrix C with dimensions M x N
  options.tensor_d.resize(options.problem_size.mn());       // <- Create matrix D with dimensions M x N used to store output from CUTLASS kernel
  options.tensor_ref_d.resize(options.problem_size.mn());   // <- Create matrix D with dimensions M x N used to store output from reference kernel

  // Fill matrix A on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(
      options.tensor_a.host_view(),
      1,
      ElementA(2),
      ElementA(-2),
      0);

  // Fill matrix B on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(
      options.tensor_b.host_view(),
      1,
      ElementB(2),
      ElementB(-2),
      0);

  // Fill matrix C on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(
      options.tensor_c.host_view(),
      1,
      ElementC(2),
      ElementC(-2),
      0);

  // Copy data from host to GPU
  options.tensor_a.sync_device();
  options.tensor_b.sync_device();
  options.tensor_c.sync_device();
  // Zero-initialize reference output matrix D
  cutlass::reference::host::TensorFill(options.tensor_ref_d.host_view());
  options.tensor_ref_d.sync_device();

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;
  // Launch device reference gemm kernel
  gemm_reference(
    options.problem_size,
    ElementAccumulator(options.alpha),
    options.tensor_a.device_ref(),
    options.tensor_b.device_ref(),
    ElementAccumulator(options.beta),
    options.tensor_c.device_ref(),
    options.tensor_ref_d.device_ref());
  // Wait for kernels to finish
  CUDA_CHECK(cudaDeviceSynchronize());
  // Copy output data from reference kernel to host for comparison
  options.tensor_ref_d.sync_host();

  Result gemm_test_res = run<DeviceGemmTest>("Device gemm from profiler", options);
  printf("Done\n");

  return 0;
}
