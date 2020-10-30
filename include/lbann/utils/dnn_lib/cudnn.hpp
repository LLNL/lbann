////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_UTILS_DNN_LIB_CUDNN_HPP
#define LBANN_UTILS_DNN_LIB_CUDNN_HPP

#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_CUDNN

#include <cudnn.h>

// Error utility macros
#define CHECK_CUDNN_NODEBUG(cudnn_call)                         \
  do {                                                          \
    const cudnnStatus_t status_CHECK_CUDNN = (cudnn_call);      \
    if (status_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {           \
      cudaDeviceReset();                                        \
      LBANN_ERROR(std::string("cuDNN error (")                  \
                  + cudnnGetErrorString(status_CHECK_CUDNN)     \
                  + std::string(")"));                          \
    }                                                           \
  } while (0)
#define CHECK_CUDNN_DEBUG(cudnn_call)                           \
  do {                                                          \
    LBANN_CUDA_CHECK_LAST_ERROR(true);                          \
    CHECK_CUDNN_NODEBUG(cudnn_call);                            \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_CUDNN(cudnn_call) CHECK_CUDNN_DEBUG(cudnn_call)
#else
#define CHECK_CUDNN(cudnn_call) CHECK_CUDNN_NODEBUG(cudnn_call)
#endif // #ifdef LBANN_DEBUG

#define CHECK_CUDNN_DTOR(cudnn_call)            \
  try {                                         \
    CHECK_CUDNN(cudnn_call);                                            \
  }                                                                     \
  catch (std::exception const& e) {                                     \
    std::cerr << "Caught exception:\n\n    what(): "                    \
              << e.what() << "\n\nCalling std::terminate() now."        \
              <<  std::endl;                                            \
    std::terminate();                                                   \
  }                                                                     \
  catch (...) {                                                         \
    std::cerr << "Caught something that isn't an std::exception.\n\n"   \
              << "Calling std::terminate() now." << std::endl;          \
    std::terminate();                                                   \
  }


namespace lbann {

// Forward declaration
class Layer;

namespace cudnn {

using dnnHandle_t = cudnnHandle_t;
using dnnDataType_t = cudnnDataType_t;
using dnnTensorDescriptor_t = cudnnTensorDescriptor_t;
using dnnFilterDescriptor_t = cudnnFilterDescriptor_t;
using dnnTensorFormat_t = cudnnTensorFormat_t;
using dnnDropoutDescriptor_t = cudnnDropoutDescriptor_t;
using dnnRNNDescriptor_t = cudnnRNNDescriptor_t;
using dnnRNNAlgo_t = cudnnRNNAlgo_t;
using dnnRNNMode_t = cudnnRNNMode_t;
using dnnRNNBiasMode_t = cudnnRNNBiasMode_t;
using dnnDirectionMode_t = cudnnDirectionMode_t;
using dnnRNNInputMode_t = cudnnRNNInputMode_t;
using dnnMathType_t = cudnnMathType_t;
using dnnRNNDataDescriptor_t = cudnnRNNDataDescriptor_t;
using dnnRNNDataLayout_t = cudnnRNNDataLayout_t;
using dnnConvolutionDescriptor_t = cudnnConvolutionDescriptor_t;
using dnnConvolutionMode_t = cudnnConvolutionMode_t;
using dnnActivationDescriptor_t = cudnnActivationDescriptor_t;
using dnnActivationMode_t = cudnnActivationMode_t;
using dnnNanPropagation_t = cudnnNanPropagation_t;
using dnnPoolingDescriptor_t = cudnnPoolingDescriptor_t;
using dnnPoolingMode_t = cudnnPoolingMode_t;
using dnnLRNDescriptor_t = cudnnLRNDescriptor_t;
using dnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t;
using dnnConvolutionBwdDataAlgo_t = cudnnConvolutionBwdDataAlgo_t;
using dnnConvolutionBwdFilterAlgo_t = cudnnConvolutionBwdFilterAlgo_t;

constexpr dnnConvolutionMode_t DNN_CROSS_CORRELATION = CUDNN_CROSS_CORRELATION;
constexpr dnnNanPropagation_t DNN_PROPAGATE_NAN = CUDNN_PROPAGATE_NAN;

////////////////////////////////////////////////////////////
// Functions for to/from cuDNN types conversion
////////////////////////////////////////////////////////////

/** @brief Convert a LBANN forward convolution algorithm to the cuDNN
 * equivalent value. */
inline cudnnConvolutionFwdAlgo_t to_cudnn(fwd_conv_alg a)
{
  switch (a)
  {
  case fwd_conv_alg::IMPLICIT_GEMM: return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  case fwd_conv_alg::IMPLICIT_PRECOMP_GEMM: return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  case fwd_conv_alg::GEMM: return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  case fwd_conv_alg::DIRECT: return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
  case fwd_conv_alg::FFT: return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  case fwd_conv_alg::FFT_TILING: return CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
  case fwd_conv_alg::WINOGRAD: return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  case fwd_conv_alg::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a cuDNN forward convolution algorithm to the LBANN
 * equivalent value. */
inline fwd_conv_alg from_cudnn(cudnnConvolutionFwdAlgo_t a)
{
  switch (a)
  {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return fwd_conv_alg::IMPLICIT_GEMM;
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return fwd_conv_alg::IMPLICIT_PRECOMP_GEMM;
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return fwd_conv_alg::GEMM;
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return fwd_conv_alg::DIRECT;
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return fwd_conv_alg::FFT;
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return fwd_conv_alg::FFT_TILING;
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return fwd_conv_alg::WINOGRAD;
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return fwd_conv_alg::WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution algorithm to the cuDNN
 * equivalent value. */
inline cudnnConvolutionBwdDataAlgo_t to_cudnn(bwd_data_conv_alg a)
{
  switch (a)
  {
  case bwd_data_conv_alg::CUDNN_ALGO_0: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  case bwd_data_conv_alg::CUDNN_ALGO_1: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  case bwd_data_conv_alg::FFT: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
  case bwd_data_conv_alg::FFT_TILING: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
  case bwd_data_conv_alg::WINOGRAD: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
  case bwd_data_conv_alg::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a cuDNN backward convolution algorithm to the LBANN
 * equivalent value. */
inline bwd_data_conv_alg from_cudnn(cudnnConvolutionBwdDataAlgo_t a)
{
  switch (a)
  {
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: return bwd_data_conv_alg::CUDNN_ALGO_0;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: return bwd_data_conv_alg::CUDNN_ALGO_1;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: return bwd_data_conv_alg::FFT;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: return bwd_data_conv_alg::FFT_TILING;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: return bwd_data_conv_alg::WINOGRAD;
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return bwd_data_conv_alg::WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution filter algorithm to the cuDNN
 * equivalent value. */
inline cudnnConvolutionBwdFilterAlgo_t to_cudnn(bwd_filter_conv_alg a)
{
  switch (a)
  {
  case bwd_filter_conv_alg::CUDNN_ALGO_0: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  case bwd_filter_conv_alg::CUDNN_ALGO_1: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  case bwd_filter_conv_alg::FFT: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
  case bwd_filter_conv_alg::CUDNN_ALGO_3: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
  case bwd_filter_conv_alg::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
  case bwd_filter_conv_alg::FFT_TILING: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

/** @brief Convert a cuDNN backward convolution filter algorithm to the LBANN
 * equivalent value. */
inline bwd_filter_conv_alg from_cudnn(cudnnConvolutionBwdFilterAlgo_t a)
{
  switch (a)
  {
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0: return bwd_filter_conv_alg::CUDNN_ALGO_0;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1: return bwd_filter_conv_alg::CUDNN_ALGO_1;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: return bwd_filter_conv_alg::FFT;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3: return bwd_filter_conv_alg::CUDNN_ALGO_3;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: return bwd_filter_conv_alg::WINOGRAD_NONFUSED;
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING: return bwd_filter_conv_alg::FFT_TILING;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

} // namespace cudnn
} // namespace lbann

#endif // LBANN_HAS_CUDNN
#endif // LBANN_UTILS_DNN_LIB_CUDNN_HPP
