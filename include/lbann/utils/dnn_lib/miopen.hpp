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

#ifndef LBANN_UTILS_DNN_LIB_MIOPEN_HPP
#define LBANN_UTILS_DNN_LIB_MIOPEN_HPP

#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_MIOPEN

#include <hipDNN.h>

// Error utility macros
#define CHECK_MIOPEN_NODEBUG(cudnn_call)                          \
  do {                                                            \
    const hipdnnStatus_t status_CHECK_MIOPEN = (miopen_call);     \
    if (status_CHECK_MIOPEN != HIPDNN_STATUS_SUCCESS) {           \
      hipDeviceReset();                                           \
      LBANN_ERROR(std::string("MIOpen error (")                   \
                  + hipdnnGetErrorString(status_CHECK_MIOPEN)     \
                  + std::string(")"));                            \
    }                                                             \
  } while (0)
#define CHECK_MIOPEN_DEBUG(miopen_call)                           \
  do {                                                            \
    LBANN_ROCM_CHECK_LAST_ERROR(true);                            \
    CHECK_MIOPEN_NODEBUG(cudnn_call);                             \
  } while (0)
#ifdef LBANN_DEBUG
#define CHECK_MIOPEN(miopen_call) CHECK_MIOPEN_DEBUG(miopen_call)
#else
#define CHECK_MIOPEN(miopen_call) CHECK_MIOPEN_NODEBUG(miopen_call)
#endif // #ifdef LBANN_DEBUG

#define CHECK_MIOPEN_DTOR(miopen_call)                                  \
  try {                                                                 \
    CHECK_MIOPEN(miopen_call);                                          \
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

namespace miopen {

using dnnHandle_t = hipdnnHandle_t;
using dnnDataType_t = hipdnnDataType_t;
using dnnTensorDescriptor_t = hipdnnTensorDescriptor_t;
using dnnFilterDescriptor_t = hipdnnFilterDescriptor_t;
using dnnTensorFormat_t = hipdnnTensorFormat_t;
using dnnDropoutDescriptor_t = hipdnnDropoutDescriptor_t;
using dnnRNNDescriptor_t = hipdnnRNNDescriptor_t;
using dnnRNNAlgo_t = hipdnnRNNAlgo_t;
using dnnRNNMode_t = hipdnnRNNMode_t;
using dnnRNNBiasMode_t = hipdnnRNNBiasMode_t;
using dnnDirectionMode_t = hipdnnDirectionMode_t;
using dnnRNNInputMode_t = hipdnnRNNInputMode_t;
using dnnMathType_t = hipdnnMathType_t;
//using dnnRNNDataDescriptor_t = cudnnRNNDataDescriptor_t;
//using dnnRNNDataLayout_t = cudnnRNNDataLayout_t;
using dnnConvolutionDescriptor_t = hipdnnConvolutionDescriptor_t;
using dnnConvolutionMode_t = hipdnnConvolutionMode_t;
using dnnActivationDescriptor_t = hipdnnActivationDescriptor_t;
using dnnActivationMode_t = hipdnnActivationMode_t;
using dnnNanPropagation_t = hipdnnNanPropagation_t;
using dnnPoolingDescriptor_t = hipdnnPoolingDescriptor_t;
using dnnPoolingMode_t = hipdnnPoolingMode_t;
using dnnLRNDescriptor_t = hipdnnLRNDescriptor_t;
using dnnConvolutionFwdAlgo_t = hipdnnConvolutionFwdAlgo_t;
using dnnConvolutionBwdDataAlgo_t = hipdnnConvolutionBwdDataAlgo_t;
using dnnConvolutionBwdFilterAlgo_t = hipdnnConvolutionBwdFilterAlgo_t;

constexpr dnnConvolutionMode_t DNN_CROSS_CORRELATION = HIPDNN_CROSS_CORRELATION;
constexpr dnnNanPropagation_t DNN_PROPAGATE_NAN = HIPDNN_PROPAGATE_NAN;

////////////////////////////////////////////////////////////
// Functions for to/from cuDNN types conversion
////////////////////////////////////////////////////////////

/** @brief Convert a LBANN forward convolution algorithm to the cuDNN
 * equivalent value. */
inline hipdnnConvolutionFwdAlgo_t to_cudnn(fwd_conv_alg a)
{
  switch (a)
  {
  case fwd_conv_alg::IMPLICIT_GEMM: return HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  case fwd_conv_alg::IMPLICIT_PRECOMP_GEMM: return HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  case fwd_conv_alg::GEMM: return HIPDNN_CONVOLUTION_FWD_ALGO_GEMM;
  case fwd_conv_alg::DIRECT: return HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;
  case fwd_conv_alg::FFT: return HIPDNN_CONVOLUTION_FWD_ALGO_FFT;
  case fwd_conv_alg::FFT_TILING: return HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
  case fwd_conv_alg::WINOGRAD: return HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  case fwd_conv_alg::WINOGRAD_NONFUSED: return HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a cuDNN forward convolution algorithm to the LBANN
 * equivalent value. */
inline fwd_conv_alg from_cudnn(hipdnnConvolutionFwdAlgo_t a)
{
  switch (a)
  {
  case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return fwd_conv_alg::IMPLICIT_GEMM;
  case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return fwd_conv_alg::IMPLICIT_PRECOMP_GEMM;
  case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM: return fwd_conv_alg::GEMM;
  case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT: return fwd_conv_alg::DIRECT;
  case HIPDNN_CONVOLUTION_FWD_ALGO_FFT: return fwd_conv_alg::FFT;
  case HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return fwd_conv_alg::FFT_TILING;
  case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return fwd_conv_alg::WINOGRAD;
  case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return fwd_conv_alg::WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution algorithm to the cuDNN
 * equivalent value. */
inline hipdnnConvolutionBwdDataAlgo_t to_cudnn(bwd_data_conv_alg a)
{
  switch (a)
  {
  case bwd_data_conv_alg::CUDNN_ALGO_0: return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  case bwd_data_conv_alg::CUDNN_ALGO_1: return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  case bwd_data_conv_alg::FFT: return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
  case bwd_data_conv_alg::FFT_TILING: return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
  case bwd_data_conv_alg::WINOGRAD: return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
  case bwd_data_conv_alg::WINOGRAD_NONFUSED: return HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a cuDNN backward convolution algorithm to the LBANN
 * equivalent value. */
inline bwd_data_conv_alg from_cudnn(hipdnnConvolutionBwdDataAlgo_t a)
{
  switch (a)
  {
  case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0: return bwd_data_conv_alg::CUDNN_ALGO_0;
  case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1: return bwd_data_conv_alg::CUDNN_ALGO_1;
  case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: return bwd_data_conv_alg::FFT;
  case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: return bwd_data_conv_alg::FFT_TILING;
  case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: return bwd_data_conv_alg::WINOGRAD;
  case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return bwd_data_conv_alg::WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution filter algorithm to the cuDNN
 * equivalent value. */
inline hipdnnConvolutionBwdFilterAlgo_t to_cudnn(bwd_filter_conv_alg a)
{
  switch (a)
  {
  case bwd_filter_conv_alg::CUDNN_ALGO_0: return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  case bwd_filter_conv_alg::CUDNN_ALGO_1: return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  case bwd_filter_conv_alg::FFT: return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
  case bwd_filter_conv_alg::CUDNN_ALGO_3: return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
  case bwd_filter_conv_alg::WINOGRAD_NONFUSED: return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
  case bwd_filter_conv_alg::FFT_TILING: return HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

/** @brief Convert a cuDNN backward convolution filter algorithm to the LBANN
 * equivalent value. */
inline bwd_filter_conv_alg from_cudnn(hipdnnConvolutionBwdFilterAlgo_t a)
{
  switch (a)
  {
  case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0: return bwd_filter_conv_alg::CUDNN_ALGO_0;
  case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1: return bwd_filter_conv_alg::CUDNN_ALGO_1;
  case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: return bwd_filter_conv_alg::FFT;
  case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3: return bwd_filter_conv_alg::CUDNN_ALGO_3;
  case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: return bwd_filter_conv_alg::WINOGRAD_NONFUSED;
  case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING: return bwd_filter_conv_alg::FFT_TILING;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

} // namespace miopen
} // namespace lbann

#endif // LBANN_HAS_MIOPEN
#endif // LBANN_UTILS_DNN_LIB_MIOPEN_HPP
