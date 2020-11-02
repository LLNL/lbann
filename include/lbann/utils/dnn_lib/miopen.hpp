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

#include <miopen/miopen.h>

// Error utility macros
#define CHECK_MIOPEN_NODEBUG(cudnn_call)                          \
  do {                                                            \
    const miopenStatus_t status_CHECK_MIOPEN = (miopen_call);     \
    if (status_CHECK_MIOPEN != HIPDNN_STATUS_SUCCESS) {           \
      hipDeviceReset();                                           \
      LBANN_ERROR(std::string("MIOpen error (")                   \
                  + miopenGetErrorString(status_CHECK_MIOPEN)     \
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

using dnnHandle_t = miopenHandle_t;
using dnnDataType_t = miopenDataType_t;
using dnnTensorDescriptor_t = miopenTensorDescriptor_t;
using dnnFilterDescriptor_t = miopenFilterDescriptor_t;
using dnnTensorFormat_t = miopenTensorFormat_t;
using dnnDropoutDescriptor_t = miopenDropoutDescriptor_t;
using dnnRNNDescriptor_t = miopenRNNDescriptor_t;
using dnnRNNAlgo_t = miopenRNNAlgo_t;
using dnnRNNMode_t = miopenRNNMode_t;
using dnnRNNBiasMode_t = miopenRNNBiasMode_t;
using dnnDirectionMode_t = miopenDirectionMode_t;
using dnnRNNInputMode_t = miopenRNNInputMode_t;
using dnnMathType_t = miopenMathType_t;
//using dnnRNNDataDescriptor_t = cudnnRNNDataDescriptor_t;
//using dnnRNNDataLayout_t = cudnnRNNDataLayout_t;
using dnnConvolutionDescriptor_t = miopenConvolutionDescriptor_t;
using dnnConvolutionMode_t = miopenConvolutionMode_t;
using dnnActivationDescriptor_t = miopenActivationDescriptor_t;
using dnnActivationMode_t = miopenActivationMode_t;
using dnnNanPropagation_t = miopenNanPropagation_t;
using dnnPoolingDescriptor_t = miopenPoolingDescriptor_t;
using dnnPoolingMode_t = miopenPoolingMode_t;
using dnnLRNDescriptor_t = miopenLRNDescriptor_t;
using dnnConvolutionFwdAlgo_t = miopenConvFwdAlgorithm_t;
using dnnConvolutionBwdDataAlgo_t = miopenConvBwdDataAlgorithm_t;
using dnnConvolutionBwdFilterAlgo_t = miopenConvBwdWeightsAlgorithm_t;

constexpr dnnConvolutionMode_t DNN_CROSS_CORRELATION = HIPDNN_CROSS_CORRELATION;
constexpr dnnNanPropagation_t DNN_PROPAGATE_NAN = HIPDNN_PROPAGATE_NAN;

////////////////////////////////////////////////////////////
// Functions for to/from MIOpen types conversion
////////////////////////////////////////////////////////////

/** @brief Convert a LBANN forward convolution algorithm to the MIOpen
 * equivalent value. */
inline miopenConvFwdAlgorithm_t to_miopen(fwd_conv_alg a)
{
  switch (a)
  {
  case fwd_conv_alg::IMPLICIT_GEMM: return miopenConvolutionFwdAlgoGEMM;
  case fwd_conv_alg::GEMM: return miopenConvolutionFwdAlgoGEMM;
  case fwd_conv_alg::DIRECT: return miopenConvolutionFwdAlgoDirect;
  case fwd_conv_alg::FFT: return miopenConvolutionFwdAlgoFFT;
  case fwd_conv_alg::WINOGRAD: return miopenConvolutionFwdAlgoWinograd;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a MIOpen forward convolution algorithm to the LBANN
 * equivalent value. */
inline fwd_conv_alg from_miopen(miopenConvFwdAlgorithm_t a)
{
  switch (a)
  {
  case miopenConvolutionFwdAlgoGEMM: return fwd_conv_alg::GEMM;
  case miopenConvolutionFwdAlgoDirect: return fwd_conv_alg::DIRECT;
  case miopenConvolutionFwdAlgoFFT: return fwd_conv_alg::FFT;
  case miopenConvolutionFwdAlgoWinograd: return fwd_conv_alg::WINOGRAD;
  default:
    LBANN_ERROR("Invalid forward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution algorithm to the MIOpen
 * equivalent value. */
inline miopenConvBwdDataAlgorithm_t to_miopen(bwd_data_conv_alg a)
{
  switch (a)
  {
  case bwd_data_conv_alg::CUDNN_ALGO_0: return miopenConvolutionBwdDataAlgoGEMM;
  case bwd_data_conv_alg::CUDNN_ALGO_1: return miopenConvolutionBwdDataAlgoDirect;
  case bwd_data_conv_alg::FFT: return miopenConvolutionBwdDataAlgoFFT;
  case bwd_data_conv_alg::WINOGRAD: return miopenConvolutionBwdDataAlgoWinograd;
  case bwd_data_conv_alg::WINOGRAD_NONFUSED: return miopenConvolutionBwdDataAlgoWinograd;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a MIOpen backward convolution algorithm to the LBANN
 * equivalent value. */
inline bwd_data_conv_alg from_miopen(miopenConvBwdDataAlgorithm_t a)
{
  switch (a)
  {
  case miopenConvolutionBwdDataAlgoGEMM: return bwd_data_conv_alg::CUDNN_ALGO_0;
  case miopenConvolutionBwdDataAlgoDirect: return bwd_data_conv_alg::CUDNN_ALGO_1;
  case miopenConvolutionBwdDataAlgoFFT: return bwd_data_conv_alg::FFT;
  case miopenConvolutionBwdDataAlgoWinograd: return bwd_data_conv_alg::WINOGRAD;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

/** @brief Convert a LBANN backward convolution filter algorithm to the MIOpen
 * equivalent value.
 * https://github.com/ROCmSoftwarePlatform/hipDNN/blob/44df79868ebc5b7bc7deb9b24b03da92d6bd30dc/library/src/hcc_detail/hipdnn_miopen.cpp#L517*/
inline miopenConvBwdWeightsAlgorithm_t to_miopen(bwd_filter_conv_alg a)
{
  switch (a)
  {
  case bwd_filter_conv_alg::CUDNN_ALGO_0: return miopenConvolutionBwdWeightsAlgoGEMM;
  case bwd_filter_conv_alg::CUDNN_ALGO_1: return miopenConvolutionBwdWeightsAlgoDirect;
  case bwd_filter_conv_alg::WINOGRAD: return miopenConvolutionBwdWeightsAlgoWinograd;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

/** @brief Convert a MIOpen backward convolution filter algorithm to the LBANN
 * equivalent value. */
inline bwd_filter_conv_alg from_miopen(miopenConvBwdWeightsAlgorithm_t a)
{
  switch (a)
  {
  case miopenConvolutionBwdWeightsAlgoGEMM: return bwd_filter_conv_alg::CUDNN_ALGO_0;
  case miopenConvolutionBwdWeightsAlgoDirect: return bwd_filter_conv_alg::CUDNN_ALGO_1;
  case miopenConvolutionBwdWeightsAlgoWinograd: return bwd_filter_conv_alg::WINOGRAD;
  default:
    LBANN_ERROR("Invalid backward convolution filter requested.");
  }
}

} // namespace miopen
} // namespace lbann

#endif // LBANN_HAS_MIOPEN
#endif // LBANN_UTILS_DNN_LIB_MIOPEN_HPP
