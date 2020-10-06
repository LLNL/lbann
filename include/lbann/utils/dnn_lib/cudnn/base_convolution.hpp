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
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_BASECONVOLUTION_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_BASECONVOLUTION_HPP_

#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

/** @brief Which convolution mode to use. */
enum class convolution_mode
{
  CONVOLUTION,
  CROSS_CORRELATION,
};// enum class softmax_mode

enum class tensor_format
{
  NCHW,
  NHWC,
  NCHW_VECT_C,
};

enum class fwd_conv_alg
{
  IMPLICIT_GEMM,
  IMPLICIT_PRECOMP_GEMM,
  GEMM,
  DIRECT,
  FFT,
  FFT_TILING,
  WINOGRAD,
  WINOGRAD_NONFUSED,
};

enum class bwd_conv_alg
{
  ALGO_0, // need a better name
  ALGO_1, // need a better name
  FFT,
  FFT_TILING,
  WINOGRAD,
  WINOGRAD_NONFUSED,
};

namespace cudnn
{

/** @brief Convert an LBANN convolution_mode to the cuDNN equivalent value. */
inline cudnnConvolutionMode_t to_cudnn(convolution_mode m)
{
  switch (m)
  {
  case convolution_mode::CONVOLUTION: return CUDNN_CONVOLUTION;
  case convolution_mode::CROSS_CORRELATION: return CUDNN_CROSS_CORRELATION;
  default:
    LBANN_ERROR("Invalid convolution mode requested.");
  }
}

inline cudnnTensorFormat_t to_cudnn(tensor_format f)
{
  switch (f)
  {
  case tensor_format::NCHW: return CUDNN_TENSOR_NCHW;
  case tensor_format::NHWC: return CUDNN_TENSOR_NHWC;
  case tensor_format::NCHW_VECT_C: return CUDNN_TENSOR_NCHW_VECT_C;
  default:
    LBANN_ERROR("Invalid tensor format requested.");
  }
}

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

inline cudnnConvolutionBwdDataAlgo_t to_cudnn(bwd_conv_alg a)
{
  switch (a)
  {
  case bwd_conv_alg::ALGO_0: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  case bwd_conv_alg::ALGO_1: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  case bwd_conv_alg::FFT: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
  case bwd_conv_alg::FFT_TILING: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
  case bwd_conv_alg::WINOGRAD: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
  case bwd_conv_alg::WINOGRAD_NONFUSED: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
  default:
    LBANN_ERROR("Invalid backward convolution algorithm requested.");
  }
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_forward(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  ConvolutionDescriptor const& convDesc,
  fwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& yDesc,
  El::AbstractMatrix<TensorDataType>& y)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(x),
                                     gpu::get_sync_info(w),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(y));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionForward(handle_manager.get(),
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      wDesc,
                                      w.LockedBuffer(),
                                      convDesc,
                                      to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      yDesc,
                                      y.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward(
  ScalarParameterType const& alpha_in,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  cudnnConvolutionBwdDataAlgo_t alg,
  //bwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dxDesc,
  El::AbstractMatrix<TensorDataType>& dx)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(w),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dx));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionBackwardData(handle_manager.get(),
                                      &alpha,
                                      wDesc,
                                      w.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      alg,
                                      //to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height() * sizeof(TensorDataType),
                                      &beta,
                                      dxDesc,
                                      dx.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_bias(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dbDesc,
  El::AbstractDistMatrix<TensorDataType>& db)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dy),
                                     gpu::get_sync_info(db));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionBackwardBias(handle_manager.get(),
                                           &alpha,
                                           dyDesc,
                                           dy.LockedBuffer(),
                                           &beta,
                                           dbDesc,
                                           db.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_filter(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  cudnnConvolutionBwdFilterAlgo_t alg,
  //bwd_conv_filter alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  FilterDescriptor const& dwDesc,
  El::AbstractDistMatrix<TensorDataType>& dw)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dw),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(x));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnConvolutionBackwardFilter(handle_manager.get(),
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      alg,
                                      //to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height() * sizeof(TensorDataType),
                                      &beta,
                                      dwDesc,
                                      dw.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void add_tensor(ScalarParameterType const& alpha_in,
                TensorDescriptor const& aDesc,
                El::AbstractDistMatrix<TensorDataType> const& A,
                ScalarParameterType const& beta_in,
                TensorDescriptor const& cDesc,
                El::AbstractMatrix<TensorDataType>& C)
{
  using LibScalingParamT = cudnn::ScalingParamType<TensorDataType>;
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(A),
                                     gpu::get_sync_info(C));
  auto handle_manager = internal::make_default_handle_manager(multisync);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(cudnnAddTensor(handle_manager.get(),
                             &alpha,
                             aDesc,
                             A.LockedBuffer(),
                             &beta,
                             cDesc,
                             C.Buffer()));
}

}// namespace cudnn
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_BASECONVOLUTION_HPP_
