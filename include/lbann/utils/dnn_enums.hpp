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
#ifndef LBANN_UTILS_DNN_ENUMS_HPP
#define LBANN_UTILS_DNN_ENUMS_HPP

namespace lbann
{

/** @brief Which forward convolution algorithm to use. */
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
};// enum class fwd_conv_alg

/** @brief Which backward convolution algorithm to use. */
enum class bwd_data_conv_alg
{
  CUDNN_ALGO_0,
  CUDNN_ALGO_1,
  FFT,
  FFT_TILING,
  WINOGRAD,
  WINOGRAD_NONFUSED,
  IMPLICIT_GEMM,
};// enum class bwd_conv_alg

/** @brief Which backward convolution filter algorithm to use. */
enum class bwd_filter_conv_alg
{
  CUDNN_ALGO_0,
  CUDNN_ALGO_1,
  FFT,
  CUDNN_ALGO_3,
  WINOGRAD,
  WINOGRAD_NONFUSED,
  FFT_TILING,
  IMPLICIT_GEMM,
};// enum class bwd_conv_filter

/** @brief Internal LBANN names for supported LRN layer modes.  */
// Only one implemented in cudnn currently:
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnLRNCrossChannelForward
enum class lrn_mode
{
  CROSS_CHANNEL_DIM1,
};// enum class lrn_mode

/** @brief Which pooling mode to use. */
enum class pooling_mode
{
  MAX,
  AVERAGE_COUNT_INCLUDE_PADDING,
  AVERAGE_COUNT_EXCLUDE_PADDING,
  MAX_DETERMINISTIC,
}; // enum class pooling_mode

/** @brief Which tensor dimensions to apply softmax over. */
enum class softmax_mode
{
  INVALID,
  /** @brief Sample-wise softmax.
   *
   *  Slice tensor along the sample dimension (assuming data in NCHW
   *  format) and apply softmax independently to each slice (once per
   *  sample).
   */
  INSTANCE,
  /** @brief Position-wise softmax.
   *
   *  Split tensor along all but the channel dimension (assuming data
   *  in NCHW format) and apply softmax independently to each piece
   *  (once per spatial position per sample).
   *
   *  This is not to be confused with @c channelwise_softmax, which
   *  slices along the sample and channel dimensions.
   */
  CHANNEL
};// enum class softmax_mode

/** @brief Internal LBANN names for supported softmax algorithms. */
enum class softmax_alg
{
  FAST,
  ACCURATE,
  LOG,
};// enum class softmax_alg

}// namespace lbann
#endif // LBANN_UTILS_DNN_ENUMS_HPP
