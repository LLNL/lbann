////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
//
// batch_normalization_cuda.hpp - CUDA functions for batch normalization layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_CUDA_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_CUDA_HPP_INCLUDED
#ifdef __LIB_CUDNN

namespace lbann {

namespace batch_normalization_cuda {
/** Compute sums and squares of sums over channels on GPUs. */
template <typename T>
void channel_sums_and_sqsums(int height,
                             int width,
                             int num_channels,
                             const T *data_d,
                                   T *sums_d,
                                   T *sqsums_d,
                             cudaStream_t stream);
/** Compute statistics and running statistics on GPUs.
 *  The sums and sums of squares are assumed to be stored in means_d
 *  and vars_d, respectively.
 */
template <typename T>
void sums_to_statistics(int num_entries,
                        int samples_per_sum,
                        T decay,
                        T *mean_d,
                        T *var_d,
                        T *running_mean_d,
                        T *running_var_d,
                        cudaStream_t stream);
/** Apply batch normalization on GPUs. */
template <typename T>
void batch_normalization(int height,
                         int width,
                         int num_channels,
                         const T *prev_activations_d,
                         const T *mean_d,
                         const T *var_d,
                         T epsilon,
                         const T *scale_d,
                         const T *bias_d,
                               T *activations_d,
                         cudaStream_t stream);
/** Perform first phase of batch normalization backprop on GPUs.
 *  Compute gradient w.r.t. scaling factor, bias term, mean, and
 *  variance.
 */
template <typename T>
void batch_normalization_backprop1(int height,
                                   int width,
                                   int num_channels,
                                   const T *prev_activations_d,
                                   const T *prev_error_signal_d,
                                   const T *mean_d,
                                   const T *var_d,
                                   T epsilon,
                                   const T *scale_d,
                                         T *dscale_d,
                                         T *dbias_d,
                                         T *dmean_d,
                                         T *dvar_d,
                                   cudaStream_t stream);
/** Perform second phase of batch normalization backprop on GPUs.
 *  Compute error signal (i.e. gradient w.r.t. inputs).
 */
template <typename T>
void batch_normalization_backprop2(int height,
                                   int local_width,
                                   int global_width,
                                   int num_channels,
                                   const T *prev_activations_d,
                                   const T *prev_error_signal_d,
                                   const T *mean_d,
                                   const T *var_d,
                                   T epsilon,
                                   const T *scale_d,
                                   const T *dmean_d,
                                   const T *dvar_d,
                                         T *error_signal_d,
                                   cudaStream_t stream);
} // namespace batch_normalization_cuda

} // namespace lbann

#endif // __LIB_CUDNN
#endif  // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_CUDA_HPP_INCLUDED
