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
//
// kfac_block_bn .hpp .cpp - A BN building block for the K-FAC method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_BN_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_BN_HPP_INCLUDED

#include "lbann/callbacks/kfac/kfac_block.hpp"

namespace lbann {
namespace callback {

/** A BN building block for K-FAC.
 */
class kfac_block_bn: public kfac_block {
 public:

  /** Constructor.
   */
  kfac_block_bn(Layer *layer,
                kfac *callback,
                const size_t layer_id,
                const size_t inverse_proc_rank,
                const bool is_after_conv,
                const size_t num_channels, const size_t spatial_prod)
      : kfac_block(layer, callback, layer_id, inverse_proc_rank),
        m_is_after_conv(is_after_conv),
        m_num_channels(num_channels), m_spatial_prod(spatial_prod) {
  }
  kfac_block_bn(const kfac_block_bn&) = default;
  kfac_block_bn& operator=(const kfac_block_bn&) = default;

#ifdef LBANN_HAS_GPU

  void update_kronecker_factors(
      lbann_comm* comm,
      const DataType kronecker_decay,
      const bool print_matrix,
      const bool print_matrix_summary) override;

  void update_kronecker_inverse(
      lbann_comm* comm,
      const bool use_pi,
      const DataType damping_act, const DataType damping_err,
      const bool print_matrix,
      const bool print_matrix_summary,
      const bool print_time) override;

  void update_preconditioned_grads(
      lbann_comm* comm) override;

#endif // LBANN_HAS_GPU

  std::string get_info() const override {
    std::ostringstream oss;
    oss << kfac_block::get_info()
        << ", is_after_conv=" << m_is_after_conv;
    return oss.str();
  }

 private:

#ifdef LBANN_HAS_GPU

  /** @brief Compute the factor of a batch-normalization layer.
   *  TODO: Remove as compute_bn_factor_data2col is used as default. **/
  template <typename TensorDataType>
  static void compute_bn_factor(
      const TensorDataType * __restrict__ activations,
      const TensorDataType * __restrict__ errors,
      const TensorDataType * __restrict__ scales,
      const TensorDataType * __restrict__ biases,
      TensorDataType * __restrict__ factor,
      const size_t batch_size,
      const size_t num_channels,
      const size_t spatial_prod,
      const cudaStream_t& stream);

  /** @brief The memory copy part of compute_bn_factor. Combined with
   *  GEMM. **/
  template <typename TensorDataType>
  static void compute_bn_factor_data2col(
      const TensorDataType * __restrict__ activations,
      const TensorDataType * __restrict__ errors,
      const TensorDataType * __restrict__ scales,
      const TensorDataType * __restrict__ biases,
      TensorDataType * __restrict__ cols,
      const size_t batch_size,
      const size_t num_channels,
      const size_t spatial_prod,
      const cudaStream_t& stream);

#endif // LBANN_HAS_GPU

  /** @brief Information to perform its computation. **/
  const bool m_is_after_conv;
  const size_t m_num_channels, m_spatial_prod;

#ifdef LBANN_HAS_GPU

  /** @brief Exponential moving average of the Fisher matrix. */
  El::Matrix<DataType, El::Device::GPU>
  m_fisher_average;

  /** @brief Inverse of the average Fisher matrix. */
  El::Matrix<DataType, El::Device::GPU>
  m_fisher_inverse;

#endif // LBANN_HAS_GPU

};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_BN_HPP_INCLUDED
