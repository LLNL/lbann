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
// kfac_block .hpp .cpp - A building block for the K-FAC method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/kfac.hpp"
#include "lbann/callbacks/kfac/kfac_metadata.hpp"

namespace lbann {
namespace callback {

/** A building block for K-FAC.
 */
class kfac_block {
 public:

  /** Constructor.
   */
  kfac_block(Layer *layer,
             kfac *callback,
             const struct kfac_layer_metadata metadata)
      : m_layer(layer),
        m_callback(callback),
        m_metadata(metadata) {
    m_has_kronecker_inverse = false;
  }

  /** @brief Update the average Kronecker factors.
   * TODO: Split into sub-classes. */
  void update_kronecker_factors_fc_conv(
      lbann_comm* comm,
      const DataType kronecker_decay,
      const bool print_matrix,
      const bool print_matrix_summary);

  /** @brief Update the average Kronecker factors. */
  void update_kronecker_factors_bn(
      lbann_comm* comm,
      const DataType kronecker_decay,
      const bool print_matrix,
      const bool print_matrix_summary);

  /** @brief Compute the inverse of the average Kronecker factors. */
  void update_kronecker_inverse_fc_conv(
      lbann_comm* comm,
      const bool use_pi,
      const DataType damping_act, const DataType damping_err,
      const bool print_matrix,
      const bool print_matrix_summary,
      const bool print_time);

  /** @brief Compute the inverse of the average Kronecker factors. */
  void update_kronecker_inverse_bn(
      lbann_comm* comm,
      const bool use_pi,
      const DataType damping_act, const DataType damping_err,
      const bool print_matrix,
      const bool print_matrix_summary,
      const bool print_time);

  /** @brief Scatter preconditioned gradients. */
  void update_preconditioned_grads_fc_conv(
      lbann_comm* comm);

  /** @brief Scatter preconditioned gradients. */
  void update_preconditioned_grads_bn(
      lbann_comm* comm);

  // TODO: Remove this.
  const struct kfac_layer_metadata& get_metadata() const {
    return m_metadata;
  }

  /** @brief Return whether this block already has an inverse history. */
  bool has_kronecker_inverse() const {
    return m_has_kronecker_inverse;
  }

 private:

  /** @brief Return the default stream that may used in update functions. */
  cudaStream_t get_stream() {
    return hydrogen::cuda::GetDefaultStream();
  }

  /** @brief The target layer. */
  Layer *m_layer;

  /** @brief The parent callback.
   * TODO: Use its own workspace and remove this pointer. */
  kfac *m_callback;

  /** @brief Metadata of the layer.
      TODO: merge with this block class. */
  const struct kfac_layer_metadata m_metadata;

  /** @brief Whether this block already has an inverse history. */
  bool m_has_kronecker_inverse;

  /** @brief Exponential moving average of Kronecker factors. */
  El::Matrix<DataType, El::Device::GPU>
  m_kronecker_average_A, m_kronecker_average_G;

  /** @brief Inverse of the average Kronecker factors. */
  El::Matrix<DataType, El::Device::GPU>
  m_kronecker_inverse_A, m_kronecker_inverse_G;
};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_HPP_INCLUDED
