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

#include "lbann/callbacks/kfac.hpp"
#include "lbann/callbacks/kfac/kfac_metadata.hpp"

namespace lbann {
namespace callback {

// Forward declarations
// TODO: Remove if kfac_block no longer refers kfac
class kfac;

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
  virtual ~kfac_block() = default;

  /** @brief Update the average Kronecker factors. */
  virtual void update_kronecker_factors(
      lbann_comm* comm,
      const DataType kronecker_decay,
      const bool print_matrix,
      const bool print_matrix_summary) = 0;

  /** @brief Compute the inverse of the average Kronecker factors. */
  virtual void update_kronecker_inverse(
      lbann_comm* comm,
      const bool use_pi,
      const DataType damping_act, const DataType damping_err,
      const bool print_matrix,
      const bool print_matrix_summary,
      const bool print_time) = 0;

  /** @brief Scatter preconditioned gradients. */
  virtual void update_preconditioned_grads(
      lbann_comm* comm) = 0;

  // TODO: Remove this.
  const struct kfac_layer_metadata& get_metadata() const {
    return m_metadata;
  }

  /** @brief Return whether this block already has an inverse history. */
  bool has_kronecker_inverse() const {
    return m_has_kronecker_inverse;
  }

 protected:

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

};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_HPP_INCLUDED
