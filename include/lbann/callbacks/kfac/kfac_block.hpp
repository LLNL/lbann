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
             const size_t layer_id,
             const size_t inverse_proc_rank)
      : m_layer(layer),
        m_callback(callback),
        m_layer_id(layer_id),
        m_inverse_proc_rank(inverse_proc_rank) {
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

  /** @brief Return whether this block already has an inverse history. */
  bool has_kronecker_inverse() const {
    return m_has_kronecker_inverse;
  }

  virtual std::string get_info() const {
    std::ostringstream oss;
    oss << "name=" << m_layer->get_name()
        << ", id=" << m_layer_id
        << ", type=" << m_layer->get_type()
        << ", inverse_proc_rank=" << m_inverse_proc_rank;
    return oss.str();
  }

  size_t get_inverse_proc_rank() const {
    return m_inverse_proc_rank;
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

  /** @brief The layer ID in the model.
      TODO: Remove this. */
  const size_t m_layer_id;

  const int m_inverse_proc_rank;

  /** @brief Whether this block already has an inverse history. */
  bool m_has_kronecker_inverse;

};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_HPP_INCLUDED
