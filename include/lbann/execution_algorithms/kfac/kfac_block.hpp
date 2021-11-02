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

#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_HPP_INCLUDED

#include "lbann/execution_algorithms/kfac/execution_context.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

// Forward declaration
namespace kfac {
class KFACExecutionContext;
}

/** A building block for K-FAC.
 */
template <El::Device Device>
class kfac_block {
 public:

  /** Constructor.
   */
  kfac_block(Layer* layer,
             kfac::KFACExecutionContext* context,
             size_t layer_id,
             size_t inverse_proc_rank)
      : m_layer(layer),
        m_layer_id(layer_id),
        m_inverse_proc_rank(inverse_proc_rank),
        m_context(context) {
    m_has_kronecker_inverse = false;
  }
  virtual ~kfac_block() = default;

  virtual void on_forward_prop_end(lbann_comm* comm) {}

  /** @brief Compute Kronecker factors. */
  virtual void compute_local_kronecker_factors(
      lbann_comm* comm,
      bool print_matrix,
      bool print_matrix_summary) {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Get buffers of Kronecker factors for reduce-scatter. */
  virtual const std::vector<El::AbstractMatrix<DataType>*>
  get_local_kronecker_buffers() {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Update the average Kronecker factors. */
  virtual void update_kronecker_average(
      lbann_comm* comm,
      DataType kronecker_decay,
      bool print_matrix,
      bool print_matrix_summary) {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Compute the inverse of the average Kronecker factors. */
  virtual void update_kronecker_inverse(
      lbann_comm* comm,
      bool use_pi,
      DataType damping_act, DataType damping_err,
      DataType learning_rate_factor,
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Compute the inverse of the average Kronecker factors. */
  virtual void compute_preconditioned_gradients(
      lbann_comm* comm,
      DataType learning_rate_factor,
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Copies activations, errors, and weights from model class to
  private variables to be used in KFAC computation. */
  virtual void initialize_activations_and_errors(
      lbann_comm* comm,
      int num_local_activations,
      int num_local_errors,
      int num_weights){
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Get buffers of preconditioned parameter gradients. */
  virtual const std::vector<El::AbstractMatrix<DataType>*>
  get_preconditioned_grad_buffers() {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

  /** @brief Copy inverse matrices to output buffer. */
  virtual int
  get_inverse_matrices(
      El::Matrix<DataType, Device>& output,
      int offset) = 0;

  /** @brief Get inverse matrices size (offset). */
  virtual int
  get_inverse_matrices_size(lbann_comm *comm) = 0;

  /** @brief Get inverse matrices size vector */
  virtual std::vector<int>
  get_inverse_matrices_size_vector(lbann_comm *comm) = 0;

  /** @brief Get inverse matrices size vector */
  virtual void
  resize_inverse_matrices_size(El::Matrix<double, El::Device::CPU>& inverse_matrices_size, int block_number) = 0;

  /** @brief Copy inverse matrices from output buffer. */
  virtual int
  set_inverse_matrices(
      El::Matrix<DataType, Device>& workspace,
      int offset,
      lbann_comm *comm) = 0;

  /** @brief Get block's information in one line. */
  virtual std::string get_info() const {
    std::ostringstream oss;
    oss << "name=" << m_layer->get_name()
        << ", id=" << m_layer_id
        << ", type=" << m_layer->get_type()
        << ", inverse_proc_rank=" << m_inverse_proc_rank;
    return oss.str();
  }

  std::string get_name() const {
    return m_layer->get_name();
  }

  size_t get_inverse_proc_rank() const {
    return m_inverse_proc_rank;
  }

  DataType* get_local_activation_buffer(int index){
    return m_parent_local_activations[index]->Buffer();
  }

  DataType* get_local_error_buffer(int index){
    return m_child_local_errors[index]->Buffer();
  }

  DataType* get_weight_buffer(int index){
    return m_weight_values[index]->Buffer();
  }

  DataType* get_gradient_wrt_weight_buffer(int index){
    return m_weight_gradients[index]->Buffer();
  }

  /** @brief Return the list of internal matrices' (name, height,
   * width) for debugging. All internal matrices should be ready when
   * this function is called. */
  virtual std::vector<std::tuple<std::string, size_t, size_t>>
  get_internal_matrix_info() const {
    LBANN_ERROR("this function should be called via a sub-class.");
  }

 protected:

  /** @brief Gets the Kronecker factor matrix of a FC layer.
   *  The same key is tied with the same matrix instance. */
  El::Matrix<DataType, Device>& get_workspace_matrix(
      const std::string& key, size_t height, size_t width);

  /** @brief Return the default sync info that may used in update functions. */
  El::SyncInfo<Device> get_sync_info();

  /** @brief The target layer. */
  Layer *m_layer;

  /** @brief The layer ID in the model.
      TODO: Remove this. */
  const size_t m_layer_id;

  /** @brief The process ID which perform inverse on Kronecker. */
  const int m_inverse_proc_rank;

  /** @brief Whether this block already has an inverse history. */
  bool m_has_kronecker_inverse;

  /** @brief distributed martices for activations and gradients. */
  std::vector<std::unique_ptr<AbsDistMat>> m_parent_local_activations,
    m_child_local_errors, m_weight_gradients;

  /** @brief Translatebetweengrid  funciton has a basic implementation for STAR,STAR
   * distributed matrices. Therefore, using local matrices for weights  */
  std::vector<std::unique_ptr<El::Matrix<DataType, Device>>> m_weight_values;


 private:

  /** @brief The execution context that created this block.
   *  TODO: Use its own workspace and remove this pointer. */
  kfac::KFACExecutionContext* m_context;

};

} // namespace lbann

#endif  // LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_HPP_INCLUDED
