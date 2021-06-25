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
// kfac_block_gru .hpp .cpp - A GRU building block for the K-FAC method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_GRU_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_GRU_HPP_INCLUDED

#include "lbann/callbacks/kfac/kfac_block.hpp"
#include "lbann/layers/learning/gru.hpp"

namespace lbann {
namespace callback {

namespace kfac_gru_util {

enum class weight_type {
  Wr, Wi, Wh,
  Rr, Ri, Rh,
  bWr, bWi, bWh,
  bRr, bRi, bRh,
};

const std::vector<weight_type> LEARNABLE_MATRICES = {
  weight_type::Rr,
  weight_type::Ri,
  weight_type::Rh,
  weight_type::Wr,
  weight_type::Wi,
  weight_type::Wh
};

/** @brief Get the name of a GRU weight matrix. **/
std::string get_matrix_type_name(const weight_type& matrix_type);

/** @brief Return whether the height of a GRU weight matrix is the hidden size. **/
bool is_matrix_height_hidden(const weight_type& matrix_type);

/** @brief Get the weight ID and the row offset ID of a GRU weight matrix. **/
std::pair<int, int> get_gru_weight_offset(
    weight_type matrix_type);

/** @brief Copy r_t and i_t from the reserve space after the forward pass. **/
template <El::Device Device>
void unpack_reserve_space(
    const DataType* reserve_space_fwd,
    El::Matrix<DataType, Device>& r,
    El::Matrix<DataType, Device>& i,
    size_t hidden_size,
    size_t seq_length,
    size_t local_batch_size,
    const El::SyncInfo<Device>& sync_info);

/** @brief Compute internal GRU gate state (r or i). **/
template <El::Device Device>
void gru_gate_forward(
    const El::Matrix<DataType, Device>& W_y,
    const El::Matrix<DataType, Device>& R_y,
    const El::Matrix<DataType, Device>& b_Wy,
    const El::Matrix<DataType, Device>& b_Ry,
    const El::Matrix<DataType, Device>& x_t,
    const El::Matrix<DataType, Device>& hprev_t,
    const El::Matrix<DataType, Device>& biases_ones,
    El::Matrix<DataType, Device>& y_t);

/** @brief Compute d h_t / d g_t. **/
template <El::Device Device>
void get_g(
    const El::Matrix<DataType, Device>& h,
    const El::Matrix<DataType, Device>& h0,
    const El::Matrix<DataType, Device>& dh,
    const El::Matrix<DataType, Device>& hfc,
    const El::Matrix<DataType, Device>& r,
    const El::Matrix<DataType, Device>& i,
    El::Matrix<DataType, Device>& g_Rr,
    El::Matrix<DataType, Device>& g_Ri,
    El::Matrix<DataType, Device>& g_Rh,
    El::Matrix<DataType, Device>& g_Wr,
    El::Matrix<DataType, Device>& g_Wi,
    El::Matrix<DataType, Device>& g_Wh,
    size_t hidde_size,
    size_t seq_length,
    size_t local_batch_size,
    const El::SyncInfo<Device>& sync_info);

} // namespace kfac_bn_util

/** A BN building block for K-FAC.
 */
template <El::Device Device>
class kfac_block_gru: public kfac_block<Device> {
 public:

  /** Constructor.
   */
  kfac_block_gru(Layer *layer,
                 kfac<Device> *callback,
                 size_t layer_id,
                 size_t inverse_proc_rank)
      : kfac_block<Device>(layer, callback, layer_id, inverse_proc_rank) {

    check_dnn_lib_spec();

    const auto num_layers = get_gru_layer()->get_num_layers();
    if(num_layers > 1) {
      std::stringstream err;
      err << "The K-FAC callback only supports single-layer GRU layer."
          << " layer: " << layer->get_name()
          << ", num_layers: "
          << num_layers;
      LBANN_ERROR(err.str());
    }

  }
  kfac_block_gru(const kfac_block_gru&) = default;
  kfac_block_gru& operator=(const kfac_block_gru&) = default;

  void on_forward_prop_end() override;

  const std::vector<El::AbstractMatrix<DataType>*>
  get_local_kronecker_buffers();

  void compute_local_kronecker_factors(
      lbann_comm* comm,
      bool print_matrix,
      bool print_matrix_summary) override;

  void update_kronecker_average(
      lbann_comm* comm,
      DataType kronecker_decay,
      bool print_matrix,
      bool print_matrix_summary) override;

  void update_kronecker_inverse(
      lbann_comm* comm,
      bool use_pi,
      DataType damping_act, DataType damping_err,
      DataType learning_rate_factor,
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) override;

  const std::vector<El::AbstractMatrix<DataType>*>
  get_preconditioned_grad_buffers() override;

 private:

  void check_dnn_lib_spec() const;

  /** @brief Recompute or copy (from cuDNN's reserve space if
   * available) forward internal state (r and i).  **/
  void get_r_i(
      El::Matrix<DataType, Device>& r,
      El::Matrix<DataType, Device>& i,
      const El::Matrix<DataType, Device>& biases_ones,
      const El::Matrix<DataType, Device>& local_inputs,
      const El::Matrix<DataType, Device>& local_outputs,
      const El::Matrix<DataType, Device>& h0,
      size_t local_batch_size,
      const El::SyncInfo<Device>& sync_info);

  /** @brief Get the view of a weight matrix or a bias vector or its gradients. **/
  void get_weight_matrix(
      kfac_gru_util::weight_type matrix_type,
      El::Matrix<DataType, Device>& view);
  void get_gradient_matrix(
      kfac_gru_util::weight_type matrix_type,
      El::Matrix<DataType, Device>& view);
  void get_gradient_buffer(
      kfac_gru_util::weight_type matrix_type,
      El::Matrix<DataType, Device>& view);

  std::vector<std::tuple<std::string, size_t, size_t>>
  get_internal_matrix_info() const override;

  /** @brief Get the pointer to its GRU_layer. **/
  gru_layer<DataType, data_layout::DATA_PARALLEL, Device>*
  get_gru_layer() const {
    return dynamic_cast<gru_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(this->m_layer);
  }

  size_t get_input_size() const {
    const auto input_dims = this->m_layer->get_input_dims();
    return input_dims[1];
  }
  size_t get_hidden_size() const {
    return get_gru_layer()->get_hidden_size();
  }
  size_t get_seq_length() const {
    const auto input_dims = this->m_layer->get_input_dims();
    return input_dims[0];
  }

  /** @brief A copy of the reserve space after forward passes. */
  hydrogen::simple_buffer<El::byte, Device>
  m_reserve_space_fwd;

  /** @brief Lower triangle buffers of Kronecker factors. */
  El::Matrix<DataType, Device>
  m_kronecker_factor_buf_A_h, m_kronecker_factor_buf_A_x;
  std::unordered_map<
    kfac_gru_util::weight_type,
    El::Matrix<DataType, Device>>
  m_kronecker_factor_buf_G;

  /** @brief Exponential moving average of Kronecker factors. */
  El::Matrix<DataType, Device>
  m_kronecker_average_A_h, m_kronecker_average_A_x;
  std::unordered_map<
    kfac_gru_util::weight_type,
    El::Matrix<DataType, Device>>
  m_kronecker_average_G;

  /** @brief Inverse of the average Kronecker factors. */
  El::Matrix<DataType, Device>
  m_kronecker_inverse_A_h, m_kronecker_inverse_A_x;
  std::unordered_map<
    kfac_gru_util::weight_type,
    El::Matrix<DataType, Device>>
  m_kronecker_inverse_G;

  El::Matrix<DataType, Device>
  m_grad_buffer_A_h, m_grad_buffer_A_x;
  std::unordered_map<
    kfac_gru_util::weight_type,
    El::Matrix<DataType, Device>>
  m_grad_buffer_G;

};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_GRU_HPP_INCLUDED
