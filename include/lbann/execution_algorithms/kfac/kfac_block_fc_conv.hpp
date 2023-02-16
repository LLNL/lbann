////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_FC_CONV_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_FC_CONV_HPP_INCLUDED

#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/layers/learning/convolution.hpp"

namespace lbann {

namespace kfac_fc_conv_util {

/** @brief Get diagonal elements of a matrix. **/
template <El::Device Device>
void get_diagonal(
    El::Matrix<DataType, Device>& diag,
    const El::Matrix<DataType, Device>& A,
    const El::SyncInfo<Device>& sync_info);

/** @brief Transpose NC(D)HW matrix to N(D)HWC. **/
template <El::Device Device>
void conv_transpose(
    const El::Matrix<DataType, Device>& activations,
    El::Matrix<DataType, Device>& act_columns,
    size_t mini_batch_size, size_t num_channels,
    size_t spatial_prod,
    const El::SyncInfo<Device>& sync_info);

/** @brief im2col. **/
template <El::Device Device>
void im2col(const El::Matrix<DataType, Device>& im,
            El::Matrix<DataType, Device>& col,
            const int num_channels,
            const int im_num_dims,
            const int * im_dims,
            const int * im_pads,
            const int * window_dims,
            const int * window_strides,
            const int batch_size,
            const El::SyncInfo<Device>& sync_info);

} // namespace kfac_fc_conv_util

/** An FC/conv building block for K-FAC.
 * TODO: Split into kfac_block_fc and kfac_block_conv.
 */
template <El::Device Device>
class kfac_block_fc_conv: public kfac_block<Device> {
 public:

  /** Constructor.
   */
  kfac_block_fc_conv(Layer* layer,
                     kfac::KFACExecutionContext* context,
                     const size_t layer_id,
                     const size_t inverse_proc_rank,
                     const bool enable_copy_errors,
                     const bool enable_copy_activations,
                     const int input_size,
                     const int output_size,
                     const bool is_conv)
      : kfac_block<Device>(layer, context, layer_id, inverse_proc_rank, enable_copy_errors, enable_copy_activations, input_size, output_size),
        m_is_conv(is_conv), m_has_bias(layer->num_weights() > 1) {
    if(m_is_conv) {
      m_conv_input_spatial_prod = 1;
      const auto input_dims = layer->get_input_dims();
      for(auto i = input_dims.begin()+1; i != input_dims.end(); i++) {
        m_conv_input_spatial_prod *= *i;
        m_conv_input_spatial_dims.push_back(*i);
      }

      m_conv_output_spatial_prod = 1;
      const auto output_dims = layer->get_output_dims();
      for(auto i = output_dims.begin()+1; i != output_dims.end(); i++) {
        m_conv_output_spatial_prod *= *i;
        m_conv_output_spatial_dims.push_back(*i);
      }

      if(input_dims.size() != 3 && input_dims.size() != 4) {
        std::stringstream err;
        err << "The K-FAC only supports 2D or 3D tensors."
            << " layer: " << layer->get_name()
            << ", input_dims: ";
        for(auto i = input_dims.begin(); i != input_dims.end(); i++)
          err << (std::distance(input_dims.begin(), i) > 0 ? "," : "") << *i;
        LBANN_ERROR(err.str());
      }
    }

    if(m_is_conv && m_has_bias) {
      std::stringstream err;
      err << "The K-FAC does not currently support biases for convolutional layers."
          << " layer: " << layer->get_name();
      LBANN_ERROR(err.str());
    }
  }

  kfac_block_fc_conv(const kfac_block_fc_conv&) = default;
  kfac_block_fc_conv& operator=(const kfac_block_fc_conv&) = default;

  int get_local_memory_consumption() override {
    int total_size = 0;
    total_size += m_kronecker_inverse_A.Height() * m_kronecker_inverse_A.Width();
    total_size += m_kronecker_inverse_G.Height() * m_kronecker_inverse_G.Width();
    total_size += m_kronecker_average_A.Height() * m_kronecker_average_A.Width();
    total_size += m_kronecker_average_G.Height() * m_kronecker_average_G.Width();
    total_size += m_kronecker_factor_buf_A.Height() * m_kronecker_factor_buf_A.Width();
    total_size += m_kronecker_factor_buf_G.Height() * m_kronecker_factor_buf_G.Width();
    total_size += m_grad_buffer_v.Height() * m_grad_buffer_v.Width();
    return total_size;
  }

  void compute_local_kronecker_factors(
      lbann_comm* comm,
      bool print_matrix,
      bool print_matrix_summary) override;

  const std::vector<El::AbstractMatrix<DataType>*>
  get_local_kronecker_buffers() override {
    std::vector<El::AbstractMatrix<DataType>*> ret =
        {&m_kronecker_factor_buf_A, &m_kronecker_factor_buf_G};
    return ret;
  }

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
      bool use_eigen_decomposition,
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) override;

  void compute_preconditioned_gradients(
      lbann_comm* comm,
      DataType learning_rate_factor,
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) override;

  void initialize_activations_and_errors(
      lbann_comm* comm,
      int num_local_activations,
      int num_local_errors,
      int num_weights) override;

  void start_communication_forward_end(
      lbann_comm* comm) override;
  void end_communication_forward_end(
      lbann_comm* comm) override;
  void start_communication_backward_end(
      lbann_comm* comm) override;
  void end_communication_backward_end(
      lbann_comm* comm) override;

  const std::vector<El::AbstractMatrix<DataType>*>
  get_preconditioned_grad_buffers() override;

  int get_inverse_matrices(
      El::Matrix<DataType, Device>& output,
      int offset) override;

  int get_inverse_matrices_size(lbann_comm *comm) override;

  std::vector<int> get_inverse_matrices_size_vector(lbann_comm *comm) override;

  void resize_inverse_matrices_size(El::Matrix<double, El::Device::CPU>& inverse_matrices_size, int block_number) override;


  int set_inverse_matrices(
      El::Matrix<DataType, Device>& workspace,
      int offset,
      lbann_comm *comm) override;

  std::string get_info() const override {
    std::ostringstream oss;
    oss << kfac_block<Device>::get_info()
        << ", is_conv=" << m_is_conv;
    return oss.str();
  }

 private:

  /** @brief Gets the Kronecker factor matrix of a FC layer. **/
  static void get_kronecker_factor_fc(
      El::AbstractMatrix<DataType>& factor,
      const El::AbstractMatrix<DataType>& activations,
      DataType alpha);

  /** @brief Gets the Kronecker factor matrix of a convolutional layer. **/
  static void get_kronecker_factor_conv(
      El::Matrix<DataType, Device>& factor,
      El::Matrix<DataType, Device>& Acol,
      const El::Matrix<DataType, Device>& activations,
      DataType alpha,
      size_t local_batch_size, size_t num_channels,
      const std::vector<int>& spatial_dims,
      const convolution_layer<DataType, data_layout::DATA_PARALLEL, Device> *l_conv,
      bool use_im2col,
      const El::SyncInfo<Device>& sync_info);

  /** @brief Returns the pi constant. **/
  static double compute_pi(
      const El::Matrix<DataType, Device>& A,
      const El::Matrix<DataType, Device>& G,
      El::Matrix<DataType, Device>& ws,
      const El::SyncInfo<Device>& sync_info);

  /** @brief Get the pointer to its convolution_layer. **/
  convolution_layer<DataType, data_layout::DATA_PARALLEL, Device>*
  get_conv_layer() {
    return dynamic_cast<convolution_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(this->m_layer);
  }

  std::vector<std::tuple<std::string, size_t, size_t>>
  get_internal_matrix_info() const override;

  /** @brief Information to perform its computation. **/
  const bool m_is_conv, m_has_bias;
  size_t m_conv_input_spatial_prod, m_conv_output_spatial_prod;
  std::vector<int> m_conv_input_spatial_dims, m_conv_output_spatial_dims;

  /** @brief Lower triangle buffers of Kronecker factors. */
  El::Matrix<DataType, Device>
  m_kronecker_factor_buf_A, m_kronecker_factor_buf_G;

  /** @brief The heights of the Kronecker factors. */
  size_t m_height_A, m_height_G;

  /** @brief Exponential moving average of Kronecker factors. */
  El::Matrix<DataType, Device>
  m_kronecker_average_A, m_kronecker_average_G;

  /** @brief Inverse of the average Kronecker factors. */
  El::Matrix<DataType, Device>
  m_kronecker_inverse_A, m_kronecker_inverse_G;

  /** @brief Size and height of inverse matrices. */
  size_t m_Ainv_height=0, m_Ainv_width=0, m_Ginv_height=0, m_Ginv_width=0;

  /** @brief Vectorized gradient buffer (only for fully-connecter layers). */
  El::Matrix<DataType, Device>
  m_grad_buffer_v;

};

} // namespace lbann

#endif  // LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_FC_CONV_HPP_INCLUDED
