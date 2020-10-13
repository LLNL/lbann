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
// kfac_block_fc_conv .hpp .cpp - An FC/conv building block for the K-FAC method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_FC_CONV_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_FC_CONV_HPP_INCLUDED

#include "lbann/callbacks/kfac/kfac_block.hpp"
#include "lbann/layers/learning/convolution.hpp"

namespace lbann {
namespace callback {

/** An FC/conv building block for K-FAC.
 * TODO: Split into kfac_block_fc and kfac_block_conv.
 */
class kfac_block_fc_conv: public kfac_block {
 public:

  /** Constructor.
   */
  kfac_block_fc_conv(Layer *layer,
                     kfac *callback,
                     const size_t layer_id,
                     const size_t inverse_proc_rank,
                     const bool is_conv,
                     const size_t conv_input_spatial_prod,
                     const size_t conv_output_spatial_prod,
                     const std::vector<int> conv_input_spatial_dims,
                     const std::vector<int> conv_output_spatial_dims)
  : kfac_block(layer, callback, layer_id, inverse_proc_rank),
        m_is_conv(is_conv),
        m_conv_input_spatial_prod(conv_input_spatial_prod),
        m_conv_output_spatial_prod(conv_output_spatial_prod),
        m_conv_input_spatial_dims(conv_input_spatial_dims),
        m_conv_output_spatial_dims(conv_output_spatial_dims) {
  }
  kfac_block_fc_conv(const kfac_block_fc_conv&) = default;
  kfac_block_fc_conv& operator=(const kfac_block_fc_conv&) = default;

#ifdef LBANN_HAS_GPU

  void compute_local_kronecker_factors(
      lbann_comm* comm,
      const bool print_matrix,
      const bool print_matrix_summary) override;

  const std::vector<El::AbstractMatrix<DataType>*>
  get_local_kronecker_buffers() {
    std::vector<El::AbstractMatrix<DataType>*> ret =
        {&m_kronecker_factor_buf_A, &m_kronecker_factor_buf_G};
    return ret;
  }

  void update_kronecker_average(
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

  const std::vector<El::AbstractMatrix<DataType>*>
  get_preconditioned_grad_buffers() override;

#endif // LBANN_HAS_GPU

  std::string get_info() const override {
    std::ostringstream oss;
    oss << kfac_block::get_info()
        << ", is_conv=" << m_is_conv;
    return oss.str();
  }

 private:

#ifdef LBANN_HAS_GPU

  /** @brief Gets the Kronecker factor matrix of a FC layer. **/
  static void get_kronecker_factor_fc(
      El::AbstractMatrix<DataType>& factor,
      const El::AbstractMatrix<DataType>& activations,
      const DataType alpha);

  /** @brief Gets the Kronecker factor matrix of a convolutional layer. **/
  static void get_kronecker_factor_conv(
      El::Matrix<DataType, El::Device::GPU>& factor,
      El::Matrix<DataType, El::Device::GPU>& Acol,
      const El::Matrix<DataType, El::Device::GPU>& activations,
      const DataType alpha,
      const size_t local_batch_size, const size_t num_channels,
      const std::vector<int> spatial_dims,
      const convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU> *l_conv,
      const bool use_im2col,
      const cudaStream_t& stream);

  /** @brief Get diagonal elements of a matrix. **/
  template <typename TensorDataType>
  static void get_diagonal(
      TensorDataType * __restrict__ diag,
      const TensorDataType * __restrict__ A,
      const size_t height,
      const cudaStream_t& stream);

  /** @brief Returns the pi constant. **/
  static double compute_pi(
      const El::Matrix<DataType, El::Device::GPU>& A,
      const El::Matrix<DataType, El::Device::GPU>& G,
      El::Matrix<DataType, El::Device::GPU>& ws,
      const cudaStream_t& stream);

  /** @brief Transpose NC(D)HW matrix to N(D)HWC. **/
  template <typename TensorDataType>
  static void conv_transpose(
      const TensorDataType * __restrict__ activations,
      TensorDataType * __restrict__ act_columns,
      const size_t mini_batch_size, const size_t num_channels,
      const size_t spatial_prod,
      const cudaStream_t& stream);

  /** @brief Get the pointer to its convolution_layer. **/
  convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*
  get_conv_layer() {
    return dynamic_cast<convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>*>(m_layer);
  }

#endif // LBANN_HAS_GPU

  /** @brief Information to perform its computation. **/
  const bool m_is_conv;
  const size_t m_conv_input_spatial_prod, m_conv_output_spatial_prod;
  const std::vector<int> m_conv_input_spatial_dims, m_conv_output_spatial_dims;

#ifdef LBANN_HAS_GPU

  /** @brief Lower triangle buffers of Kronecker factors. */
  El::Matrix<DataType, El::Device::GPU>
  m_kronecker_factor_buf_A, m_kronecker_factor_buf_G;

  /** @brief The heights of the Kronecker factors. */
  size_t m_height_A, m_height_G;

  /** @brief Exponential moving average of Kronecker factors. */
  El::Matrix<DataType, El::Device::GPU>
  m_kronecker_average_A, m_kronecker_average_G;

  /** @brief Inverse of the average Kronecker factors. */
  El::Matrix<DataType, El::Device::GPU>
  m_kronecker_inverse_A, m_kronecker_inverse_G;

#endif // LBANN_HAS_GPU

};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_BLOCK_FC_CONV_HPP_INCLUDED
