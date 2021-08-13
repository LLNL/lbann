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

#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_BN_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_BN_HPP_INCLUDED

#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/convolution.hpp"

namespace lbann {

namespace kfac_bn_util {

/** @brief The memory copy part of compute_bn_factor. Combined with
 *  GEMM. **/
template <El::Device Device>
void compute_bn_factor_data2col(
    const El::Matrix<DataType, Device>& activations,
    const El::Matrix<DataType, Device>& errors,
    const El::Matrix<DataType, Device>& scales,
    const El::Matrix<DataType, Device>& biases,
    El::Matrix<DataType, Device>& cols,
    size_t batch_size,
    size_t num_channels,
    size_t spatial_prod,
    const El::SyncInfo<Device>& sync_info);

} // namespace kfac_bn_util

/** A BN building block for K-FAC.
 */
template <El::Device Device>
class kfac_block_bn: public kfac_block<Device> {
 public:

  /** Constructor.
   */
  kfac_block_bn(Layer* layer,
                kfac::KFACExecutionContext* context,
                size_t layer_id,
                size_t inverse_proc_rank)
      : kfac_block<Device>(layer, context, layer_id, inverse_proc_rank) {
    const auto parent = layer->get_parent_layers()[0];
    const bool is_after_fc =
        (dynamic_cast<const fully_connected_layer<DataType,
         data_layout::DATA_PARALLEL, Device>*>(parent) != nullptr);
    m_is_after_conv =
        (dynamic_cast<const convolution_layer<DataType,
         data_layout::DATA_PARALLEL, Device>*>(parent) != nullptr);
    if(!is_after_fc && !m_is_after_conv) {
      std::stringstream err;
      err << "The K-FAC only supports batch-normalization layers after "
          << "fully-connected layers or convolutional layers."
          << " layer: " << layer->get_name()
          << " parent type: " << parent->get_type();
      LBANN_ERROR(err.str());
    }

    if(is_after_fc) {
      const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
      const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
      m_num_channels = local_activations.Height();
      m_spatial_prod = 1;
    } else {
      const auto input_dims = layer->get_input_dims();
      m_num_channels = input_dims[0];
      m_spatial_prod = 1;
      // std::accumulate might overflow for large 3D layers
      for(auto i = input_dims.begin()+1; i != input_dims.end(); i++)
        m_spatial_prod *= *i;
    }
  }
  kfac_block_bn(const kfac_block_bn&) = default;
  kfac_block_bn& operator=(const kfac_block_bn&) = default;

  void compute_local_kronecker_factors(
      lbann_comm* comm,
      bool print_matrix,
      bool print_matrix_summary) override;

  const std::vector<El::AbstractMatrix<DataType>*>
  get_local_kronecker_buffers() override {
    std::vector<El::AbstractMatrix<DataType>*> ret = {&m_fisher_buf};
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
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) override;
  
  void compute_preconditioned_gradients(
      lbann_comm* comm,
      DataType learning_rate_factor,
      bool print_matrix,
      bool print_matrix_summary,
      bool print_time) override;

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

  std::vector<std::tuple<std::string, size_t, size_t>>
  get_internal_matrix_info() const override;

  std::string get_info() const override {
    std::ostringstream oss;
    oss << kfac_block<Device>::get_info()
        << ", is_after_conv=" << m_is_after_conv;
    return oss.str();
  }
  /** @brief Copy inverse matrices to output buffer. */
  int
  get_inverse_matrices(
      El::Matrix<DataType, Device>& output,
      int offset) override;

  /** @brief Get inverse matrices size (offset). */
  int
  get_inverse_matrices_size(lbann_comm *comm) override;

  /** @brief Get inverse matrices size vector */
  std::vector<int>
  get_inverse_matrices_size_vector(lbann_comm *comm) override
  {
    LBANN_ERROR("Sub-grid parallelism  is not implemented for BN layer");
  }

  /** @brief Get inverse matrices size vector */
  void
  resize_inverse_matrices_size(El::Matrix<double, El::Device::CPU>& inverse_matrices_size, int block_number) override
  {
    LBANN_ERROR("Sub-grid parallelism  is not implemented for BN layer");
  }

  /** @brief Copy inverse matrices from output buffer. */
  int
  set_inverse_matrices(
      El::Matrix<DataType, Device>& workspace,
      int offset,
      lbann_comm *comm) override;

 private:

  /** @brief Information to perform its computation. **/
  bool m_is_after_conv;
  size_t m_num_channels, m_spatial_prod;

  /** @brief Lower triangle buffers of the Fisher block. */
  El::Matrix<DataType, Device>
  m_fisher_buf;

  /** @brief Exponential moving average of the Fisher matrix. */
  El::Matrix<DataType, Device>
  m_fisher_average;

  /** @brief Inverse of the average Fisher matrix. */
  El::Matrix<DataType, Device>
  m_fisher_inverse;

};

} // namespace lbann

#endif  // LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_BLOCK_BN_HPP_INCLUDED
