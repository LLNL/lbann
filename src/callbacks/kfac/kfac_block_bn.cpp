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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/kfac/kfac_block_bn.hpp"
#include "lbann/callbacks/kfac/kfac_util.hpp"
#include "lbann/layers/data_type_layer.hpp"

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_GPU

void kfac_block_bn::compute_local_kronecker_factors(
    lbann_comm* comm,
    const bool print_matrix,
    const bool print_matrix_summary) {

  const auto stream = get_stream();

  const auto parent = m_layer->get_parent_layers()[0];
  const auto child = m_layer->get_child_layers()[0];
  const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
  const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
  const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
  const auto mini_batch_size = dtl_parent.get_activations().Width();
  assert(mini_batch_size == dtl_child.get_error_signals().Width());
  const auto local_batch_size = local_activations.Width();

  assert(m_layer->num_weights() == 4); // scale, bias, r_mean, r_var
  auto& scales = m_layer->get_weights(0);
  auto& biases = m_layer->get_weights(1);
  optimizer *s_optimizer = scales.get_optimizer();
  optimizer *b_optimizer = biases.get_optimizer();
  auto* s_dto = dynamic_cast<data_type_optimizer<DataType>*>(s_optimizer);
  auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
  El::Matrix<DataType, El::Device::GPU> s_gradients = s_dto->get_gradient().Matrix();
  El::Matrix<DataType, El::Device::GPU> b_gradients = b_dto->get_gradient().Matrix();
  const auto &s_dtw = dynamic_cast<data_type_weights<DataType>*>(&scales);
  const auto &b_dtw = dynamic_cast<data_type_weights<DataType>*>(&biases);
  const auto &scale_values = s_dtw->get_values();
  const auto &bias_values = b_dtw->get_values();
  assert(m_num_channels == (size_t) scale_values.Height());
  assert(m_num_channels == (size_t) scale_values.LocalHeight());
  assert(m_num_channels == (size_t) bias_values.Height());
  assert(m_num_channels == (size_t) bias_values.LocalHeight());

  auto& cols = get_workspace_matrix(
      "bn_cols",
      m_num_channels*2*local_batch_size,
      m_spatial_prod);
  compute_bn_factor_data2col(
      local_activations.LockedBuffer(),
      local_errors.LockedBuffer(),
      scale_values.LockedMatrix().LockedBuffer(),
      bias_values.LockedMatrix().LockedBuffer(),
      cols.Buffer(),
      local_batch_size,
      m_num_channels,
      m_spatial_prod,
      stream);

  auto& ones = get_workspace_matrix(
      "bn_ones",
      m_spatial_prod, 1);
  auto& factor_v = get_workspace_matrix(
      "bn_factor_v",
      m_num_channels*2*local_batch_size, 1);
  El::Ones(ones, ones.Height(), ones.Width()); // TODO: Call once
  El::Gemm(
      El::NORMAL, El::NORMAL,
      El::TypeTraits<DataType>::One(), cols, ones,
      El::TypeTraits<DataType>::Zero(), factor_v);

  El::Matrix<DataType, El::Device::GPU> factor;
  factor.LockedAttach(m_num_channels*2, local_batch_size,
                      factor_v.LockedBuffer(),
                      m_num_channels*2);
  auto& fisher_block = get_workspace_matrix(
      "bn_fisher_block",
      m_num_channels*2, m_num_channels*2);
  const DataType alpha = mini_batch_size;
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, factor, factor,
      El::TypeTraits<DataType>::Zero(), fisher_block);

  m_fisher_buf.Resize(fisher_block.Height()*(fisher_block.Height()+1)/2, 1);
  kfac_util::pack_lower_tri(
      m_fisher_buf.Buffer(), fisher_block.LockedBuffer(),
      fisher_block.Height(), stream);

  // dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat(scale_values.LockedMatrix(), "scale")
        << ", " << kfac_util::get_matrix_stat(bias_values.LockedMatrix(), "bias")
        << ", " << kfac_util::get_matrix_stat(local_activations, "acts")
        << ", " << kfac_util::get_matrix_stat(local_errors, "errs")
        << std::endl;
    std::cout << oss.str();
  }

}

void kfac_block_bn::update_kronecker_average(
    lbann_comm* comm,
    const DataType kronecker_decay,
    const bool print_matrix,
    const bool print_matrix_summary) {

  const auto stream = get_stream();

  auto& fisher_block = get_workspace_matrix(
      "bn_fisher_block",
      m_num_channels*2, m_num_channels*2);
  kfac_util::unpack_lower_tri(
      fisher_block.Buffer(), m_fisher_buf.LockedBuffer(),
      fisher_block.Height(), stream);

  // Update average Kronecker factors
  if(!m_has_kronecker_inverse) {
    El::Copy(fisher_block, m_fisher_average);
  }
  auto &Fave = m_fisher_average;
  kfac_util::update_kronecker_average(
      Fave.Buffer(), fisher_block.Buffer(),
      fisher_block.Height()*fisher_block.Width(),
      kronecker_decay, stream);

}

void kfac_block_bn::update_kronecker_inverse(
    lbann_comm* comm,
    const bool use_pi,
    const DataType damping_act, const DataType damping_err,
    const DataType learning_rate_factor,
    const bool print_matrix,
    const bool print_matrix_summary,
    const bool print_time) {

  const auto stream = get_stream();

  const auto &Fave = m_fisher_average;
  if(!m_has_kronecker_inverse) {
    m_has_kronecker_inverse = true;
    m_fisher_inverse.Resize(Fave.Height(), Fave.Width());
  }
  // TODO: Refactoring
  auto& Finv = m_fisher_inverse;
  auto& FLinv = get_workspace_matrix(
      "bn_FLinv",
      Fave.Height(), Fave.Height());
  kfac_util::get_matrix_inverse(
      Finv, FLinv, Fave, comm->am_trainer_master() && print_time,
      DataType(damping_act), DataType(damping_err),
      true, stream);

  // dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat(Fave, "Fave")
        << std::endl;
    std::cout << oss.str();
  }

  auto& scales = m_layer->get_weights(0);
  auto& biases = m_layer->get_weights(1);
  optimizer *s_optimizer = scales.get_optimizer();
  optimizer *b_optimizer = biases.get_optimizer();
  auto* s_dto = dynamic_cast<data_type_optimizer<DataType>*>(s_optimizer);
  auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
  El::Matrix<DataType, El::Device::GPU> s_gradients = s_dto->get_gradient().Matrix();
  El::Matrix<DataType, El::Device::GPU> b_gradients = b_dto->get_gradient().Matrix();

  auto& stacked_grads = get_workspace_matrix(
      "bn_stacked_grads",
      m_num_channels*2, 1);
  auto stacked_grads_scale = El::View(
      stacked_grads, El::IR(0, m_num_channels), El::ALL);
  auto stacked_grads_bias = El::View(
      stacked_grads, El::IR(m_num_channels, m_num_channels*2), El::ALL);
  El::Copy(s_gradients, stacked_grads_scale);
  El::Copy(b_gradients, stacked_grads_bias);

  auto& Fgrad = get_workspace_matrix(
      "bn_Fgrad",
      m_num_channels*2, 1);
  El::Gemm(
      El::NORMAL, El::NORMAL,
      learning_rate_factor, Finv, stacked_grads,
      El::TypeTraits<DataType>::Zero(), Fgrad);

  const auto Fgrad_scale = El::View(Fgrad, El::IR(0, m_num_channels), El::ALL);
  const auto Fgrad_bias = El::View(Fgrad, El::IR(m_num_channels, m_num_channels*2), El::ALL);
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  auto& s_grad_buffer = s_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);
  auto& b_grad_buffer = b_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);
  El::Copy(Fgrad_scale, s_grad_buffer.Matrix());
  El::Copy(Fgrad_bias, b_grad_buffer.Matrix());

  // dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< m_layer->get_name() << ": "
        << ", " << kfac_util::get_matrix_stat(Finv, "Finv")
        << ", " << kfac_util::get_matrix_stat(Fgrad, "Fgrad")
        << ", " << kfac_util::get_matrix_stat(s_gradients, "scale_grad")
        << ", " << kfac_util::get_matrix_stat(b_gradients, "bias_grad")
        << std::endl;
    std::cout << oss.str();
  }
}

const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_bn::get_preconditioned_grad_buffers() {
  auto& scales = m_layer->get_weights(0);
  auto& biases = m_layer->get_weights(1);
  optimizer *s_optimizer = scales.get_optimizer();
  optimizer *b_optimizer = biases.get_optimizer();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  auto& s_grad_buffer = s_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);
  auto& b_grad_buffer = b_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);
  std::vector<El::AbstractMatrix<DataType>*>
      ret = {&s_grad_buffer.Matrix(), &b_grad_buffer.Matrix()};
  return ret;
}

std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_bn::get_internal_matrix_info() const {
  std::vector<std::tuple<std::string, size_t, size_t>> list;
  const auto emplace =
      [&list](const std::string name,
              const El::Matrix<DataType, El::Device::GPU>& m) {
        list.emplace_back(name, m.Height(), m.Width());
      };
  emplace("fisher_buf", m_fisher_buf);
  emplace("fisher_average", m_fisher_average);
  emplace("fisher_inverse", m_fisher_inverse);
  return list;
}

#endif // LBANN_HAS_GPU

} // namespace callback
} // namespace lbann
