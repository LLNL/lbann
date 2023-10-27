////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/execution_algorithms/kfac/kfac_block_bn.hpp"
#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

template <El::Device Device>
void kfac_block_bn<Device>::compute_local_kronecker_factors(
  lbann_comm* comm,
  const bool print_matrix,
  const bool print_matrix_summary)
{

  const auto& sync_info = this->get_sync_info();

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_child =
    dynamic_cast<const data_type_layer<DataType>&>(*child);
  const El::AbstractMatrix<DataType>& local_activations =
    dtl_parent.get_local_activations();
  const El::AbstractMatrix<DataType>& local_errors =
    dtl_child.get_local_error_signals();
  const auto mini_batch_size = dtl_parent.get_activations().Width();
  assert(mini_batch_size == dtl_child.get_error_signals().Width());
  const auto local_batch_size = local_activations.Width();

  assert(this->m_layer->num_weights() == 4); // scale, bias, r_mean, r_var
  auto& scales = this->m_layer->get_weights(0);
  auto& biases = this->m_layer->get_weights(1);
  const auto& s_dtw = dynamic_cast<data_type_weights<DataType>*>(&scales);
  const auto& b_dtw = dynamic_cast<data_type_weights<DataType>*>(&biases);
  const auto& scale_values = s_dtw->get_values();
  const auto& bias_values = b_dtw->get_values();
  assert(m_num_channels == (size_t)scale_values.Height());
  assert(m_num_channels == (size_t)scale_values.LocalHeight());
  assert(m_num_channels == (size_t)bias_values.Height());
  assert(m_num_channels == (size_t)bias_values.LocalHeight());

  auto& cols = this->get_workspace_matrix("bn_cols",
                                          m_num_channels * 2 * local_batch_size,
                                          m_spatial_prod);
  kfac_bn_util::compute_bn_factor_data2col<Device>(
    (El::Matrix<DataType, Device>&)local_activations,
    (El::Matrix<DataType, Device>&)local_errors,
    (El::Matrix<DataType, Device>&)scale_values.LockedMatrix(),
    (El::Matrix<DataType, Device>&)bias_values.LockedMatrix(),
    cols,
    local_batch_size,
    m_num_channels,
    m_spatial_prod,
    sync_info);

  auto& ones = this->get_workspace_matrix("bn_ones", m_spatial_prod, 1);
  auto& factor_v =
    this->get_workspace_matrix("bn_factor_v",
                               m_num_channels * 2 * local_batch_size,
                               1);
  El::Ones(ones, ones.Height(), ones.Width()); // TODO: Call once
  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           cols,
           ones,
           El::TypeTraits<DataType>::Zero(),
           factor_v);

  El::Matrix<DataType, Device> factor;
  factor.LockedAttach(m_num_channels * 2,
                      local_batch_size,
                      factor_v.LockedBuffer(),
                      m_num_channels * 2);
  auto& fisher_block = this->get_workspace_matrix("bn_fisher_block",
                                                  m_num_channels * 2,
                                                  m_num_channels * 2);
  const DataType alpha = mini_batch_size;
  El::Gemm(El::NORMAL,
           El::TRANSPOSE,
           alpha,
           factor,
           factor,
           El::TypeTraits<DataType>::Zero(),
           fisher_block);

  m_fisher_buf.Resize(fisher_block.Height() * (fisher_block.Height() + 1) / 2,
                      1);
  kfac::pack_lower_tri(m_fisher_buf, fisher_block, sync_info);

  // dump L2 norm of matrices
  if (comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)scale_values.LockedMatrix(),
             "scale")
        << ", "
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)bias_values.LockedMatrix(),
             "bias")
        << ", "
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)local_activations,
             "acts")
        << ", "
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)local_errors,
             "errs")
        << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_bn<Device>::update_kronecker_average(
  lbann_comm* comm,
  const DataType kronecker_decay,
  const bool print_matrix,
  const bool print_matrix_summary)
{

  const auto& sync_info = this->get_sync_info();

  auto& fisher_block = this->get_workspace_matrix("bn_fisher_block",
                                                  m_num_channels * 2,
                                                  m_num_channels * 2);
  kfac::unpack_lower_tri(fisher_block, m_fisher_buf, sync_info);

  // Update average Kronecker factors
  if (!this->m_has_kronecker_inverse) {
    El::Copy(fisher_block, m_fisher_average);
  }
  auto& Fave = m_fisher_average;
  kfac::update_kronecker_average(Fave,
                                 fisher_block,
                                 fisher_block.Height() * fisher_block.Width(),
                                 kronecker_decay,
                                 sync_info);
}

template <El::Device Device>
void kfac_block_bn<Device>::update_kronecker_inverse(
  lbann_comm* comm,
  const bool use_pi,
  const DataType damping_act,
  const DataType damping_err,
  const DataType learning_rate_factor,
  const bool use_eigen_decomposition,
  const bool print_matrix,
  const bool print_matrix_summary,
  const bool print_time)
{

  const auto& sync_info = this->get_sync_info();

  const auto& Fave = m_fisher_average;
  if (!this->m_has_kronecker_inverse) {
    this->m_has_kronecker_inverse = true;
    m_fisher_inverse.Resize(Fave.Height(), Fave.Width());
  }
  // TODO: Refactoring
  auto& Finv = m_fisher_inverse;
  auto& FLinv =
    this->get_workspace_matrix("bn_FLinv", Fave.Height(), Fave.Height());

  if (use_eigen_decomposition) {
    kfac::get_matrix_inverse_eigen(Finv,
                                   FLinv,
                                   Fave,
                                   comm->am_trainer_master() && print_time,
                                   DataType(damping_act),
                                   DataType(damping_err),
                                   true,
                                   sync_info);
  }
  else {
    kfac::get_matrix_inverse(Finv,
                             FLinv,
                             Fave,
                             comm->am_trainer_master() && print_time,
                             DataType(damping_act),
                             DataType(damping_err),
                             true,
                             sync_info);
  }

  // dump L2 norm of matrices
  if (comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << kfac::get_matrix_stat(Fave, "Fave") << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_bn<Device>::compute_preconditioned_gradients(
  lbann_comm* comm,
  const DataType learning_rate_factor,
  const bool print_matrix,
  const bool print_matrix_summary,
  const bool print_time)
{

  auto& Finv = m_fisher_inverse;
  auto& scales = this->m_layer->get_weights(0);
  auto& biases = this->m_layer->get_weights(1);
  optimizer* s_optimizer = scales.get_optimizer();
  optimizer* b_optimizer = biases.get_optimizer();
  auto* s_dto = dynamic_cast<data_type_optimizer<DataType>*>(s_optimizer);
  auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
  auto& s_grad = s_dto->get_gradient_sharded();
  auto& b_grad = b_dto->get_gradient_sharded();
  const El::Matrix<DataType, Device> s_gradients = s_grad.LockedMatrix();
  const El::Matrix<DataType, Device> b_gradients = b_grad.LockedMatrix();

  auto& stacked_grads =
    this->get_workspace_matrix("bn_stacked_grads", m_num_channels * 2, 1);
  auto stacked_grads_scale =
    El::View(stacked_grads, El::IR(0, m_num_channels), El::ALL);
  auto stacked_grads_bias = El::View(stacked_grads,
                                     El::IR(m_num_channels, m_num_channels * 2),
                                     El::ALL);
  El::Copy(s_gradients, stacked_grads_scale);
  El::Copy(b_gradients, stacked_grads_bias);

  auto& Fgrad = this->get_workspace_matrix("bn_Fgrad", m_num_channels * 2, 1);
  El::Gemm(El::NORMAL,
           El::NORMAL,
           learning_rate_factor,
           Finv,
           stacked_grads,
           El::TypeTraits<DataType>::Zero(),
           Fgrad);

  const auto Fgrad_scale = El::View(Fgrad, El::IR(0, m_num_channels), El::ALL);
  const auto Fgrad_bias =
    El::View(Fgrad, El::IR(m_num_channels, m_num_channels * 2), El::ALL);
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
           gradient_scale = El::TypeTraits<DataType>::One();
  auto& s_grad_buffer =
    s_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
  auto& b_grad_buffer =
    b_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
  El::Copy(Fgrad_scale, s_grad_buffer.Matrix());
  El::Copy(Fgrad_bias, b_grad_buffer.Matrix());

  // dump L2 norm of matrices
  if (comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << ", " << kfac::get_matrix_stat(Finv, "Finv") << ", "
        << kfac::get_matrix_stat(Fgrad, "Fgrad") << ", "
        << kfac::get_matrix_stat(s_gradients, "scale_grad") << ", "
        << kfac::get_matrix_stat(b_gradients, "bias_grad") << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_bn<Device>::get_preconditioned_grad_buffers()
{
  auto& scales = this->m_layer->get_weights(0);
  auto& biases = this->m_layer->get_weights(1);
  optimizer* s_optimizer = scales.get_optimizer();
  optimizer* b_optimizer = biases.get_optimizer();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
           gradient_scale = El::TypeTraits<DataType>::One();
  auto& s_grad_buffer =
    s_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
  auto& b_grad_buffer =
    b_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
  std::vector<El::AbstractMatrix<DataType>*> ret = {&s_grad_buffer.Matrix(),
                                                    &b_grad_buffer.Matrix()};
  return ret;
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_bn<Device>::get_internal_matrix_info() const
{
  std::vector<std::tuple<std::string, size_t, size_t>> list;
  const auto emplace = [&list](const std::string name,
                               const El::Matrix<DataType, Device>& m) {
    list.emplace_back(name, m.Height(), m.Width());
  };
  emplace("fisher_buf", m_fisher_buf);
  emplace("fisher_average", m_fisher_average);
  emplace("fisher_inverse", m_fisher_inverse);
  return list;
}

template <>
void kfac_bn_util::compute_bn_factor_data2col(
  const El::Matrix<DataType, El::Device::CPU>& activations,
  const El::Matrix<DataType, El::Device::CPU>& errors,
  const El::Matrix<DataType, El::Device::CPU>& scales,
  const El::Matrix<DataType, El::Device::CPU>& biases,
  El::Matrix<DataType, El::Device::CPU>& cols,
  const size_t batch_size,
  const size_t num_channels,
  const size_t spatial_prod,
  const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const size_t num_threads = batch_size * num_channels * spatial_prod;
#pragma omp parallel for
  for (size_t gid = 0; gid < num_threads; gid++) {
    const size_t i_c = gid % num_channels;
    const size_t i_n = (gid / num_channels) % batch_size;
    const size_t i_s = gid / num_channels / batch_size;
    const auto scale = scales.LockedBuffer()[i_c];
    const auto bias = biases.LockedBuffer()[i_c];
    const auto i_act =
      i_s + i_c * spatial_prod + i_n * spatial_prod * num_channels;
    const auto error = errors.LockedBuffer()[i_act];
    const auto act = (activations.LockedBuffer()[i_act] - bias) / scale;
    const auto i_out =
      i_c + i_n * num_channels * 2 + i_s * (num_channels * 2 * batch_size);
    cols.Buffer()[i_out] = error * act;
    cols.Buffer()[i_out + num_channels] = error;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Sub-grid Communication functions
////////////////////////////////////////////////////////////////////////////////

template <El::Device Device>
void kfac_block_bn<Device>::start_communication_forward_end(lbann_comm* comm)
{

  int num_local_activations = 1, num_weights = 2;

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& local_inputs = dtl_parent.get_activations();

  if (comm->get_KFAC_subgrid_create_two_models() or
      comm->get_grid_type() == GridType::NO_GRID) {
    this->m_parent_local_activations.resize(num_local_activations);
    this->m_weight_values.resize(num_weights);
  }
  else {
    if (this->m_parent_local_activations.size() == 0) {
      // Resize vectors
      this->m_parent_local_activations.resize(num_local_activations);
      this->m_weight_values.resize(num_weights);

      // Initialize Dist Matrices
      for (auto& input : this->m_parent_local_activations) {
        if (dtl_parent.get_data_layout() == data_layout::DATA_PARALLEL) {
          // Data Parallel Layout
          input = make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            this->m_subset_matrix.push_back(
              make_unique<
                El::
                  DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
                comm->get_subset_grid(),
                0));
        }
        else { // Model-Parallel Layout
          input = make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            LBANN_ERROR("Async prgoress is not supported for model-parallel "
                        "layer layout in sub-grid parallelism");
        }
      }

      for (auto& weight : this->m_weight_values) {
        weight = make_unique<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
          comm->get_secondary_grid(),
          0);
      }
    }

    if (comm->enable_subgrid_async_communication() == true) {
      const auto local_inputs_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(local_inputs));
      auto local_activations0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_parent_local_activations[0]));
      auto subset0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[0]));

      kfac::TranslateBetweenGridsVCAsync(*local_inputs_vc,
                                         *local_activations0,
                                         *subset0,
                                         this->m_requests_forward_end);

      int iter = 0;
      for (auto& weight : this->m_weight_values) {
        auto& weights = this->m_layer->get_weights(iter);
        const auto& dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
        const auto& weight_values = dtw->get_values();
        const auto weight_ptr = dynamic_cast<
          const El::
            DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
          &weight_values);
        auto weight_input_ptr = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
          &(*weight));
        El::Copy(weight_values, *weight);
        kfac::TranslateBetweenGridsSTARAsync(*weight_ptr,
                                             *weight_input_ptr,
                                             this->m_requests_forward_end);
        iter++;
      }
    }
    else {
      El::Copy(local_inputs, *this->m_parent_local_activations[0]);

      int iter = 0;
      for (auto& weight : this->m_weight_values) {
        auto& weights = this->m_layer->get_weights(iter);
        const auto& dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
        El::Copy(dtw->get_values(), *weight);
        iter++;
      }
    }
  }

  if (comm->get_grid_type() == GridType::NO_GRID or
      comm->get_KFAC_subgrid_create_two_models()) {

    for (auto& input : this->m_parent_local_activations) {
      if (dtl_parent.get_data_layout() == data_layout::DATA_PARALLEL)
        input = make_unique<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
      else
        input = make_unique<
          El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
    }

    for (auto& weight : this->m_weight_values) {
      weight = make_unique<
        El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
        comm->get_trainer_grid(),
        0);
    }

    El::LockedView(*(this->m_parent_local_activations[0]), local_inputs);

    for (int i = 0; i < num_weights; ++i) {
      const auto& weights = this->m_layer->get_weights(i);
      const auto& dtw =
        dynamic_cast<const data_type_weights<DataType>*>(&weights);
      auto& weight_values = dtw->get_values_sharded();
      const auto weight_ptr = dynamic_cast<
        const El::
          DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
        &weight_values);
      El::LockedView(*(this->m_weight_values[i]), *weight_ptr);
    }
  }
}

template <El::Device Device>
void kfac_block_bn<Device>::end_communication_forward_end(lbann_comm* comm)
{
  if ((comm->get_grid_type() == GridType::SECONDARY_GRID or
       comm->get_grid_type() == GridType::PRIMARY_GRID) and
      comm->enable_subgrid_async_communication() and
      comm->get_KFAC_subgrid_create_two_models() == false) {
    int num_local_activations = 1;
    auto primary_grid_ranks = comm->get_primary_grid_ranks();
    auto secondary_grid_ranks = comm->get_secondary_grid_ranks();

    for (auto& req : this->m_requests_forward_end) {
      ::Al::Wait<kfac::BackendT>(req);
    }

    if (primary_grid_ranks.size() < secondary_grid_ranks.size()) {
      for (int i = 0; i < num_local_activations; ++i) {
        auto local_activations0 = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
          &(*this->m_parent_local_activations[i]));
        auto subset0 = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
          &(*this->m_subset_matrix[i]));
        kfac::TranslateBetweenGridsVC(*subset0, *local_activations0);
      }
    }
    this->m_requests_forward_end.clear();
  }
}

template <El::Device Device>
void kfac_block_bn<Device>::start_communication_backward_end(lbann_comm* comm)
{
  int num_local_errors = 1, num_gradients = 2;
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_child =
    dynamic_cast<const data_type_layer<DataType>&>(*child);
  const auto& local_errors = dtl_child.get_error_signals();

  if (comm->get_KFAC_subgrid_create_two_models() or
      comm->get_grid_type() == GridType::NO_GRID) {
    this->m_child_local_errors.resize(num_local_errors);
    this->m_weight_gradients.resize(num_gradients);
  }
  else {
    if (this->m_child_local_errors.size() == 0) {
      // Resize vectors
      this->m_child_local_errors.resize(num_local_errors);
      this->m_weight_gradients.resize(num_gradients);

      // Initialize Dist Matrices
      for (auto& error : this->m_child_local_errors) {

        if (dtl_child.get_data_layout() == data_layout::DATA_PARALLEL) {
          error = make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            this->m_subset_matrix.push_back(
              make_unique<
                El::
                  DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
                comm->get_subset_grid(),
                0));
        }
        else {
          error = make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            LBANN_ERROR("Async prgoress is not supported for model-parallel "
                        "layer layout in sub-grid parallelism");
        }
      }
      // Initialize gradients
      for (auto& gradient : this->m_weight_gradients) {
        gradient = make_unique<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
          comm->get_secondary_grid(),
          0);
      }
    }

    if (comm->enable_subgrid_async_communication() == false) {
      El::Copy(local_errors, *this->m_child_local_errors[0]);

      int iter = 0;
      for (auto& gradient : this->m_weight_gradients) {
        auto& weights = this->m_layer->get_weights(iter);
        optimizer* optimizer = weights.get_optimizer();
        auto* dto = dynamic_cast<data_type_optimizer<DataType>*>(optimizer);

        El::Copy(dto->get_gradient_sharded(), *gradient);
        iter++;
      }
    }
    else {
      const auto local_errors_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(local_errors));
      auto local_errors0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_child_local_errors[0]));
      auto subset1 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[1]));

      kfac::TranslateBetweenGridsVCAsync(*local_errors_vc,
                                         *local_errors0,
                                         *subset1,
                                         this->m_requests_backward_end);
      int iter = 0;
      for (auto& gradient : this->m_weight_gradients) {
        auto& weights = this->m_layer->get_weights(iter);
        optimizer* optimizer = weights.get_optimizer();
        auto* dto = dynamic_cast<data_type_optimizer<DataType>*>(optimizer);
        auto& gradient_values = dto->get_gradient_sharded();
        const auto gradient_ptr = dynamic_cast<
          const El::
            DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
          &gradient_values);
        auto gradient_input_ptr = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
          &(*gradient));
        kfac::TranslateBetweenGridsSTARAsync(*gradient_ptr,
                                             *gradient_input_ptr,
                                             this->m_requests_backward_end);
        iter++;
      }
    } // Async progress
  }

  if (comm->get_grid_type() == GridType::NO_GRID or
      comm->get_KFAC_subgrid_create_two_models()) {
    for (auto& error : this->m_child_local_errors) {
      if (dtl_child.get_data_layout() == data_layout::DATA_PARALLEL)
        error = make_unique<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
      else
        error = make_unique<
          El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
    }
    // Initialize gradients
    for (auto& gradient : this->m_weight_gradients) {
      gradient = make_unique<
        El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
        comm->get_trainer_grid(),
        0);
    }
    El::LockedView(*(this->m_child_local_errors[0]), local_errors);

    for (int i = 0; i < num_gradients; ++i) {
      auto& weights = this->m_layer->get_weights(i);
      optimizer* optimizer = weights.get_optimizer();
      auto* dto = dynamic_cast<data_type_optimizer<DataType>*>(optimizer);
      auto& gradient_values = dto->get_gradient_sharded();
      const auto gradient_ptr = dynamic_cast<
        const El::
          DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
        &gradient_values);
      El::LockedView(*(this->m_weight_gradients[i]), *gradient_ptr);
    }
  }
}

template <El::Device Device>
void kfac_block_bn<Device>::end_communication_backward_end(lbann_comm* comm)
{
  if ((comm->get_grid_type() == GridType::SECONDARY_GRID or
       comm->get_grid_type() == GridType::PRIMARY_GRID) and
      comm->enable_subgrid_async_communication() and
      comm->get_KFAC_subgrid_create_two_models() == false) {
    auto primary_grid_ranks = comm->get_primary_grid_ranks();
    auto secondary_grid_ranks = comm->get_secondary_grid_ranks();

    for (auto& req : this->m_requests_backward_end) {
      ::Al::Wait<kfac::BackendT>(req);
    }

    if (primary_grid_ranks.size() < secondary_grid_ranks.size()) {
      auto local_errors0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_child_local_errors[0]));
      auto subset1 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[1]));
      kfac::TranslateBetweenGridsVC(*subset1, *local_errors0);
    }
    this->m_requests_backward_end.clear();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions to communicate inverse matrices
////////////////////////////////////////////////////////////////////////////////
template <El::Device Device>
int kfac_block_bn<Device>::get_inverse_matrices(
  El::Matrix<DataType, Device>& output,
  int offset)
{

  El::SyncInfo<Device> sync_info = El::SyncInfoFromMatrix(m_fisher_inverse);
  const size_t height = m_num_channels * 2;
  const size_t size_inv = height * height;

  El::copy::util::InterleaveMatrix(size_inv,
                                   1,
                                   m_fisher_inverse.LockedBuffer(),
                                   1,
                                   size_inv,
                                   output.Buffer(offset, 0),
                                   1,
                                   size_inv,
                                   sync_info);
  offset += size_inv;

  return offset;
}

template <El::Device Device>
int kfac_block_bn<Device>::set_inverse_matrices(
  El::Matrix<DataType, Device>& workspace,
  int offset,
  lbann_comm* comm)
{
  El::SyncInfo<Device> sync_info = El::SyncInfoFromMatrix(m_fisher_inverse);

  const size_t height = m_num_channels * 2;
  const size_t size_inv = height * height;

  if (m_fisher_inverse.Height() == 0)
    m_fisher_inverse.Resize(height, height);

  El::copy::util::InterleaveMatrix(size_inv,
                                   1,
                                   workspace.LockedBuffer(offset, 0),
                                   1,
                                   size_inv,
                                   m_fisher_inverse.Buffer(),
                                   1,
                                   size_inv,
                                   sync_info);
  offset += size_inv;
  return offset;
}

template <El::Device Device>
int kfac_block_bn<Device>::get_inverse_matrices_size(lbann_comm* comm)
{
  const size_t height = m_num_channels * 2;
  const size_t inverse_size = height * height;
  return inverse_size;
}

template class kfac_block_bn<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac_block_bn<El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
