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

#include "lbann/execution_algorithms/kfac/kfac_block_fc_conv.hpp"
#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

template <El::Device Device>
void kfac_block_fc_conv<Device>::compute_local_kronecker_factors(
  lbann_comm* comm,
  const bool print_matrix,
  const bool print_matrix_summary)
{

  const auto& sync_info = this->get_sync_info();

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const El::Matrix<DataType, Device>& local_activations =
    this->m_parent_local_activations[0]->Matrix();
  const El::Matrix<DataType, Device>& local_errors =
    this->m_child_local_errors[0]->Matrix();

  const auto mini_batch_size = dtl_parent.get_activations().Width();
  const auto local_batch_size = local_activations.Width();

  // Compute Kronecker factors, assuming that local_errors are
  // already multiplied by 1/N in the loss layer.
  const auto input_dims = this->m_layer->get_input_dims();   // CHW
  const auto output_dims = this->m_layer->get_output_dims(); // KH'W'
  const size_t num_input_channels = input_dims[0];
  const size_t num_output_channels = output_dims[0];
  m_height_A = local_activations.Height();
  if (m_is_conv) {
    const auto conv_dims = get_conv_layer()->get_conv_dims();
    m_height_A = num_input_channels * get_linear_size(conv_dims);
  }
  if (m_has_bias)
    m_height_A++;

  m_height_G = !m_is_conv ? local_errors.Height() : num_output_channels;
  auto& A = this->get_workspace_matrix("A", m_height_A, m_height_A);
  auto& G = this->get_workspace_matrix("G", m_height_G, m_height_G);
  if (!m_is_conv) {
    if (m_has_bias) {
      auto& local_activations_with_ones =
        this->get_workspace_matrix("local_activations_with_ones",
                                   local_activations.Height() + 1,
                                   local_activations.Width());
      auto local_activations_dst =
        El::View(local_activations_with_ones,
                 El::IR(0, local_activations.Height()),
                 El::ALL);
      El::Copy(local_activations, local_activations_dst);
      auto local_activations_ones = El::View(
        local_activations_with_ones,
        El::IR(local_activations.Height(), local_activations.Height() + 1),
        El::ALL);
      El::Ones(local_activations_ones, 1, local_activations.Width());
      get_kronecker_factor_fc(A,
                              local_activations_with_ones,
                              1.0 / mini_batch_size);
    }
    else {
      get_kronecker_factor_fc(A, local_activations, 1.0 / mini_batch_size);
    }
    get_kronecker_factor_fc(G, local_errors, mini_batch_size);
  }
  else {
    assert((size_t)local_activations.Height() ==
           num_input_channels * m_conv_input_spatial_prod);
    assert((size_t)local_errors.Height() ==
           num_output_channels * m_conv_output_spatial_prod);

    const auto Acol_size =
      get_im2col_output_size(local_batch_size,
                             num_input_channels,
                             m_conv_input_spatial_dims.size(),
                             &(m_conv_input_spatial_dims[0]),
                             &(get_conv_layer()->get_pads()[0]),
                             &(get_conv_layer()->get_conv_dims()[0]),
                             &(get_conv_layer()->get_strides()[0]));
    auto& Acol =
      this->get_workspace_matrix("Acol", Acol_size.first, Acol_size.second);
    auto& Gcol =
      this->get_workspace_matrix("Gcol",
                                 num_output_channels,
                                 local_batch_size * m_conv_output_spatial_prod);
    get_kronecker_factor_conv(A,
                              Acol,
                              local_activations,
                              1.0 / mini_batch_size,
                              local_batch_size,
                              num_input_channels,
                              m_conv_input_spatial_dims,
                              get_conv_layer(),
                              true,
                              sync_info);
    get_kronecker_factor_conv(G,
                              Gcol,
                              local_errors,
                              DataType(mini_batch_size) /
                                m_conv_output_spatial_prod,
                              local_batch_size,
                              num_output_channels,
                              m_conv_output_spatial_dims,
                              get_conv_layer(),
                              false,
                              sync_info);
  }

  m_kronecker_factor_buf_A.Resize(A.Height() * (A.Height() + 1) / 2, 1);
  m_kronecker_factor_buf_G.Resize(G.Height() * (G.Height() + 1) / 2, 1);
  kfac::pack_lower_tri(m_kronecker_factor_buf_A, A, sync_info);
  kfac::pack_lower_tri(m_kronecker_factor_buf_G, G, sync_info);

  // Dump L2 norm of matrices
  if (comm->am_trainer_master() && print_matrix_summary) {
    // TODO: Show weights' stats
    // const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
    // const auto &w_values = dtw->get_values();
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name()
        << ": "
        // << kfac::get_matrix_stat(w_values.LockedMatrix(), "W")
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)local_activations,
             "acts")
        << ", "
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)local_errors,
             "errs")
        << ", "
        << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)A, "A")
        << ", "
        << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)G, "G")
        << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::update_kronecker_average(
  lbann_comm* comm,
  const DataType kronecker_decay,
  const bool print_matrix,
  const bool print_matrix_summary)
{

  const auto& sync_info = this->get_sync_info();

  auto& A = this->get_workspace_matrix("A", m_height_A, m_height_A);
  auto& G = this->get_workspace_matrix("G", m_height_G, m_height_G);

  kfac::unpack_lower_tri(A, m_kronecker_factor_buf_A, sync_info);
  kfac::unpack_lower_tri(G, m_kronecker_factor_buf_G, sync_info);

  // Update average Kronecker factors
  if (!this->m_has_kronecker_inverse) {
    El::Copy(A, m_kronecker_average_A);
    El::Copy(G, m_kronecker_average_G);
  }
  auto& Aave = m_kronecker_average_A;
  auto& Gave = m_kronecker_average_G;
  kfac::update_kronecker_average(Aave,
                                 A,
                                 A.Height() * A.Width(),
                                 kronecker_decay,
                                 sync_info);
  kfac::update_kronecker_average(Gave,
                                 G,
                                 G.Height() * G.Width(),
                                 kronecker_decay,
                                 sync_info);

  // Dump matrices for debugging
  if (comm->am_trainer_master() && print_matrix) {
    if (comm->am_trainer_master()) {
      std::cout << std::endl;
      El::Print(A, "A");
      std::cout << std::endl;
      El::Print(G, "G");
      std::cout << std::endl;
      El::Print(Aave, "Aave");
      std::cout << std::endl;
      El::Print(Gave, "Gave");
      std::cout << std::endl;
    }
  }

  // Dump L2 norm of matrices
  if (comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << kfac::get_matrix_stat(Aave, "Aave") << ", "
        << kfac::get_matrix_stat(Gave, "Gave") << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::update_kronecker_inverse(
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

  // TODO: Refactoring
  const auto& Aave = m_kronecker_average_A;
  const auto& Gave = m_kronecker_average_G;
  // Compute the pi constant
  DataType pi = 1.0;
  if (use_pi) {
    auto& ws =
      this->get_workspace_matrix("pi_ws",
                                 std::max(Aave.Height(), Gave.Height()) * 2 + 1,
                                 1);
    pi = compute_pi(Aave, Gave, ws, sync_info);
  }
  // Compute the inverse of the factors
  // Since setting different damping constants for A and G is an
  // alternative heuristics to pi, they should be the same if pi is used.
  if (use_pi && damping_act != damping_err) {
    std::stringstream err;
    err << "Damping values for activations and errors are different while the "
           "pi constant is used."
        << " layer: " << this->m_layer->get_name()
        << ", damping_act: " << damping_act << ", damping_err: " << damping_err;
    LBANN_WARNING(err.str());
  }

  if (!this->m_has_kronecker_inverse) {
    this->m_has_kronecker_inverse = true;
    m_kronecker_inverse_A.Resize(Aave.Height(), Aave.Width());
    m_kronecker_inverse_G.Resize(Gave.Height(), Gave.Width());
  }
  // TODO: Refactoring
  auto& Ainv = m_kronecker_inverse_A;
  auto& Ginv = m_kronecker_inverse_G;
  auto& ALinv =
    this->get_workspace_matrix("ALinv", Aave.Height(), Aave.Height());
  auto& GLinv =
    this->get_workspace_matrix("GLinv", Gave.Height(), Gave.Height());

  if (use_eigen_decomposition) {
    kfac::get_matrix_inverse_eigen(Ainv,
                                   ALinv,
                                   Aave,
                                   comm->am_trainer_master() && print_time,
                                   DataType(damping_act * pi),
                                   0,
                                   false,
                                   sync_info);
    kfac::get_matrix_inverse_eigen(Ginv,
                                   GLinv,
                                   Gave,
                                   comm->am_trainer_master() && print_time,
                                   DataType(damping_err / pi),
                                   0,
                                   false,
                                   sync_info);
  }
  else {
    kfac::get_matrix_inverse(Ainv,
                             ALinv,
                             Aave,
                             comm->am_trainer_master() && print_time,
                             DataType(damping_act * pi),
                             0,
                             false,
                             sync_info);
    kfac::get_matrix_inverse(Ginv,
                             GLinv,
                             Gave,
                             comm->am_trainer_master() && print_time,
                             DataType(damping_err / pi),
                             0,
                             false,
                             sync_info);
  }

  if (print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: pi=" << pi << " @ " << this->m_layer->get_name()
        << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::compute_preconditioned_gradients(
  lbann_comm* comm,
  const DataType learning_rate_factor,
  const bool print_matrix,
  const bool print_matrix_summary,
  const bool print_time)
{

  auto& weights = this->m_layer->get_weights(0);
  optimizer* w_optimizer = weights.get_optimizer();
  auto* w_dto = dynamic_cast<data_type_optimizer<DataType>*>(w_optimizer);
  auto learning_rate = w_dto->get_learning_rate();
  auto& Ainv = m_kronecker_inverse_A;
  auto& Ginv = m_kronecker_inverse_G;
  auto& w_grad = w_dto->get_gradient_sharded();
  const auto& w_grads_orig = w_grad.LockedMatrix();
  El::Matrix<DataType, Device> w_gradients;

  // BVE FIXME unused variables
  // auto& w_grads_scaling = this->get_workspace_matrix(
  //         "w_grads_scaling", 1, 1);
  // auto& b_grads_scaling = this->get_workspace_matrix(
  //         "b_grads_scaling", 1, 1);

  // auto& w_grads_scaling_mat = this->get_workspace_matrix(
  //         "w_grads_scaling_mat", w_grads_orig.Height(),
  //         w_grads_orig.Width());

  // w_gradients is already synchronized among processes.
  if (m_is_conv) {
    const auto num_output_channels = this->m_layer->get_output_dims()[0];
    assert((w_grads_orig.Height() % num_output_channels) == 0);
    const auto height = w_grads_orig.Height() / num_output_channels;
    w_gradients.LockedAttach(height,
                             num_output_channels,
                             w_grads_orig.LockedBuffer(),
                             height);
  }
  else {
    if (m_has_bias) {
      auto& w_grads_concat =
        this->get_workspace_matrix("A",
                                   w_grads_orig.Height(),
                                   w_grads_orig.Width() + 1);

      auto w_grads_concat_weights =
        El::View(w_grads_concat, El::ALL, El::IR(0, w_grads_orig.Width()));
      auto w_grads_concat_biases =
        El::View(w_grads_concat,
                 El::ALL,
                 El::IR(w_grads_orig.Width(), w_grads_orig.Width() + 1));

      auto& biases = this->m_layer->get_weights(1);
      optimizer* b_optimizer = biases.get_optimizer();
      auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
      auto& b_grad = b_dto->get_gradient_sharded();
      const auto& b_grads_orig = b_grad.LockedMatrix();

      El::Copy(w_grads_orig, w_grads_concat_weights);
      El::Copy(b_grads_orig, w_grads_concat_biases);

      w_gradients.LockedAttach(w_grads_concat.Height(),
                               w_grads_concat.Width(),
                               w_grads_concat.LockedBuffer(),
                               w_grads_concat.Height());
    }
    else {
      w_gradients.LockedAttach(w_grads_orig.Height(),
                               w_grads_orig.Width(),
                               w_grads_orig.LockedBuffer(),
                               w_grads_orig.Height());
    }
  }

  // Compute preconditioned gradients
  auto& Gg = this->get_workspace_matrix("Gg",
                                        Ginv.Height(),
                                        m_is_conv ? w_gradients.Height()
                                                  : w_gradients.Width());

  El::Gemm(El::NORMAL,
           m_is_conv ? El::TRANSPOSE : El::NORMAL,
           El::TypeTraits<DataType>::One(),
           Ginv,
           w_gradients,
           El::TypeTraits<DataType>::Zero(),
           Gg);
  auto& Fgrad =
    this->get_workspace_matrix("Fgrad", Ginv.Height(), Ainv.Width());
  El::Gemm(El::NORMAL,
           El::NORMAL,
           learning_rate_factor,
           Gg,
           Ainv,
           El::TypeTraits<DataType>::Zero(),
           Fgrad);

  // Update gradients in the buffer
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
           gradient_scale = El::TypeTraits<DataType>::One();
  // grad_buffer is already synchronized among processes,
  // and won't be all-reduced later.
  auto& grad_buffer =
    w_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
  if (m_is_conv) {
    El::Matrix<DataType, Device> grad_buffer_v;
    grad_buffer_v.Attach(Fgrad.Width(),
                         Fgrad.Height(),
                         grad_buffer.Buffer(),
                         Fgrad.Width());

    // Cliping
    //  El::Gemm(
    //    El::NORMAL, El::NORMAL, El::TypeTraits<DataType>::One(),
    //    El::Reshape(1, Fgrad.Height()*Fgrad.Width(), Fgrad),
    //    El::Reshape(w_grads_orig.Height()*w_grads_orig.Width(), 1,
    //    w_grads_orig), El::TypeTraits<DataType>::Zero(), w_grads_scaling
    //  );

    El::Matrix<DataType> XCPU(Fgrad);
    auto w_grads_scaling = El::Dot(XCPU, XCPU);
    auto w_grads_scaling_factor = std::min(
      1.0,
      std::sqrt(0.001 / (w_grads_scaling * learning_rate * learning_rate)));
    El::Scale(w_grads_scaling_factor, Fgrad);

    El::Transpose(Fgrad, grad_buffer_v);
  }
  else {
    assert(Fgrad.Height() == w_gradients.Height());
    assert(Fgrad.Width() == w_gradients.Width());

    if (m_has_bias) {
      auto& biases = this->m_layer->get_weights(1);
      optimizer* b_optimizer = biases.get_optimizer();
      // BVE FIXME unused variable
      // auto* b_dto =
      // dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer); const auto&
      // b_grads_orig = b_dto->get_gradient_sharded().LockedMatrix();

      auto& grad_buffer_biases =
        b_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
      auto Fgrad_weights =
        El::View(Fgrad, El::ALL, El::IR(0, grad_buffer.Width()));
      auto Fgrad_biases =
        El::View(Fgrad,
                 El::ALL,
                 El::IR(grad_buffer.Width(), grad_buffer.Width() + 1));

      // BVE FIXME
      // auto& b_grads_scaled = this->get_workspace_matrix(
      //     "b_grads_scaling_mat", b_grads_orig.Height(),
      //     b_grads_orig.Width());

      // Cliping
      //  El::Gemm(
      //    El::NORMAL, El::NORMAL, El::TypeTraits<DataType>::One(),
      //    El::Reshape(1, Fgrad_weights.Height()*Fgrad_weights.Width(),
      //    Fgrad_weights),
      //    El::Reshape(w_grads_orig.Height()*w_grads_orig.Width(), 1,
      //    w_grads_orig), El::TypeTraits<DataType>::Zero(), w_grads_scaling
      //  );
      //  El::Gemm(
      //    El::NORMAL, El::NORMAL, El::TypeTraits<DataType>::One(),
      //    El::Reshape(1, Fgrad_biases.Height()*Fgrad_biases.Width(),
      //    Fgrad_biases),
      //    El::Reshape(b_grads_orig.Height()*b_grads_orig.Width(), 1,
      //    b_grads_orig), El::TypeTraits<DataType>::Zero(), b_grads_scaling
      //  );
      El::Matrix<DataType> XCPU(Fgrad_weights);
      El::Matrix<DataType> BCPU(Fgrad_biases);
      auto w_grads_scaling = El::Dot(XCPU, XCPU);
      auto b_grads_scaling = El::Dot(BCPU, BCPU);

      auto w_grads_scaling_factor = std::min(
        1.0,
        std::sqrt(0.001 / (w_grads_scaling * learning_rate * learning_rate)));
      auto w_bias_scaling_factor = std::min(
        1.0,
        std::sqrt(0.001 / (b_grads_scaling * learning_rate * learning_rate)));

      El::Scale(w_grads_scaling_factor, Fgrad_weights);
      El::Scale(w_bias_scaling_factor, Fgrad_biases);

      El::Copy(Fgrad_weights, grad_buffer.Matrix());
      El::Copy(Fgrad_biases, grad_buffer_biases.Matrix());
    }
    else {
      // Cliping
      //  El::Gemm(
      //    El::NORMAL, El::NORMAL, El::TypeTraits<DataType>::One(),
      //    El::Reshape(1, Fgrad.Height()*Fgrad.Width(), Fgrad),
      //    El::Reshape(w_grads_orig.Height()*w_grads_orig.Width(), 1,
      //    w_grads_orig), El::TypeTraits<DataType>::Zero(), w_grads_scaling
      //  );
      El::Matrix<DataType> XCPU(Fgrad);
      auto w_grads_scaling = El::Dot(XCPU, XCPU);
      auto w_grads_scaling_factor = std::min(
        1.0,
        std::sqrt(0.001 / (w_grads_scaling * learning_rate * learning_rate)));
      El::Scale(w_grads_scaling_factor, Fgrad);

      El::Copy(Fgrad, grad_buffer.Matrix());
    }
  }

  // Dump matrices for debugging
  if (print_matrix) {
    std::cout << std::endl;
    El::Print(Ainv, "Ainv");
    std::cout << std::endl;
    El::Print(Ginv, "Ginv");
    std::cout << std::endl;
    El::Print(w_gradients, "w_grad");
    std::cout << std::endl;
    El::Print(Fgrad, "Fgrad");
    std::cout << std::endl;
  }

  // Dump L2 norm of matrices
  if (print_matrix_summary) {
    const auto& dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
    const auto& w_values = dtw->get_values();
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << kfac::get_matrix_stat(
             (const El::Matrix<DataType, Device>&)w_values.LockedMatrix(),
             "W")
        << ", " << kfac::get_matrix_stat(Ainv, "Ainv") << ", "
        << kfac::get_matrix_stat(Ginv, "Ginv") << ", "
        << kfac::get_matrix_stat(w_gradients, "grad") << ", "
        << kfac::get_matrix_stat(Fgrad, "Finvgrad") << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::start_communication_forward_end(
  lbann_comm* comm)
{
  int num_local_activations = 1;
  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& parent_activations = dtl_parent.get_activations();

  if (comm->get_KFAC_subgrid_create_two_models() or
      comm->get_grid_type() == GridType::NO_GRID) {
    this->m_parent_local_activations.resize(num_local_activations);
  }
  else {
    if (this->m_parent_local_activations.size() == 0) {
      // Resize vectors
      this->m_parent_local_activations.resize(num_local_activations);

      // Initialize Dist Matrices
      for (auto& input : this->m_parent_local_activations) {
        if (dtl_parent.get_data_layout() == data_layout::DATA_PARALLEL) {
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
          if (this->m_enable_copy_activations)
            this->m_activations_copy.push_back(
              make_unique<
                El::
                  DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
                parent_activations.Grid(),
                0));
        }
        else {
          input = make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            LBANN_ERROR("Async prgoress is not supported for model-parallel "
                        "layer layout in sub-grid parallelism");
        }
      }
    }

    std::vector<El::mpi::Request<DataType>> requests;

    if (comm->enable_subgrid_async_communication() == false) {
      El::Copy(parent_activations, *this->m_parent_local_activations[0]);
    }
    else {
      El::DistMatrixReadProxy<DataType,
                              DataType,
                              El::STAR,
                              El::VC,
                              El::ELEMENT,
                              Device>
        star_vc_prox(parent_activations);
      El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device> const&
        star_vc_mat = star_vc_prox.GetLocked();

      auto local_activations0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_parent_local_activations[0]));
      // BVE FIXME unused variable
      // auto subset0 = dynamic_cast<El::DistMatrix<DataType, El::STAR, El::VC,
      // El::ELEMENT, Device>*>(&(*this->m_subset_matrix[0]));

      if (this->m_enable_copy_activations) {
        // create a copy of forward activations
        auto copied_acts = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
          &(*this->m_activations_copy[0]));
        El::Copy(star_vc_mat, *copied_acts);
        kfac::TranslateBetweenGridsVCAsyncDirect(*copied_acts,
                                                 *local_activations0,
                                                 this->get_input_size(),
                                                 this->get_current_batch_size(),
                                                 this->m_requests_forward_end);
      }
      else {
        // Directly transfer forward activations (must end before the begining
        // of the next iteration)
        kfac::TranslateBetweenGridsVCAsyncDirect(star_vc_mat,
                                                 *local_activations0,
                                                 this->get_input_size(),
                                                 this->get_current_batch_size(),
                                                 this->m_requests_forward_end);
      }

    } // Async progress
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
    El::LockedView(*(this->m_parent_local_activations[0]), parent_activations);
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::end_communication_forward_end(lbann_comm* comm)
{
  if ((comm->get_grid_type() == GridType::SECONDARY_GRID or
       comm->get_grid_type() == GridType::PRIMARY_GRID) and
      comm->enable_subgrid_async_communication() and
      comm->get_KFAC_subgrid_create_two_models() == false) {
    auto primary_grid_ranks = comm->get_primary_grid_ranks();
    auto secondary_grid_ranks = comm->get_secondary_grid_ranks();

    for (auto& req : this->m_requests_forward_end) {
      ::Al::Wait<kfac::BackendT>(req);
    }

    if (primary_grid_ranks.size() < secondary_grid_ranks.size() and false) {
      auto local_activations0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_parent_local_activations[0]));
      auto subset0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[0]));
      kfac::TranslateBetweenGridsVC(*subset0, *local_activations0);
    }
    this->m_requests_forward_end.clear();
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::start_communication_backward_end(
  lbann_comm* comm)
{
  int num_local_errors = 1;
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_child =
    dynamic_cast<const data_type_layer<DataType>&>(*child);
  const auto& local_errors = dtl_child.get_error_signals();

  if (comm->get_KFAC_subgrid_create_two_models() or
      comm->get_grid_type() == GridType::NO_GRID) {
    this->m_child_local_errors.resize(num_local_errors);
  }
  else {
    if (this->m_child_local_errors.size() == 0) {
      // Resize vectors
      this->m_child_local_errors.resize(num_local_errors);

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
          if (this->m_enable_copy_errors)
            this->m_errors_copy.push_back(
              make_unique<
                El::
                  DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
                local_errors.Grid(),
                0));
        }
        else {
          error = make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (this->m_enable_copy_errors)
            this->m_errors_copy.push_back(
              make_unique<
                El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
                local_errors.Grid(),
                0));
          if (comm->enable_subgrid_async_communication())
            LBANN_ERROR("Async prgoress is not supported for model-parallel "
                        "layer layout in sub-grid parallelism");
        }
      }
    }

    if (comm->enable_subgrid_async_communication() == false) {
      El::Copy(local_errors, *this->m_child_local_errors[0]);
    }
    else {
      const auto local_errors_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(local_errors));
      auto local_errors0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_child_local_errors[0]));
      // auto subset1 = dynamic_cast<El::DistMatrix<DataType, El::STAR, El::VC,
      // El::ELEMENT, Device>*>(&(*this->m_subset_matrix[1]));

      if (this->m_enable_copy_errors) {
        auto copied_errors = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
          &(*this->m_errors_copy[0]));
        El::Copy(*local_errors_vc, *copied_errors);
        kfac::TranslateBetweenGridsVCAsyncDirect(*copied_errors,
                                                 *local_errors0,
                                                 this->get_output_size(),
                                                 this->get_current_batch_size(),
                                                 this->m_requests_backward_end);
      }
      else {
        kfac::TranslateBetweenGridsVCAsyncDirect(*local_errors_vc,
                                                 *local_errors0,
                                                 this->get_output_size(),
                                                 this->get_current_batch_size(),
                                                 this->m_requests_backward_end);
      }

      // kfac::TranslateBetweenGridsVCAsync(*local_errors_vc,
      //                                         *local_errors0,
      //                                         *subset1,
      //                                         this->m_requests_backward_end);
      // kfac::TranslateBetweenGridsVCAsyncDirect(*local_errors_vc,
      //                                         *local_errors0,
      //                                         this->m_requests_backward_end);
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
    El::LockedView(*(this->m_child_local_errors[0]), local_errors);
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::end_communication_backward_end(
  lbann_comm* comm)
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

    if (primary_grid_ranks.size() < secondary_grid_ranks.size() and false) {
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

template <El::Device Device>
void kfac_block_fc_conv<Device>::initialize_activations_and_errors(
  lbann_comm* comm,
  int num_local_activations,
  int num_local_errors,
  int num_weights)
{

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_child =
    dynamic_cast<const data_type_layer<DataType>&>(*child);
  const auto& local_activations = dtl_parent.get_activations();
  const auto& local_errors = dtl_child.get_error_signals();

  if (comm->get_KFAC_subgrid_create_two_models() or
      comm->get_grid_type() == GridType::NO_GRID) {
    this->m_parent_local_activations.resize(num_local_activations);
    this->m_child_local_errors.resize(num_local_errors);
  }
  else {
    if (this->m_parent_local_activations.size() == 0) {
      // Resize vectors
      this->m_parent_local_activations.resize(num_local_activations);
      this->m_child_local_errors.resize(num_local_errors);

      // Initialize Dist Matrices
      for (auto& input : this->m_parent_local_activations) {
        if (dtl_parent.get_data_layout() == data_layout::DATA_PARALLEL) {
          input = std::make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          this->m_subset_matrix.push_back(
            std::make_unique<
              El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
              comm->get_subset_grid(),
              0));
        }
        else
          input = std::make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
      }

      for (auto& error : this->m_child_local_errors) {

        if (dtl_child.get_data_layout() == data_layout::DATA_PARALLEL) {
          error = std::make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          this->m_subset_matrix.push_back(
            std::make_unique<
              El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
              comm->get_subset_grid(),
              0));
        }
        else
          error = std::make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
      }
    }

    int async_progress = true;

    std::vector<kfac::BackendT::req_type> Requests, requests_subset;

    if (async_progress == false) {
      El::Copy(local_activations, *this->m_parent_local_activations[0]);
      El::Copy(local_errors, *this->m_child_local_errors[0]);
    }
    else {
      const auto local_activations_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(local_activations));
      const auto local_errors_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(local_errors));
      auto local_activations0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_parent_local_activations[0]));
      auto local_errors0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_child_local_errors[0]));
      auto subset0 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[0]));
      auto subset1 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[1]));

      // kfac::TranslateBetweenGridsVCAsync(*local_activations_vc,
      //                                         *local_activations0,
      //                                         *subset0,
      //                                         Requests);
      kfac::TranslateBetweenGridsVCAsyncDirect(*local_activations_vc,
                                               *local_activations0,
                                               this->get_input_size(),
                                               this->get_current_batch_size(),
                                               Requests);
      for (auto& req : Requests) {
        ::Al::Wait<kfac::BackendT>(req);
      }
      Requests.clear();
      // kfac::TranslateBetweenGridsVCAsync(*local_errors_vc,
      //                                         *local_errors0,
      //                                         *subset1,
      //                                         Requests);
      kfac::TranslateBetweenGridsVCAsyncDirect(*local_errors_vc,
                                               *local_errors0,
                                               this->get_output_size(),
                                               this->get_current_batch_size(),
                                               Requests);
      auto primary_grid_ranks = comm->get_primary_grid_ranks();
      auto secondary_grid_ranks = comm->get_secondary_grid_ranks();

      for (auto& req : Requests) {
        ::Al::Wait<kfac::BackendT>(req);
      }

      if (async_progress and
          primary_grid_ranks.size() < secondary_grid_ranks.size()) {
        kfac::TranslateBetweenGridsVC(*subset0, *local_activations0);
        kfac::TranslateBetweenGridsVC(*subset1, *local_errors0);
      }
    } // Async progress
  }

  if (comm->get_grid_type() == GridType::NO_GRID) {

    for (auto& input : this->m_parent_local_activations) {
      if (dtl_parent.get_data_layout() == data_layout::DATA_PARALLEL)
        input = std::make_unique<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
      else
        input = std::make_unique<
          El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
    }

    for (auto& error : this->m_child_local_errors) {
      if (dtl_child.get_data_layout() == data_layout::DATA_PARALLEL)
        error = std::make_unique<
          El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
      else
        error = std::make_unique<
          El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
          comm->get_trainer_grid(),
          0);
    }

    El::LockedView(*(this->m_parent_local_activations[0]), local_activations);
    El::LockedView(*(this->m_child_local_errors[0]), local_errors);
  }
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_fc_conv<Device>::get_preconditioned_grad_buffers()
{
  auto& weights = this->m_layer->get_weights(0);
  optimizer* w_optimizer = weights.get_optimizer();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
           gradient_scale = El::TypeTraits<DataType>::One();
  // grad_buffer is already synchronized among processes,
  // and won't be all-reduced later.
  auto& grad_buffer =
    w_optimizer->get_gradient_buffer(dst_scale, gradient_scale, false);
  if (m_is_conv) {
    std::vector<El::AbstractMatrix<DataType>*> ret = {&grad_buffer.Matrix()};
    return ret;
  }
  else {
    // Returns the vectorized version of the matrix.
    auto& mat = grad_buffer.Matrix();
    if (mat.Buffer() != m_grad_buffer_v.Buffer())
      m_grad_buffer_v.Attach(mat.Height() * mat.Width(),
                             1,
                             mat.Buffer(),
                             mat.Height() * mat.Width());
    std::vector<El::AbstractMatrix<DataType>*> ret = {&m_grad_buffer_v};
    return ret;
  }
}

//////////////////////////////////////////////////////////////
// Communication functions
//////////////////////////////////////////////////////////////

template <El::Device Device>
int kfac_block_fc_conv<Device>::get_inverse_matrices(
  El::Matrix<DataType, Device>& output,
  int offset)
{
  const int size_Ainv =
    m_kronecker_inverse_A.Height() * m_kronecker_inverse_A.Width();
  const int size_Ginv =
    m_kronecker_inverse_G.Height() * m_kronecker_inverse_G.Width();

  El::SyncInfo<Device> sync_info = El::SyncInfoFromMatrix(output);

  {
    auto view = El::View(output, El::IR(offset, offset + size_Ainv), El::ALL);
    // El::Copy(m_kronecker_inverse_A, view);
    El::copy::util::InterleaveMatrix(size_Ainv,
                                     1,
                                     m_kronecker_inverse_A.LockedBuffer(),
                                     1,
                                     size_Ainv,
                                     output.Buffer(offset, 0),
                                     1,
                                     size_Ainv,
                                     sync_info);
  }

  offset += size_Ainv;

  {
    auto view = El::View(output, El::IR(offset, offset + size_Ginv), El::ALL);
    // El::Copy(m_kronecker_inverse_G, view);
    El::copy::util::InterleaveMatrix(size_Ginv,
                                     1,
                                     m_kronecker_inverse_G.LockedBuffer(),
                                     1,
                                     size_Ginv,
                                     output.Buffer(offset, 0),
                                     1,
                                     size_Ginv,
                                     sync_info);
  }
  return offset + size_Ginv;
}

template <El::Device Device>
std::vector<int>
kfac_block_fc_conv<Device>::get_inverse_matrices_size_vector(lbann_comm* comm)
{
  std::vector<int> inverse_matrices_sizes;
  inverse_matrices_sizes.push_back(m_kronecker_inverse_A.Height());
  inverse_matrices_sizes.push_back(m_kronecker_inverse_A.Width());
  inverse_matrices_sizes.push_back(m_kronecker_inverse_G.Height());
  inverse_matrices_sizes.push_back(m_kronecker_inverse_G.Width());
  return inverse_matrices_sizes;
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::resize_inverse_matrices_size(
  El::Matrix<double, El::Device::CPU>& inverse_matrices_size,
  int block_number)
{
  m_kronecker_inverse_A.Resize(inverse_matrices_size(block_number, 0),
                               inverse_matrices_size(block_number, 1));
  m_kronecker_inverse_G.Resize(inverse_matrices_size(block_number, 2),
                               inverse_matrices_size(block_number, 3));

  m_Ainv_height = inverse_matrices_size(block_number, 0);
  m_Ainv_width = inverse_matrices_size(block_number, 1);
  m_Ginv_height = inverse_matrices_size(block_number, 2);
  m_Ginv_width = inverse_matrices_size(block_number, 3);
}

template <El::Device Device>
int kfac_block_fc_conv<Device>::get_inverse_matrices_size(lbann_comm* comm)
{
  if (this->m_Ainv_height > 0) {
    return m_Ainv_height * m_Ainv_width + m_Ginv_height * m_Ginv_width;
  }
  const auto input_dims = this->m_layer->get_input_dims(); // CHW
  const size_t num_input_channels = input_dims[0];

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_child =
    dynamic_cast<const data_type_layer<DataType>&>(*child);

  const auto& local_activations =
    comm->get_grid_type() == GridType::PRIMARY_GRID
      ? dtl_parent.get_local_activations()
      : this->m_parent_local_activations[0]->Matrix();

  const auto& local_errors = comm->get_grid_type() == GridType::PRIMARY_GRID
                               ? dtl_child.get_local_error_signals()
                               : this->m_child_local_errors[0]->Matrix();

  int my_height_A = local_activations.Height();
  const auto output_dims = this->m_layer->get_output_dims(); // KH'W'
  const size_t num_output_channels = output_dims[0];

  if (m_is_conv) {
    const auto conv_dims = get_conv_layer()->get_conv_dims();
    my_height_A = num_input_channels * get_linear_size(conv_dims);
  }
  if (m_has_bias)
    my_height_A++;

  int my_height_G = !m_is_conv ? local_errors.Height() : num_output_channels;

  this->m_Ainv_height = my_height_A;
  this->m_Ainv_width = my_height_A;
  this->m_Ginv_height = my_height_G;
  this->m_Ginv_width = my_height_G;

  return my_height_A * my_height_A + my_height_G * my_height_G;
}

template <El::Device Device>
int kfac_block_fc_conv<Device>::set_inverse_matrices(
  El::Matrix<DataType, Device>& workspace,
  int offset,
  lbann_comm* comm)
{
  this->get_inverse_matrices_size(comm);
  El::SyncInfo<Device> sync_infoA =
    El::SyncInfoFromMatrix(m_kronecker_inverse_A);
  El::SyncInfo<Device> sync_infoG =
    El::SyncInfoFromMatrix(m_kronecker_inverse_G);
  if (m_kronecker_inverse_A.Height() > 0) {
    if ((size_t)m_kronecker_inverse_A.Height() == m_Ainv_height and
        (size_t) m_kronecker_inverse_A.Width() == m_Ainv_width and
        (size_t) m_kronecker_inverse_G.Height() == m_Ginv_height and
        (size_t) m_kronecker_inverse_G.Width() == m_Ginv_width) {
    }
    else {
      LBANN_ERROR("Size mismatch");
    }
  }
  m_kronecker_inverse_A.Resize(m_Ainv_height, m_Ainv_width);
  m_kronecker_inverse_G.Resize(m_Ginv_height, m_Ginv_width);

  const int size_Ainv = m_Ainv_height * m_Ainv_width;
  const int size_Ginv = m_Ginv_height * m_Ginv_width;

  {
    El::copy::util::InterleaveMatrix(size_Ainv,
                                     1,
                                     workspace.LockedBuffer(offset, 0),
                                     1,
                                     size_Ainv,
                                     m_kronecker_inverse_A.Buffer(),
                                     1,
                                     size_Ainv,
                                     sync_infoA);
  }

  offset += size_Ainv;

  {
    El::copy::util::InterleaveMatrix(size_Ginv,
                                     1,
                                     workspace.LockedBuffer(offset, 0),
                                     1,
                                     size_Ginv,
                                     m_kronecker_inverse_G.Buffer(),
                                     1,
                                     size_Ginv,
                                     sync_infoG);
  }
  return offset + size_Ginv;
}

//////////////////////////////////////////////////////////////

template <El::Device Device>
void kfac_block_fc_conv<Device>::get_kronecker_factor_fc(
  El::AbstractMatrix<DataType>& factor,
  const El::AbstractMatrix<DataType>& activations,
  const DataType alpha)
{
  assert(activations.GetDevice() == Device);
  assert(factor.Height() == activations.Height());
  assert(factor.Width() == activations.Height());
  El::Gemm(El::NORMAL,
           El::TRANSPOSE,
           alpha,
           activations,
           activations,
           El::TypeTraits<DataType>::Zero(),
           factor);
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::get_kronecker_factor_conv(
  El::Matrix<DataType, Device>& factor,
  El::Matrix<DataType, Device>& Acol,
  const El::Matrix<DataType, Device>& activations,
  const DataType alpha,
  const size_t local_batch_size,
  const size_t num_channels,
  const std::vector<int>& spatial_dims,
  const convolution_layer<DataType, data_layout::DATA_PARALLEL, Device>* l_conv,
  const bool use_im2col,
  const El::SyncInfo<Device>& sync_info)
{
  const auto dilations = l_conv->get_dilations();
  for (auto i = dilations.begin(); i != dilations.end(); i++)
    if (*i != 1) {
      std::stringstream err;
      err << "K-FAC only supports dilation width of 1."
          << " layer: " << l_conv->get_name();
      LBANN_ERROR(err.str());
    }

  if (use_im2col) {
    kfac_fc_conv_util::im2col(activations,
                              Acol,
                              num_channels,
                              spatial_dims.size(),
                              &(spatial_dims[0]),
                              &(l_conv->get_pads()[0]),
                              &(l_conv->get_conv_dims()[0]),
                              &(l_conv->get_strides()[0]),
                              local_batch_size,
                              sync_info);
  }
  else {
    size_t spatial_prod = 1;
    for (auto i = spatial_dims.begin(); i != spatial_dims.end(); i++)
      spatial_prod *= *i;
    assert((size_t)Acol.Height() == num_channels);
    assert((size_t)Acol.Width() == local_batch_size * spatial_prod);
    kfac_fc_conv_util::conv_transpose(activations,
                                      Acol,
                                      local_batch_size,
                                      num_channels,
                                      spatial_prod,
                                      sync_info);
  }

  assert(factor.Height() == Acol.Height());
  assert(factor.Width() == Acol.Height());
  El::Gemm(El::NORMAL,
           El::TRANSPOSE,
           alpha,
           Acol,
           Acol,
           El::TypeTraits<DataType>::Zero(),
           factor);
}

template <El::Device Device>
double
kfac_block_fc_conv<Device>::compute_pi(const El::Matrix<DataType, Device>& A,
                                       const El::Matrix<DataType, Device>& G,
                                       El::Matrix<DataType, Device>& ws,
                                       const El::SyncInfo<Device>& sync_info)
{
  assert(ws.Height() >= A.Height() * 2 + 1);
  assert(ws.Height() >= G.Height() * 2 + 1);
  // TODO: Replace with El::Trace once GPU matrices get supported.
  const auto get_trace = [](const El::Matrix<DataType, Device>& X,
                            El::Matrix<DataType, Device>& w,
                            const El::SyncInfo<Device>& s) {
    auto diag = El::View(w, El::IR(0, X.Height()), El::ALL);
    auto ones = El::View(w, El::IR(X.Height(), X.Height() * 2), El::ALL);
    auto ret = El::View(w, El::IR(X.Height() * 2, X.Height() * 2 + 1), El::ALL);
    kfac_fc_conv_util::get_diagonal(diag, X, s);
    El::Ones(ones, ones.Height(), ones.Width());
    El::Gemm(El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<DataType>::One(),
             diag,
             ones,
             El::TypeTraits<DataType>::Zero(),
             ret);
    El::Matrix<DataType> pi;
    El::Copy(ret, pi);
    return pi(0, 0);
  };
  return sqrt((get_trace(A, ws, sync_info) / A.Height()) /
              (get_trace(G, ws, sync_info) / G.Height()));
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_fc_conv<Device>::get_internal_matrix_info() const
{
  std::vector<std::tuple<std::string, size_t, size_t>> list;
  const auto emplace = [&list](const std::string name,
                               const El::Matrix<DataType, Device>& m) {
    list.emplace_back(name, m.Height(), m.Width());
  };
  emplace("buf_A", m_kronecker_factor_buf_A);
  emplace("buf_G", m_kronecker_factor_buf_G);
  emplace("average_A", m_kronecker_average_A);
  emplace("average_G", m_kronecker_average_G);
  emplace("inverse_A", m_kronecker_inverse_A);
  emplace("inverse_G", m_kronecker_inverse_G);
  emplace("grad_buffer_v", m_grad_buffer_v);
  return list;
}

namespace kfac_fc_conv_util {

template <>
void get_diagonal(El::Matrix<DataType, El::Device::CPU>& diag,
                  const El::Matrix<DataType, El::Device::CPU>& A,
                  const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const size_t height = A.Height();
#pragma omp parallel for
  for (size_t gid = 0; gid < height; gid++)
    diag.Buffer()[gid] = A.LockedBuffer()[gid + gid * height];
}

template <>
void conv_transpose(const El::Matrix<DataType, El::Device::CPU>& activations,
                    El::Matrix<DataType, El::Device::CPU>& act_columns,
                    size_t mini_batch_size,
                    size_t num_channels,
                    size_t spatial_prod,
                    const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const size_t num_elems = mini_batch_size * num_channels * spatial_prod;
#pragma omp parallel for
  for (size_t gid = 0; gid < num_elems; gid++) {
    const auto i_spatial = gid % spatial_prod;
    const auto i_c = (gid / spatial_prod) % num_channels;
    const auto i_n = (gid / spatial_prod / num_channels);
    act_columns.Buffer()[i_c + i_spatial * num_channels +
                         i_n * num_channels * spatial_prod] =
      activations.LockedBuffer()[gid];
  }
}

template <>
void im2col(const El::Matrix<DataType, El::Device::CPU>& im,
            El::Matrix<DataType, El::Device::CPU>& col,
            const int num_channels,
            const int im_num_dims,
            const int* im_dims,
            const int* im_pads,
            const int* window_dims,
            const int* window_strides,
            const int batch_size,
            const El::SyncInfo<El::Device::CPU>& sync_info)
{
  // Since CPU im2col does not support batched im2col explicitly,
  // treat multiple samples as different channels of a single sample.
  El::Matrix<DataType, El::Device::CPU> im_flatten;
  im_flatten.LockedAttach(im.Height() * im.Width(),
                          1,
                          im.LockedBuffer(),
                          im.Height() * im.Width());
  El::Matrix<DataType, El::Device::CPU> col_flatten;
  col_flatten.Attach(col.Height() * batch_size,
                     col.Width() / batch_size,
                     col.Buffer(),
                     col.Height() * batch_size);
  lbann::im2col(im_flatten,
                col_flatten,
                num_channels * batch_size,
                im_num_dims,
                im_dims,
                im_pads,
                window_dims,
                window_strides);
}

#ifdef LBANN_HAS_GPU
template <>
void im2col(const El::Matrix<DataType, El::Device::GPU>& im,
            El::Matrix<DataType, El::Device::GPU>& col,
            const int num_channels,
            const int im_num_dims,
            const int* im_dims,
            const int* im_pads,
            const int* window_dims,
            const int* window_strides,
            const int batch_size,
            const El::SyncInfo<El::Device::GPU>& sync_info)
{
  // Since GPU im2col supports batched im2col, the batch_size argument is
  // ignored.
  lbann::im2col(im,
                col,
                num_channels,
                im_num_dims,
                im_dims,
                im_pads,
                window_dims,
                window_strides,
                sync_info);
}
#endif // LBANN_HAS_GPU

} // namespace kfac_fc_conv_util

template class kfac_block_fc_conv<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac_block_fc_conv<El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
