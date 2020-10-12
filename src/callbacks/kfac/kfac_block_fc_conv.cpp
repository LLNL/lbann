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

#include "lbann/callbacks/kfac/kfac_block_fc_conv.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {
namespace callback {

void kfac_block_fc_conv::update_kronecker_factors(
    lbann_comm* comm,
    const DataType kronecker_decay,
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

  // Compute Kronecker factors, assuming that local_errors are
  // already multiplied by 1/N in the loss layer.
  const auto input_dims = m_layer->get_input_dims(); // CHW
  const auto output_dims = m_layer->get_output_dims(); // KH'W'
  const size_t num_input_channels = input_dims[0];
  const size_t num_output_channels = output_dims[0];
  size_t A_height = local_activations.Height();
  if(m_is_conv) {
    const auto conv_dims = get_conv_layer()->get_conv_dims();
    A_height = num_input_channels
        *std::accumulate(conv_dims.begin(), conv_dims.end(),
                         1, std::multiplies<int>());
  }
  const size_t G_height = !m_is_conv ? local_errors.Height() : num_output_channels;
  auto& A = m_callback->get_workspace_matrix(
      std::string("A_")+std::to_string(m_layer_id),
      A_height, A_height);
  auto& G = m_callback->get_workspace_matrix(
      std::string("G_")+std::to_string(m_layer_id),
      G_height, G_height);
  if(!m_is_conv) {
    get_kronecker_factor_fc(A, local_activations, 1.0/mini_batch_size);
    get_kronecker_factor_fc(G, local_errors, mini_batch_size);
  } else {
    assert((size_t) local_activations.Height() == num_input_channels*m_conv_input_spatial_prod);
    assert((size_t) local_errors.Height() == num_output_channels*m_conv_output_spatial_prod);

    const auto Acol_size = get_im2col_output_size(
        local_batch_size,
        num_input_channels, m_conv_input_spatial_dims.size(),
        &(m_conv_input_spatial_dims[0]),
        &(get_conv_layer()->get_pads()[0]),
        &(get_conv_layer()->get_conv_dims()[0]),
        &(get_conv_layer()->get_strides()[0]));
    auto& Acol = m_callback->get_workspace_matrix(
        std::string("Acol"), // im2col workspace is reused as it is huge.
        Acol_size.first, Acol_size.second);
    auto& Gcol = m_callback->get_workspace_matrix(
        std::string("Gcol_")+std::to_string(m_layer_id),
        num_output_channels, local_batch_size*m_conv_output_spatial_prod);
    get_kronecker_factor_conv(
        A, Acol,
        local_activations, 1.0/mini_batch_size,
        local_batch_size, num_input_channels, m_conv_input_spatial_dims,
        get_conv_layer(), true, stream);
    get_kronecker_factor_conv(
        G, Gcol,
        local_errors, DataType(mini_batch_size)/m_conv_output_spatial_prod,
        local_batch_size, num_output_channels, m_conv_output_spatial_dims,
        get_conv_layer(), false, stream);
  }

  // Accumulate local Kronecker factors
  auto& ALws = m_callback->get_workspace_matrix(std::string("ALws_")+std::to_string(m_layer_id),
                                                A.Height()*(A.Height()+1)/2, 1);
  auto& GLws = m_callback->get_workspace_matrix(std::string("GLws_")+std::to_string(m_layer_id),
                                                G.Height()*(G.Height()+1)/2, 1);
  kfac::allreduce_lower_tri(A, ALws, comm, stream);
  kfac::allreduce_lower_tri(G, GLws, comm, stream);

  // Update average Kronecker factors
  if(!m_has_kronecker_inverse) {
    El::Copy(A, m_kronecker_average_A);
    El::Copy(G, m_kronecker_average_G);
  }
  auto &Aave = m_kronecker_average_A;
  auto &Gave = m_kronecker_average_G;
  kfac::update_kronecker_average(
      Aave.Buffer(), A.Buffer(), A.Height()*A.Width(), kronecker_decay, stream);
  kfac::update_kronecker_average(
      Gave.Buffer(), G.Buffer(), G.Height()*G.Width(), kronecker_decay, stream);

  // Dump matrices for debugging
  if(comm->am_trainer_master() && print_matrix) {
    if(comm->am_trainer_master()) {
      std::cout << std::endl; El::Print(A, "A");
      std::cout << std::endl; El::Print(G, "G");
      std::cout << std::endl; El::Print(Aave, "Aave");
      std::cout << std::endl; El::Print(Gave, "Gave");
      std::cout << std::endl;
    }
  }

  // Dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    // TODO: Show weights's stats
    // const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
    // const auto &w_values = dtw->get_values();
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< m_layer->get_name() << ": "
        // << kfac::get_matrix_stat(w_values.LockedMatrix(), "W")
        // << ", "
        << kfac::get_matrix_stat(local_activations, "acts")
        << ", " << kfac::get_matrix_stat(local_errors, "errs")
        << ", " << kfac::get_matrix_stat(A, "A")
        << ", " << kfac::get_matrix_stat(G, "G")
        << ", " << kfac::get_matrix_stat(Aave, "Aave")
        << ", " << kfac::get_matrix_stat(Gave, "Gave")
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac_block_fc_conv::update_kronecker_inverse(
    lbann_comm* comm,
    const bool use_pi,
    const DataType damping_act, const DataType damping_err,
    const bool print_matrix,
    const bool print_matrix_summary,
    const bool print_time) {

  const auto stream = get_stream();

  auto& weights = m_layer->get_weights(0);
  optimizer *w_optimizer = weights.get_optimizer();
  auto* w_dto = dynamic_cast<data_type_optimizer<DataType>*>(w_optimizer);

  // TODO: Refactoring
  const auto &Aave = m_kronecker_average_A;
  const auto &Gave = m_kronecker_average_G;
  // Compute the pi constant
  DataType pi = 1.0;
  if(use_pi) {
    auto& ws = m_callback->get_workspace_matrix(
        std::string("pi_ws_")+std::to_string(m_layer_id),
        std::max(Aave.Height(), Gave.Height())*2+1, 1);
    pi = compute_pi(Aave, Gave, ws, stream);
  }
  // Compute the inverse of the factors
  // Since setting different damping constants for A and G is an
  // alternative heuristics to pi, they should be the same if pi is used.
  if(use_pi && damping_act != damping_err) {
    std::stringstream err;
    err << "Damping values for activations and errors are different while the pi constant is used."
        << " layer: " << m_layer->get_name()
        << ", damping_act: " << damping_act
        << ", damping_err: " << damping_err;
    LBANN_WARNING(err.str());
  }

  if(!m_has_kronecker_inverse) {
    m_has_kronecker_inverse = true;
    m_kronecker_inverse_A.Resize(Aave.Height(), Aave.Width());
    m_kronecker_inverse_G.Resize(Gave.Height(), Gave.Width());
  }
  // TODO: Refactoring
  auto& Ainv = m_kronecker_inverse_A;
  auto& Ginv = m_kronecker_inverse_G;
  auto& ALinv = m_callback->get_workspace_matrix(
      std::string("ALinv_")+std::to_string(m_layer_id),
      Aave.Height(), Aave.Height());
  auto& GLinv = m_callback->get_workspace_matrix(
      std::string("GLinv_")+std::to_string(m_layer_id),
      Gave.Height(), Gave.Height());
  kfac::get_matrix_inverse(
      Ainv, ALinv, Aave, comm->am_trainer_master() && print_time,
      DataType(damping_act*pi), 0,
      false, stream);
  kfac::get_matrix_inverse(
      Ginv, GLinv, Gave, comm->am_trainer_master() && print_time,
      DataType(damping_err/pi), 0,
      false, stream);

  if(print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: pi=" << pi << " @ "<< m_layer->get_name() << std::endl;
    std::cout << oss.str();
  }

  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  // grad_buffer is already synchronized among processes,
  // and won't be all-reduced later.
  auto& grad_buffer = w_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);

  const auto& w_grads_orig = w_dto->get_gradient().LockedMatrix();
  El::Matrix<DataType, El::Device::GPU> w_gradients;
  // w_gradients is already synchronized among processes.
  if(m_is_conv) {
    const auto num_output_channels = m_layer->get_output_dims()[0];
    assert((w_grads_orig.Height()%num_output_channels) == 0);
    const auto height = w_grads_orig.Height()/num_output_channels;
    w_gradients.LockedAttach(height, num_output_channels,
                             w_grads_orig.LockedBuffer(),
                             height);
  } else {
    w_gradients.LockedAttach(w_grads_orig.Height(), w_grads_orig.Width(),
                             w_grads_orig.LockedBuffer(),
                             w_grads_orig.Height());
  }

  // Compute preconditioned gradients
  auto& Gg = m_callback->get_workspace_matrix(
      std::string("Gg_")+std::to_string(m_layer_id),
      Ginv.Height(),
      m_is_conv ? w_gradients.Height() : w_gradients.Width());
  El::Gemm(
      El::NORMAL, m_is_conv ? El::TRANSPOSE : El::NORMAL,
      El::TypeTraits<DataType>::One(), Ginv, w_gradients,
      El::TypeTraits<DataType>::Zero(), Gg);
  auto& Fgrad = m_callback->get_workspace_matrix(
      std::string("Fgrad_")+std::to_string(m_layer_id),
      Ginv.Height(), Ainv.Width());
  El::Gemm(
      El::NORMAL, El::NORMAL,
      El::TypeTraits<DataType>::One(), Gg, Ainv,
      El::TypeTraits<DataType>::Zero(), Fgrad);

  if(m_is_conv) {
    El::Matrix<DataType, El::Device::GPU> Fgrad_v;
    Fgrad_v.LockedAttach(Fgrad.Width()*Fgrad.Height(), 1,
                         Fgrad.LockedBuffer(),
                         Fgrad.Width()*Fgrad.Height());
    El::Copy(Fgrad_v, grad_buffer.Matrix());
  } else {
    assert(Fgrad.Height() == w_gradients.Height());
    assert(Fgrad.Width() == w_gradients.Width());
    El::Copy(Fgrad, grad_buffer.Matrix());
  }

  // Apply preconditioned grads

  // Dump matrices for debugging
  if(print_matrix) {
    std::cout << std::endl; El::Print(Ainv, "Ainv");
    std::cout << std::endl; El::Print(Ginv, "Ginv");
    std::cout << std::endl; El::Print(w_gradients, "w_grad");
    std::cout << std::endl; El::Print(Fgrad, "Fgrad");
    std::cout << std::endl;
  }

  // Dump L2 norm of matrices
  if(print_matrix_summary) {
    const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
    const auto &w_values = dtw->get_values();
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< m_layer->get_name() << ": "
        << kfac::get_matrix_stat(w_values.LockedMatrix(), "W")
        << ", " << kfac::get_matrix_stat(Ainv, "Ainv")
        << ", " << kfac::get_matrix_stat(Ginv, "Ginv")
        << ", " << kfac::get_matrix_stat(w_gradients, "grad")
        << ", " << kfac::get_matrix_stat(Fgrad, "Finvgrad")
        << std::endl;
    std::cout << oss.str();
  }
}

void kfac_block_fc_conv::update_preconditioned_grads(
    lbann_comm* comm) {

  auto& weights = m_layer->get_weights(0);
  optimizer *w_optimizer = weights.get_optimizer();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  // grad_buffer is already synchronized among processes,
  // and won't be all-reduced later.
  auto& grad_buffer = w_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);
  El::Broadcast(
      grad_buffer.Matrix(), comm->get_trainer_comm(),
      m_inverse_proc_rank);
}

void kfac_block_fc_conv::get_kronecker_factor_fc(
    El::AbstractMatrix<DataType>& factor,
    const El::AbstractMatrix<DataType>& activations,
    const DataType alpha) {
  assert(activations.GetDevice() == El::Device::GPU);
  assert(factor.Height() == activations.Height());
  assert(factor.Width() == activations.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, activations, activations,
      El::TypeTraits<DataType>::Zero(), factor);
}

void kfac_block_fc_conv::get_kronecker_factor_conv(
    El::Matrix<DataType, El::Device::GPU>& factor,
    El::Matrix<DataType, El::Device::GPU>& Acol,
    const El::Matrix<DataType, El::Device::GPU>& activations,
    const DataType alpha,
    const size_t local_batch_size, const size_t num_channels,
    const std::vector<int> spatial_dims,
    const convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU> *l_conv,
    const bool use_im2col,
    const cudaStream_t& stream) {
  assert(factor.GetDevice() == El::Device::GPU);
  assert(activations.GetDevice() == El::Device::GPU);

  const auto dilations = l_conv->get_dilations();
  for(auto i = dilations.begin(); i != dilations.end(); i++)
    if(*i != 1) {
      std::stringstream err;
      err << "The K-FAC callback onky supports dilation width of 1."
          << " layer: " << l_conv->get_name();
      LBANN_ERROR(err.str());
    }

  if(use_im2col) {
    im2col(activations, Acol,
           num_channels, spatial_dims.size(),
           &(spatial_dims[0]),
           &(l_conv->get_pads()[0]),
           &(l_conv->get_conv_dims()[0]),
           &(l_conv->get_strides()[0]),
           stream);
  } else {
    size_t spatial_prod = 1;
    for(auto i = spatial_dims.begin(); i != spatial_dims.end(); i++)
      spatial_prod *= *i;
    assert((size_t) Acol.Height() == num_channels);
    assert((size_t) Acol.Width() == local_batch_size*spatial_prod);
    conv_transpose(
        activations.LockedBuffer(), Acol.Buffer(),
        local_batch_size, num_channels, spatial_prod,
        stream);
  }

  assert(factor.Height() == Acol.Height());
  assert(factor.Width() == Acol.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, Acol, Acol,
      El::TypeTraits<DataType>::Zero(), factor);
}

double kfac_block_fc_conv::compute_pi(
    const El::Matrix<DataType, El::Device::GPU>& A,
    const El::Matrix<DataType, El::Device::GPU>& G,
    El::Matrix<DataType, El::Device::GPU>& ws,
    const cudaStream_t& stream) {
  assert(ws.Height() >= A.Height()*2+1);
  assert(ws.Height() >= G.Height()*2+1);
  // TODO: Replace with El::Trace once GPU matrices get supported.
  const auto get_trace =
      [](const El::Matrix<DataType, El::Device::GPU>& X,
         El::Matrix<DataType, El::Device::GPU>& w,
         const cudaStream_t& s) {
        auto diag = El::View(w, El::IR(0, X.Height()), El::ALL);
        auto ones = El::View(w, El::IR(X.Height(), X.Height()*2), El::ALL);
        auto ret = El::View(w, El::IR(X.Height()*2, X.Height()*2+1), El::ALL);
        get_diagonal(diag.Buffer(), X.LockedBuffer(), X.Height(), s);
        El::Ones(ones, ones.Height(), ones.Width());
        El::Gemm(
            El::TRANSPOSE, El::NORMAL,
            El::TypeTraits<DataType>::One(), diag, ones,
            El::TypeTraits<DataType>::Zero(), ret);
        return El::Matrix<DataType>(ret)(0, 0);
      };
  return sqrt((get_trace(A, ws, stream)/A.Height())/(get_trace(G, ws, stream)/G.Height()));
}

} // namespace callback
} // namespace lbann
