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
#include "lbann/callbacks/kfac/kfac_util.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_GPU

template <El::Device Device>
void kfac_block_fc_conv<Device>::compute_local_kronecker_factors(
    lbann_comm* comm,
    const bool print_matrix,
    const bool print_matrix_summary) {

  const auto stream = this->get_stream();

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
  const El::AbstractMatrix<DataType>& local_activations = dtl_parent.get_local_activations();
  const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
  const auto mini_batch_size = dtl_parent.get_activations().Width();
  assert(mini_batch_size == dtl_child.get_error_signals().Width());
  const auto local_batch_size = local_activations.Width();

  // Compute Kronecker factors, assuming that local_errors are
  // already multiplied by 1/N in the loss layer.
  const auto input_dims = this->m_layer->get_input_dims(); // CHW
  const auto output_dims = this->m_layer->get_output_dims(); // KH'W'
  const size_t num_input_channels = input_dims[0];
  const size_t num_output_channels = output_dims[0];
  m_height_A = local_activations.Height();
  if(m_is_conv) {
    const auto conv_dims = get_conv_layer()->get_conv_dims();
    m_height_A = num_input_channels
        *std::accumulate(conv_dims.begin(), conv_dims.end(),
                         1, std::multiplies<int>());
  }
  if(m_has_bias)
    m_height_A++;

  m_height_G = !m_is_conv ? local_errors.Height() : num_output_channels;
  auto& A = this->get_workspace_matrix("A", m_height_A, m_height_A);
  auto& G = this->get_workspace_matrix("G", m_height_G, m_height_G);
  if(!m_is_conv) {
    if(m_has_bias) {
      auto& local_activations_with_ones = this->get_workspace_matrix(
          "local_activations_with_ones",
          local_activations.Height()+1, local_activations.Width());
      auto local_activations_dst = El::View(
          local_activations_with_ones,
          El::IR(0, local_activations.Height()),
          El::ALL);
      El::Copy(local_activations, local_activations_dst);
      auto local_activations_ones = El::View(
          local_activations_with_ones,
          El::IR(local_activations.Height(), local_activations.Height()+1),
          El::ALL);
      El::Ones(local_activations_ones, 1, local_activations.Width());
      get_kronecker_factor_fc(A, local_activations_with_ones, 1.0/mini_batch_size);
    } else {
      get_kronecker_factor_fc(A, local_activations, 1.0/mini_batch_size);
    }
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
    auto& Acol = this->get_workspace_matrix(
        "Acol", Acol_size.first, Acol_size.second);
    auto& Gcol = this->get_workspace_matrix(
        "Gcol", num_output_channels, local_batch_size*m_conv_output_spatial_prod);
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

  m_kronecker_factor_buf_A.Resize(A.Height()*(A.Height()+1)/2, 1);
  m_kronecker_factor_buf_G.Resize(G.Height()*(G.Height()+1)/2, 1);
  kfac_util::pack_lower_tri(m_kronecker_factor_buf_A.Buffer(),
                            A.LockedBuffer(), A.Height(), stream);
  kfac_util::pack_lower_tri(m_kronecker_factor_buf_G.Buffer(),
                            G.LockedBuffer(), G.Height(), stream);

  // Dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    // TODO: Show weights' stats
    // const auto &dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
    // const auto &w_values = dtw->get_values();
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        // << kfac_util::get_matrix_stat(w_values.LockedMatrix(), "W")
        << kfac_util::get_matrix_stat(local_activations, "acts")
        << ", " << kfac_util::get_matrix_stat(local_errors, "errs")
        << ", " << kfac_util::get_matrix_stat(A, "A")
        << ", " << kfac_util::get_matrix_stat(G, "G")
        << std::endl;
    std::cout << oss.str();
  }

}

template <El::Device Device>
void kfac_block_fc_conv<Device>::update_kronecker_average(
    lbann_comm* comm,
    const DataType kronecker_decay,
    const bool print_matrix,
    const bool print_matrix_summary) {

  const auto stream = this->get_stream();

  auto& A = this->get_workspace_matrix(
      "A", m_height_A, m_height_A);
  auto& G = this->get_workspace_matrix(
      "G", m_height_G, m_height_G);

  kfac_util::unpack_lower_tri(
      A.Buffer(), m_kronecker_factor_buf_A.LockedBuffer(), m_height_A, stream);
  kfac_util::unpack_lower_tri(
      G.Buffer(), m_kronecker_factor_buf_G.LockedBuffer(), m_height_G, stream);

  // Update average Kronecker factors
  if(!this->m_has_kronecker_inverse) {
    El::Copy(A, m_kronecker_average_A);
    El::Copy(G, m_kronecker_average_G);
  }
  auto &Aave = m_kronecker_average_A;
  auto &Gave = m_kronecker_average_G;
  kfac_util::update_kronecker_average(
      Aave.Buffer(), A.Buffer(), A.Height()*A.Width(), kronecker_decay, stream);
  kfac_util::update_kronecker_average(
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
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat(Aave, "Aave")
        << ", " << kfac_util::get_matrix_stat(Gave, "Gave")
        << std::endl;
    std::cout << oss.str();
  }

}

template <El::Device Device>
void kfac_block_fc_conv<Device>::update_kronecker_inverse(
    lbann_comm* comm,
    const bool use_pi,
    const DataType damping_act, const DataType damping_err,
    const DataType learning_rate_factor,
    const bool print_matrix,
    const bool print_matrix_summary,
    const bool print_time) {

  const auto stream = this->get_stream();

  auto& weights = this->m_layer->get_weights(0);
  optimizer *w_optimizer = weights.get_optimizer();
  auto* w_dto = dynamic_cast<data_type_optimizer<DataType>*>(w_optimizer);

  // TODO: Refactoring
  const auto &Aave = m_kronecker_average_A;
  const auto &Gave = m_kronecker_average_G;
  // Compute the pi constant
  DataType pi = 1.0;
  if(use_pi) {
    auto& ws = this->get_workspace_matrix(
        "pi_ws", std::max(Aave.Height(), Gave.Height())*2+1, 1);
    pi = compute_pi(Aave, Gave, ws, stream);
  }
  // Compute the inverse of the factors
  // Since setting different damping constants for A and G is an
  // alternative heuristics to pi, they should be the same if pi is used.
  if(use_pi && damping_act != damping_err) {
    std::stringstream err;
    err << "Damping values for activations and errors are different while the pi constant is used."
        << " layer: " << this->m_layer->get_name()
        << ", damping_act: " << damping_act
        << ", damping_err: " << damping_err;
    LBANN_WARNING(err.str());
  }

  if(!this->m_has_kronecker_inverse) {
    this->m_has_kronecker_inverse = true;
    m_kronecker_inverse_A.Resize(Aave.Height(), Aave.Width());
    m_kronecker_inverse_G.Resize(Gave.Height(), Gave.Width());
  }
  // TODO: Refactoring
  auto& Ainv = m_kronecker_inverse_A;
  auto& Ginv = m_kronecker_inverse_G;
  auto& ALinv = this->get_workspace_matrix(
      "ALinv", Aave.Height(), Aave.Height());
  auto& GLinv = this->get_workspace_matrix(
      "GLinv", Gave.Height(), Gave.Height());
  kfac_util::get_matrix_inverse(
      Ainv, ALinv, Aave, comm->am_trainer_master() && print_time,
      DataType(damping_act*pi), 0,
      false, stream);
  kfac_util::get_matrix_inverse(
      Ginv, GLinv, Gave, comm->am_trainer_master() && print_time,
      DataType(damping_err/pi), 0,
      false, stream);

  if(print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: pi=" << pi << " @ "<< this->m_layer->get_name() << std::endl;
    std::cout << oss.str();
  }

  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  // grad_buffer is already synchronized among processes,
  // and won't be all-reduced later.
  auto& grad_buffer = w_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);

  const auto& w_grads_orig = w_dto->get_gradient().LockedMatrix();
  El::Matrix<DataType, Device> w_gradients;
  // w_gradients is already synchronized among processes.
  if(m_is_conv) {
    const auto num_output_channels = this->m_layer->get_output_dims()[0];
    assert((w_grads_orig.Height()%num_output_channels) == 0);
    const auto height = w_grads_orig.Height()/num_output_channels;
    w_gradients.LockedAttach(height, num_output_channels,
                             w_grads_orig.LockedBuffer(),
                             height);
  } else {
    if(m_has_bias) {
      auto& w_grads_concat = this->get_workspace_matrix(
          "A", w_grads_orig.Height(), w_grads_orig.Width()+1);

      auto w_grads_concat_weights = El::View(
          w_grads_concat, El::ALL, El::IR(0, w_grads_orig.Width()));
      auto w_grads_concat_biases = El::View(
          w_grads_concat, El::ALL, El::IR(w_grads_orig.Width(), w_grads_orig.Width()+1));

      auto& biases = this->m_layer->get_weights(1);
      optimizer *b_optimizer = biases.get_optimizer();
      auto* b_dto = dynamic_cast<data_type_optimizer<DataType>*>(b_optimizer);
      const auto& b_grads_orig = b_dto->get_gradient().LockedMatrix();

      El::Copy(w_grads_orig, w_grads_concat_weights);
      El::Copy(b_grads_orig, w_grads_concat_biases);

      w_gradients.LockedAttach(w_grads_concat.Height(), w_grads_concat.Width(),
                               w_grads_concat.LockedBuffer(),
                               w_grads_concat.Height());
    } else {
      w_gradients.LockedAttach(w_grads_orig.Height(), w_grads_orig.Width(),
                               w_grads_orig.LockedBuffer(),
                               w_grads_orig.Height());
    }
  }

  // Compute preconditioned gradients
  auto& Gg = this->get_workspace_matrix(
      "Gg",
      Ginv.Height(),
      m_is_conv ? w_gradients.Height() : w_gradients.Width());

  El::Gemm(
      El::NORMAL, m_is_conv ? El::TRANSPOSE : El::NORMAL,
      El::TypeTraits<DataType>::One(), Ginv, w_gradients,
      El::TypeTraits<DataType>::Zero(), Gg);
  auto& Fgrad = this->get_workspace_matrix(
      "Fgrad", Ginv.Height(), Ainv.Width());
  El::Gemm(
      El::NORMAL, El::NORMAL,
      learning_rate_factor, Gg, Ainv,
      El::TypeTraits<DataType>::Zero(), Fgrad);

  if(m_is_conv) {
    El::Matrix<DataType, Device> Fgrad_v;
    Fgrad_v.LockedAttach(Fgrad.Width()*Fgrad.Height(), 1,
                         Fgrad.LockedBuffer(),
                         Fgrad.Width()*Fgrad.Height());
    El::Copy(Fgrad_v, grad_buffer.Matrix());
  } else {
    assert(Fgrad.Height() == w_gradients.Height());
    assert(Fgrad.Width() == w_gradients.Width());

    if(m_has_bias) {
      auto& biases = this->m_layer->get_weights(1);
      optimizer *b_optimizer = biases.get_optimizer();
      auto& grad_buffer_biases = b_optimizer->get_gradient_buffer(
          dst_scale, gradient_scale, false);
      auto Fgrad_weights = El::View(
          Fgrad, El::ALL, El::IR(0, grad_buffer.Width()));
      auto Fgrad_biases = El::View(
          Fgrad, El::ALL, El::IR(grad_buffer.Width(), grad_buffer.Width()+1));
      El::Copy(Fgrad_weights, grad_buffer.Matrix());
      El::Copy(Fgrad_biases, grad_buffer_biases.Matrix());

    } else {
      El::Copy(Fgrad, grad_buffer.Matrix());
    }
  }

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
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat(w_values.LockedMatrix(), "W")
        << ", " << kfac_util::get_matrix_stat(Ainv, "Ainv")
        << ", " << kfac_util::get_matrix_stat(Ginv, "Ginv")
        << ", " << kfac_util::get_matrix_stat(w_gradients, "grad")
        << ", " << kfac_util::get_matrix_stat(Fgrad, "Finvgrad")
        << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_fc_conv<Device>::get_preconditioned_grad_buffers() {
  auto& weights = this->m_layer->get_weights(0);
  optimizer *w_optimizer = weights.get_optimizer();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  // grad_buffer is already synchronized among processes,
  // and won't be all-reduced later.
  auto& grad_buffer = w_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, false);
  if(m_is_conv) {
    std::vector<El::AbstractMatrix<DataType>*>
        ret = {&grad_buffer.Matrix()};
    return ret;
  } else {
    // Returns the vectorized version of the matrix.
    auto& mat = grad_buffer.Matrix();
    if(mat.Buffer() != m_grad_buffer_v.Buffer())
      m_grad_buffer_v.Attach(
          mat.Height()*mat.Width(), 1,
          mat.Buffer(),
          mat.Height()*mat.Width());
    std::vector<El::AbstractMatrix<DataType>*>
        ret = {&m_grad_buffer_v};
    return ret;
  }
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::get_kronecker_factor_fc(
    El::AbstractMatrix<DataType>& factor,
    const El::AbstractMatrix<DataType>& activations,
    const DataType alpha) {
  assert(activations.GetDevice() == Device);
  assert(factor.Height() == activations.Height());
  assert(factor.Width() == activations.Height());
  El::Gemm(
      El::NORMAL, El::TRANSPOSE,
      alpha, activations, activations,
      El::TypeTraits<DataType>::Zero(),
      factor);
}

template <El::Device Device>
void kfac_block_fc_conv<Device>::get_kronecker_factor_conv(
    El::Matrix<DataType, Device>& factor,
    El::Matrix<DataType, Device>& Acol,
    const El::Matrix<DataType, Device>& activations,
    const DataType alpha,
    const size_t local_batch_size, const size_t num_channels,
    const std::vector<int>& spatial_dims,
    const convolution_layer<DataType, data_layout::DATA_PARALLEL, Device> *l_conv,
    const bool use_im2col,
    const cudaStream_t& stream) {
  assert(factor.GetDevice() == Device);
  assert(activations.GetDevice() == Device);

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
    kfac_fc_conv_util::conv_transpose(
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

template <El::Device Device>
double kfac_block_fc_conv<Device>::compute_pi(
    const El::Matrix<DataType, Device>& A,
    const El::Matrix<DataType, Device>& G,
    El::Matrix<DataType, Device>& ws,
    const cudaStream_t& stream) {
  assert(ws.Height() >= A.Height()*2+1);
  assert(ws.Height() >= G.Height()*2+1);
  // TODO: Replace with El::Trace once GPU matrices get supported.
  const auto get_trace =
      [](const El::Matrix<DataType, Device>& X,
         El::Matrix<DataType, Device>& w,
         const cudaStream_t& s) {
        auto diag = El::View(w, El::IR(0, X.Height()), El::ALL);
        auto ones = El::View(w, El::IR(X.Height(), X.Height()*2), El::ALL);
        auto ret = El::View(w, El::IR(X.Height()*2, X.Height()*2+1), El::ALL);
        kfac_fc_conv_util::get_diagonal(diag.Buffer(), X.LockedBuffer(), X.Height(), s);
        El::Ones(ones, ones.Height(), ones.Width());
        El::Gemm(
            El::TRANSPOSE, El::NORMAL,
            El::TypeTraits<DataType>::One(), diag, ones,
            El::TypeTraits<DataType>::Zero(), ret);
        El::Matrix<DataType> pi;
        El::Copy(ret, pi);
        return pi(0, 0);
      };
  return sqrt((get_trace(A, ws, stream)/A.Height())/(get_trace(G, ws, stream)/G.Height()));
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_fc_conv<Device>::get_internal_matrix_info() const {
  std::vector<std::tuple<std::string, size_t, size_t>> list;
  const auto emplace =
      [&list](const std::string name,
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

template class kfac_block_fc_conv<El::Device::GPU>;

#endif // LBANN_HAS_GPU

} // namespace callback
} // namespace lbann
