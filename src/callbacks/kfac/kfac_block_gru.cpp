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

#include "lbann/callbacks/kfac/kfac_block_gru.hpp"
#include "lbann/callbacks/kfac/kfac_util.hpp"

namespace lbann {
namespace callback {

template <>
void kfac_block_gru<El::Device::CPU>::on_forward_prop_end() {
  LBANN_ERROR("The K-FAC callback does not support CPU GRU layers.");
}

template <>
void kfac_block_gru<El::Device::GPU>::on_forward_prop_end() {
  const auto& reserve_space = get_gru_layer()->get_reserve_space();
  if(m_reserve_space_fwd.size() != reserve_space.size())
    m_reserve_space_fwd.allocate(reserve_space.size());
  const auto& sync_info = this->get_sync_info();
  CHECK_CUDA(cudaMemcpyAsync(
      m_reserve_space_fwd.data(),
      reserve_space.data(),
      reserve_space.size(),
      cudaMemcpyDeviceToDevice,
      sync_info.Stream()));
}

template <El::Device Device>
void kfac_block_gru<Device>::compute_local_kronecker_factors(
    lbann_comm* comm,
    const bool print_matrix,
    const bool print_matrix_summary) {
  // input_dims:  seq_length x input_size
  // output_dims: seq_length x hidden_size
  // local_inputs:  (seq_length*input_size)  x local_batch_size
  // local_outputs: (seq_length*hidden_size)  x local_batch_size
  // local_errors:  (seq_length*hidden_size) x local_batch_size

  const auto& sync_info = this->get_sync_info();

  const auto gru = get_gru_layer();
  const auto input_dims = this->m_layer->get_input_dims();
  const size_t input_size = input_dims[1];
  const size_t hidden_size = gru->get_hidden_size();
  const size_t seq_length = input_dims[0];

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto parent_h0 = this->m_layer->get_parent_layers()[1];
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_parent = dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_parent_h0 = dynamic_cast<const data_type_layer<DataType>&>(*parent_h0);
  const auto& dtl_child = dynamic_cast<const data_type_layer<DataType>&>(*child);
  const auto& dtl = dynamic_cast<const data_type_layer<DataType>&>(*this->m_layer);
  const El::AbstractMatrix<DataType>& local_inputs = dtl_parent.get_local_activations();
  const El::AbstractMatrix<DataType>& h0 = dtl_parent_h0.get_local_activations();
  const El::AbstractMatrix<DataType>& local_outputs = dtl.get_local_activations();
  const El::AbstractMatrix<DataType>& local_errors = dtl_child.get_local_error_signals();
  const auto local_batch_size = local_inputs.Width();

  // OPTIMIZE: m_input_size and m_hidden_size?
  auto& A = this->get_workspace_matrix("A", hidden_size, hidden_size);
  auto& G = this->get_workspace_matrix("G", hidden_size, hidden_size);

  // r, i: (hidden_size*local_batch_size) x seq_length
  auto& r = this->get_workspace_matrix("r", hidden_size*local_batch_size, seq_length);
  auto& i = this->get_workspace_matrix("i", hidden_size*local_batch_size, seq_length);
  kfac_gru_util::unpack_reserve_space(
      (const DataType *) m_reserve_space_fwd.data(),
      r, i,
      hidden_size, seq_length, local_batch_size,
      sync_info);

  // hfc_t = R_h h_{t-1} + b_Rh : hidden_size x local_batch_size
  // weights_Rh: hidden_size x hidden_size
  // biases_Rh: hidden_size x 1
  auto& hfc = this->get_workspace_matrix("hfc_t", hidden_size, local_batch_size*seq_length);
  El::Matrix<DataType, Device> weights_Rh, biases_Rh;
  get_weight_matrix(kfac_gru_util::gru_weight_matrix_type::Rh, weights_Rh);
  get_weight_matrix(kfac_gru_util::gru_weight_matrix_type::bRh, biases_Rh);

  // Recompute hfc = R_h h_t + b_Rh
  // OPTIMIZE: Merge the following two loops
  // OPTIMIZE: compute with a single GEMM call
  auto& biases_Rh_ones = this->get_workspace_matrix("b_Rh_ones", 1, local_batch_size);
  for(size_t t = 0; t < seq_length; t++) {
    auto hfc_t = El::View(hfc, El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    const auto h_prev =
        (t == 0
         ? El::LockedView((El::Matrix<DataType, Device>&) h0, El::ALL, El::ALL) // OPTIMIZE
         : El::LockedView((El::Matrix<DataType, Device>&) local_outputs,
                          El::IR(hidden_size*(t-1), hidden_size*t), El::ALL));
    El::Gemm(
        El::NORMAL, El::NORMAL, // weight matrices are in row-major
        El::TypeTraits<DataType>::One(), weights_Rh, h_prev,
        El::TypeTraits<DataType>::Zero(), hfc_t);
    El::Gemm(
        El::NORMAL, El::NORMAL,
        El::TypeTraits<DataType>::One(), biases_Rh, biases_Rh_ones,
        El::TypeTraits<DataType>::One(), hfc_t);
  }

  // OPTIMIZE: Replace with a single kernel call
  auto& g = this->get_workspace_matrix("g", hidden_size, local_batch_size*seq_length);
  for(size_t t = 0; t < seq_length; t++) {
    // hidden_size x local_batch_size
    auto g_t = El::View(g, El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    const auto x_t =
        El::LockedView((El::Matrix<DataType, Device>&) local_inputs,
                       El::IR(input_size*t, input_size*(t+1)), El::ALL);
    const auto h_t =
        El::LockedView((El::Matrix<DataType, Device>&) local_outputs,
                       El::IR(hidden_size*t, hidden_size*(t+1)), El::ALL);
    const auto dh_t =
        El::LockedView((El::Matrix<DataType, Device>&) local_errors,
                       El::IR(hidden_size*t, hidden_size*(t+1)), El::ALL);
    const auto r_t = El::LockedView(r, El::ALL, El::IR(t, t+1));
    const auto i_t = El::LockedView(i, El::ALL, El::IR(t, t+1));
    const auto hfc_t = El::LockedView(hfc, El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    kfac_gru_util::get_g_Rr(h_t, dh_t, hfc_t, r_t, i_t, g_t, hidden_size, sync_info);

    const DataType alpha = 1.0/seq_length;
    const DataType beta = (t == 0 ? El::TypeTraits<DataType>::Zero() : El::TypeTraits<DataType>::One());
    El::Gemm(
        El::NORMAL, El::TRANSPOSE,
        alpha, h_t, h_t,
        beta, A);
    El::Gemm(
        El::NORMAL, El::TRANSPOSE,
        alpha, g_t, g_t,
        beta, G);
  }

  m_kronecker_factor_buf_A.Resize(A.Height()*(A.Height()+1)/2, 1);
  m_kronecker_factor_buf_G.Resize(G.Height()*(G.Height()+1)/2, 1);
  kfac_util::pack_lower_tri(m_kronecker_factor_buf_A, A, sync_info);
  kfac_util::pack_lower_tri(m_kronecker_factor_buf_G, G, sync_info);

  // Dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) local_outputs, "h")
        << ", " << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) g, "g")
        << ", " << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) A, "A")
        << ", " << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) G, "G")
        << std::endl;
    std::cout << oss.str();
  }

  // TODO: remove
  CHECK_CUDA(cudaDeviceSynchronize());
  comm->global_barrier();
}

// TODO: Merge with fc_conv
template <El::Device Device>
void kfac_block_gru<Device>::update_kronecker_average(
    lbann_comm* comm,
    const DataType kronecker_decay,
    const bool print_matrix,
    const bool print_matrix_summary) {
  const auto& sync_info = this->get_sync_info();
  const auto gru = get_gru_layer();
  const size_t hidden_size = gru->get_hidden_size();

  auto& A = this->get_workspace_matrix(
      "A", hidden_size, hidden_size);
  auto& G = this->get_workspace_matrix(
      "G", hidden_size, hidden_size);

  kfac_util::unpack_lower_tri(A, m_kronecker_factor_buf_A, sync_info);
  kfac_util::unpack_lower_tri(G, m_kronecker_factor_buf_G, sync_info);

  // Update average Kronecker factors
  if(!this->m_has_kronecker_inverse) {
    El::Copy(A, m_kronecker_average_A);
    El::Copy(G, m_kronecker_average_G);
  }
  auto &Aave = m_kronecker_average_A;
  auto &Gave = m_kronecker_average_G;
  kfac_util::update_kronecker_average(
      Aave, A, A.Height()*A.Width(), kronecker_decay, sync_info);
  kfac_util::update_kronecker_average(
      Gave, G, G.Height()*G.Width(), kronecker_decay, sync_info);

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
void kfac_block_gru<Device>::update_kronecker_inverse(
    lbann_comm* comm,
    const bool use_pi,
    const DataType damping_act, const DataType damping_err,
    const DataType learning_rate_factor,
    const bool print_matrix,
    const bool print_matrix_summary,
    const bool print_time) {
  const auto& sync_info = this->get_sync_info();

  const auto &Aave = m_kronecker_average_A;
  const auto &Gave = m_kronecker_average_G;

  if(!this->m_has_kronecker_inverse) {
    this->m_has_kronecker_inverse = true;
    m_kronecker_inverse_A.Resize(Aave.Height(), Aave.Width());
    m_kronecker_inverse_G.Resize(Gave.Height(), Gave.Width());
  }
  // TODO: Refactoring
  const DataType pi = 1.0; // TODO: Compute pi if use_pi
  auto& Ainv = m_kronecker_inverse_A;
  auto& Ginv = m_kronecker_inverse_G;
  auto& ALinv = this->get_workspace_matrix(
      "ALinv", Aave.Height(), Aave.Height());
  auto& GLinv = this->get_workspace_matrix(
      "GLinv", Gave.Height(), Gave.Height());
  kfac_util::get_matrix_inverse(
      Ainv, ALinv, Aave, comm->am_trainer_master() && print_time,
      DataType(damping_act*pi), 0,
      false, sync_info);
  kfac_util::get_matrix_inverse(
      Ginv, GLinv, Gave, comm->am_trainer_master() && print_time,
      DataType(damping_err/pi), 0,
      false, sync_info);

  // Compute preconditioned gradients
  El::Matrix<DataType, Device> gradients_Rr;
  get_gradient_matrix(kfac_gru_util::gru_weight_matrix_type::Rr, gradients_Rr);
  auto& Gg = this->get_workspace_matrix(
      "Gg",
      Ginv.Height(),
      gradients_Rr.Width());
  El::Gemm(
      El::NORMAL, El::NORMAL, // gradient matrices are in row-major
      El::TypeTraits<DataType>::One(), Ginv, gradients_Rr,
      El::TypeTraits<DataType>::Zero(), Gg);
  auto& Fgrad = this->get_workspace_matrix(
      "Fgrad", Ginv.Height(), Ainv.Width());
  El::Gemm(
      El::NORMAL, El::NORMAL,
      learning_rate_factor, Gg, Ainv,
      El::TypeTraits<DataType>::Zero(), Fgrad);

  // Dump matrices for debugging
  if(print_matrix) {
    std::cout << std::endl; El::Print(Ainv, "Ainv");
    std::cout << std::endl; El::Print(Ginv, "Ginv");
    std::cout << std::endl; El::Print(gradients_Rr, "grad");
    std::cout << std::endl; El::Print(Fgrad, "Finvgrad");
    std::cout << std::endl;
  }

  // Dump L2 norm of matrices
  if(print_matrix_summary) {
    El::Matrix<DataType, Device> weights_Rr;
    get_weight_matrix(kfac_gru_util::gru_weight_matrix_type::Rr, weights_Rr);
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) weights_Rr, "Rr")
        << ", " << kfac_util::get_matrix_stat(Ainv, "Ainv")
        << ", " << kfac_util::get_matrix_stat(Ginv, "Ginv")
        << ", " << kfac_util::get_matrix_stat(gradients_Rr, "grad")
        << ", " << kfac_util::get_matrix_stat(Fgrad, "Finvgrad")
        << std::endl;
    std::cout << oss.str();
  }

  // Update gradients in the buffer
  El::Matrix<DataType, Device> grad_buffer_Rr;
  get_gradient_buffer(kfac_gru_util::gru_weight_matrix_type::Rr, grad_buffer_Rr);
  assert(Fgrad.Height() == grad_buffer_Rr.Height());
  assert(Fgrad.Width() == grad_buffer_Rr.Width());
  El::Copy(Fgrad, grad_buffer_Rr);
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_gru<Device>::get_preconditioned_grad_buffers() {
  get_gradient_buffer(kfac_gru_util::gru_weight_matrix_type::Rr, m_grad_buffer_Rr);
  m_grad_buffer_Rr.Attach(
      m_grad_buffer_Rr.Height()*m_grad_buffer_Rr.Width(), 1,
      m_grad_buffer_Rr.Buffer(),
      m_grad_buffer_Rr.Height()*m_grad_buffer_Rr.Width());
  std::vector<El::AbstractMatrix<DataType>*>
      ret = {&m_grad_buffer_Rr};
  return ret;
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_gru<Device>::get_internal_matrix_info() const {
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
  return list;
}

template <El::Device Device>
void kfac_block_gru<Device>::get_weight_matrix(
    const kfac_gru_util::gru_weight_matrix_type matrix_type,
    El::Matrix<DataType, Device>& view) {
  const size_t hidden_size = get_gru_layer()->get_hidden_size();
  const auto ids = kfac_gru_util::get_gru_weight_offset(matrix_type);
  auto& weights_hh = this->m_layer->get_weights(ids.first);
  const auto& dtw_hh = dynamic_cast<data_type_weights<DataType>*>(&weights_hh);
  const auto& weight_matrix_hh = dtw_hh->get_values().LockedMatrix();
  const auto& weights_mat = El::LockedView(
      (El::Matrix<DataType, Device>&) weight_matrix_hh,
      El::IR(hidden_size*ids.second, hidden_size*(ids.second+1)), El::ALL);
  El::LockedView(view, weights_mat);
}

template <El::Device Device>
void kfac_block_gru<Device>::get_gradient_matrix(
    const kfac_gru_util::gru_weight_matrix_type matrix_type,
    El::Matrix<DataType, Device>& view) {
  const size_t hidden_size = get_gru_layer()->get_hidden_size();
  const auto ids = kfac_gru_util::get_gru_weight_offset(matrix_type);
  auto& weights_hh = this->m_layer->get_weights(ids.first);
  optimizer *opt_hh = weights_hh.get_optimizer();
  auto* dto_hh = dynamic_cast<data_type_optimizer<DataType>*>(opt_hh);
  const auto& gradients_hh = dto_hh->get_gradient().LockedMatrix();
  assert(gradients_hh.Height() == hidden_size*3);
  const auto gradients_mat = El::LockedView(
      (El::Matrix<DataType, Device>&) gradients_hh,
      El::IR(hidden_size*ids.second, hidden_size*(ids.second+1)), El::ALL);
  El::LockedView(view, gradients_mat);
}

template <El::Device Device>
void kfac_block_gru<Device>::get_gradient_buffer(
    const kfac_gru_util::gru_weight_matrix_type matrix_type,
    El::Matrix<DataType, Device>& view) {
  const size_t hidden_size = get_gru_layer()->get_hidden_size();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
      gradient_scale = El::TypeTraits<DataType>::One();
  const auto ids = kfac_gru_util::get_gru_weight_offset(matrix_type);
  auto& weights_hh = this->m_layer->get_weights(ids.first);
  optimizer *opt_hh = weights_hh.get_optimizer();
  auto& grad_buffer_hh = opt_hh->get_gradient_buffer(
      dst_scale, gradient_scale, false).Matrix();
  assert(grad_buffer_hh.Height() == hidden_size*3);
  auto grad_buffer_mat = El::View(
      (El::Matrix<DataType, Device>&) grad_buffer_hh,
      El::IR(hidden_size*ids.second, hidden_size*(ids.second+1)), El::ALL);
  El::View(view, grad_buffer_mat);
}

std::pair<int, int> kfac_gru_util::get_gru_weight_offset(
    const gru_weight_matrix_type matrix_type) {
  if(matrix_type == gru_weight_matrix_type::Wr)
    return std::make_pair<int, int>(0, 0);
  else if(matrix_type == gru_weight_matrix_type::Wi)
    return std::make_pair<int, int>(0, 1);
  else if(matrix_type == gru_weight_matrix_type::Wh)
    return std::make_pair<int, int>(0, 2);
  else if(matrix_type == gru_weight_matrix_type::Rr)
    return std::make_pair<int, int>(1, 0);
  else if(matrix_type == gru_weight_matrix_type::Ri)
    return std::make_pair<int, int>(1, 1);
  else if(matrix_type == gru_weight_matrix_type::Rh)
    return std::make_pair<int, int>(1, 2);
  else if(matrix_type == gru_weight_matrix_type::bWr)
    return std::make_pair<int, int>(2, 0);
  else if(matrix_type == gru_weight_matrix_type::bWi)
    return std::make_pair<int, int>(2, 1);
  else if(matrix_type == gru_weight_matrix_type::bWh)
    return std::make_pair<int, int>(2, 2);
  else if(matrix_type == gru_weight_matrix_type::bRr)
    return std::make_pair<int, int>(3, 0);
  else if(matrix_type == gru_weight_matrix_type::bRi)
    return std::make_pair<int, int>(3, 1);
  else if(matrix_type == gru_weight_matrix_type::bRh)
    return std::make_pair<int, int>(3, 2);
  LBANN_ERROR("Invalid GRU matrix type");
}

template <>
void kfac_gru_util::unpack_reserve_space(
    const DataType* reserve_space_fwd,
    El::Matrix<DataType, El::Device::CPU>& r,
    El::Matrix<DataType, El::Device::CPU>& i,
    const size_t hidden_size,
    const size_t seq_length,
    const size_t local_batch_size,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  // TODO: implement
  LBANN_ERROR("unimplemented");
}

template <>
void kfac_gru_util::get_g_Rr(
    const El::Matrix<DataType, El::Device::CPU>& h,
    const El::Matrix<DataType, El::Device::CPU>& dh,
    const El::Matrix<DataType, El::Device::CPU>& hfc,
    const El::Matrix<DataType, El::Device::CPU>& r,
    const El::Matrix<DataType, El::Device::CPU>& i,
    El::Matrix<DataType, El::Device::CPU>& g,
    const size_t count,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  // TODO: implement
  LBANN_ERROR("unimplemented");
}

template class kfac_block_gru<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac_block_gru<El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace callback
} // namespace lbann
