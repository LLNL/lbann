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

#ifdef LBANN_HAS_GPU
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
#endif // LBANN_HAS_GPU

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_gru<Device>::get_local_kronecker_buffers() {
  std::vector<El::AbstractMatrix<DataType>*> ret =
      {&m_kronecker_factor_buf_A_h, &m_kronecker_factor_buf_A_x};
  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES)
    ret.push_back(&m_kronecker_factor_buf_G[matrix_type]);
  return ret;
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

  const auto input_dims = this->m_layer->get_input_dims();
  const size_t input_size = get_input_size();
  const size_t hidden_size = get_hidden_size();
  const size_t seq_length = get_seq_length();

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

  auto& A_h = this->get_workspace_matrix("A_h", hidden_size, hidden_size);
  auto& A_x = this->get_workspace_matrix("A_x", input_size, input_size);
  std::unordered_map<kfac_gru_util::weight_type,
                     El::Matrix<DataType, Device>*> Gs;
  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const size_t height = kfac_gru_util::is_matrix_height_hidden(matrix_type)
        ? hidden_size : input_size;
    Gs[matrix_type] = &this->get_workspace_matrix(
        std::string("G_")+kfac_gru_util::get_matrix_type_name(matrix_type),
        height, height);
  }

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
  get_weight_matrix(kfac_gru_util::weight_type::Rh, weights_Rh);
  get_weight_matrix(kfac_gru_util::weight_type::bRh, biases_Rh);

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
  std::unordered_map<kfac_gru_util::weight_type,
                     El::Matrix<DataType, Device>*> gs;
  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto height = kfac_gru_util::is_matrix_height_hidden(matrix_type)
        ? hidden_size : input_size;
    gs[matrix_type] = &this->get_workspace_matrix(
        std::string("g_")+kfac_gru_util::get_matrix_type_name(matrix_type),
        height,
        local_batch_size*seq_length);
  }
  for(size_t t = 0; t < seq_length; t++) {
    // OPTIMIZE: g_Rr_t and g_Wr_t is the same!
    // hidden_size x local_batch_size
    auto g_Rr_t = El::View(*gs[kfac_gru_util::weight_type::Rr], El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    auto g_Ri_t = El::View(*gs[kfac_gru_util::weight_type::Ri], El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    auto g_Rh_t = El::View(*gs[kfac_gru_util::weight_type::Rh], El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    auto g_Wr_t = El::View(*gs[kfac_gru_util::weight_type::Wr], El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    auto g_Wi_t = El::View(*gs[kfac_gru_util::weight_type::Wi], El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    auto g_Wh_t = El::View(*gs[kfac_gru_util::weight_type::Wh], El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    const auto x_t =
        El::LockedView((El::Matrix<DataType, Device>&) local_inputs,
                       El::IR(input_size*t, input_size*(t+1)), El::ALL);
    const auto h_t =
        El::LockedView((El::Matrix<DataType, Device>&) local_outputs,
                       El::IR(hidden_size*t, hidden_size*(t+1)), El::ALL);
    const auto hprev_t =
        (t == 0
         ? El::LockedView((El::Matrix<DataType, Device>&) h0, El::ALL, El::ALL) // OPTIMIZE
         : El::LockedView((El::Matrix<DataType, Device>&) local_outputs,
                          El::IR(hidden_size*(t-1), hidden_size*t), El::ALL));
    const auto dh_t =
        El::LockedView((El::Matrix<DataType, Device>&) local_errors,
                       El::IR(hidden_size*t, hidden_size*(t+1)), El::ALL);
    const auto r_t = El::LockedView(r, El::ALL, El::IR(t, t+1));
    const auto i_t = El::LockedView(i, El::ALL, El::IR(t, t+1));
    const auto hfc_t = El::LockedView(hfc, El::ALL, El::IR(t*local_batch_size, (t+1)*local_batch_size));
    kfac_gru_util::get_g(h_t, hprev_t, dh_t, hfc_t, r_t, i_t,
                         g_Rr_t, g_Ri_t, g_Rh_t,
                         g_Wr_t, g_Wi_t, g_Wh_t,
                         hidden_size, sync_info);

    const DataType alpha = 1.0/seq_length;
    const DataType beta = (t == 0 ? El::TypeTraits<DataType>::Zero() : El::TypeTraits<DataType>::One());
    El::Gemm(
        El::NORMAL, El::TRANSPOSE,
        alpha, h_t, h_t,
        beta, A_h);
    El::Gemm(
        El::NORMAL, El::TRANSPOSE,
        alpha, x_t, x_t,
        beta, A_x);
    for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      auto& g = *gs[matrix_type];
      auto& G = *Gs[matrix_type];
      El::Gemm(
          El::NORMAL, El::TRANSPOSE,
          alpha, g, g,
          beta, G);
    }
  }

  m_kronecker_factor_buf_A_h.Resize(A_h.Height()*(A_h.Height()+1)/2, 1);
  m_kronecker_factor_buf_A_x.Resize(A_x.Height()*(A_x.Height()+1)/2, 1);
  kfac_util::pack_lower_tri(m_kronecker_factor_buf_A_h, A_h, sync_info);
  kfac_util::pack_lower_tri(m_kronecker_factor_buf_A_x, A_x, sync_info);
  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    auto& G = *Gs[matrix_type];
    m_kronecker_factor_buf_G[matrix_type].Resize(G.Height()*(G.Height()+1)/2, 1);
    kfac_util::pack_lower_tri(m_kronecker_factor_buf_G[matrix_type], G, sync_info);
  }

  // Dump matrices for debugging
  if(comm->am_trainer_master() && print_matrix) {
    for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
      std::cout << std::endl;
      El::Print(*gs[matrix_type], std::string("g_")+mname);
      std::cout << std::endl;
    }
  }

  // Dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) local_outputs, "h")
        << ", " << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) A_h, "A_h")
        << ", " << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) A_x, "A_x");

    for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
      oss << ", " << kfac_util::get_matrix_stat(
          (const El::Matrix<DataType, Device>&) *gs[matrix_type],
          (std::string("g_")+mname).c_str())
          << ", " << kfac_util::get_matrix_stat(
              (const El::Matrix<DataType, Device>&) *Gs[matrix_type],
              (std::string("G_")+mname).c_str());
    }
    oss << std::endl;
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_gru<Device>::update_kronecker_average(
    lbann_comm* comm,
    const DataType kronecker_decay,
    const bool print_matrix,
    const bool print_matrix_summary) {
  const auto& sync_info = this->get_sync_info();
  const size_t hidden_size = get_hidden_size();
  const auto input_dims = this->m_layer->get_input_dims();
  const size_t input_size = get_input_size();

  auto& A_h = this->get_workspace_matrix("Ah", hidden_size, hidden_size);
  auto& A_x = this->get_workspace_matrix("Ax", input_size, input_size);
  kfac_util::unpack_lower_tri(A_h, m_kronecker_factor_buf_A_h, sync_info);
  kfac_util::unpack_lower_tri(A_x, m_kronecker_factor_buf_A_x, sync_info);
  if(!this->m_has_kronecker_inverse) {
    El::Copy(A_h, m_kronecker_average_A_h);
    El::Copy(A_x, m_kronecker_average_A_x);
  }
  auto &Aave_h = m_kronecker_average_A_h;
  auto &Aave_x = m_kronecker_average_A_x;
  kfac_util::update_kronecker_average(
      Aave_h, A_h, A_h.Height()*A_h.Width(), kronecker_decay, sync_info);
  kfac_util::update_kronecker_average(
      Aave_x, A_x, A_x.Height()*A_x.Width(), kronecker_decay, sync_info);

  // Dump matrices for debugging
  if(comm->am_trainer_master() && print_matrix) {
    if(comm->am_trainer_master()) {
      std::cout << std::endl; El::Print(A_h, "A_h");
      std::cout << std::endl; El::Print(A_x, "A_x");
      std::cout << std::endl; El::Print(Aave_h, "Aave_h");
      std::cout << std::endl; El::Print(Aave_x, "Aave_x");
    }
  }

  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    auto& G = this->get_workspace_matrix(
        std::string("G_")+mname,
        hidden_size, hidden_size);
    kfac_util::unpack_lower_tri(
        G, m_kronecker_factor_buf_G[matrix_type],
        sync_info);
    if(!this->m_has_kronecker_inverse)
      El::Copy(G, m_kronecker_average_G[matrix_type]);
    auto &Gave = m_kronecker_average_G[matrix_type];
    kfac_util::update_kronecker_average(
        Gave, G, G.Height()*G.Width(), kronecker_decay, sync_info);

    // Dump matrices for debugging
    if(comm->am_trainer_master() && print_matrix) {
      std::cout << std::endl; El::Print(G, std::string("G_")+mname);
      std::cout << std::endl; El::Print(Gave, std::string("Gave_")+mname);
      std::cout << std::endl;
    }

    // Dump L2 norm of matrices
    if(comm->am_trainer_master() && print_matrix_summary) {
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name()
          << ": " << kfac_util::get_matrix_stat(G, (std::string("G_")+mname).c_str())
          << std::endl;
      std::cout << oss.str();
    }
  }

  // Dump L2 norm of matrices
  if(comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name()
        << ": " << kfac_util::get_matrix_stat(Aave_h, "Aave_h")
        << ", " << kfac_util::get_matrix_stat(Aave_x, "Aave_x");
    for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
      auto &Gave = m_kronecker_average_G[matrix_type];
      oss << ", " << kfac_util::get_matrix_stat(
          Gave, (std::string("Gave_")+mname).c_str());
    }
    oss << std::endl;
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

  const auto &Aave_h = m_kronecker_average_A_h;
  const auto &Aave_x = m_kronecker_average_A_x;
  if(!this->m_has_kronecker_inverse) {
    m_kronecker_inverse_A_h.Resize(Aave_h.Height(), Aave_h.Width());
    m_kronecker_inverse_A_x.Resize(Aave_x.Height(), Aave_x.Width());
  }
  // TODO: Refactoring
  const DataType pi = 1.0; // TODO: Compute pi if use_pi
  auto& Ainv_h = m_kronecker_inverse_A_h;
  auto& Ainv_x = m_kronecker_inverse_A_x;
  auto& ALinv_h = this->get_workspace_matrix("ALinv_h", Aave_h.Height(), Aave_h.Height());
  auto& ALinv_x = this->get_workspace_matrix("ALinv_x", Aave_x.Height(), Aave_x.Height());
  kfac_util::get_matrix_inverse(
      Ainv_h, ALinv_h, Aave_h, comm->am_trainer_master() && print_time,
      DataType(damping_act*pi), 0,
      false, sync_info);
  kfac_util::get_matrix_inverse(
      Ainv_x, ALinv_x, Aave_x, comm->am_trainer_master() && print_time,
      DataType(damping_act*pi), 0,
      false, sync_info);

  // Dump matrices for debugging
  if(print_matrix) {
    std::cout << std::endl; El::Print(Ainv_h, "Ainv_h");
    std::cout << std::endl; El::Print(Ainv_x, "Ainv_x");
    std::cout << std::endl;
  }

  // Dump L2 norm of matrices
  if(print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
        << kfac_util::get_matrix_stat(Ainv_h, "Ainv_h")
        << kfac_util::get_matrix_stat(Ainv_x, "Ainv_x")
        << std::endl;
    std::cout << oss.str();
  }

  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    const auto &Gave = m_kronecker_average_G[matrix_type];
    if(!this->m_has_kronecker_inverse)
      m_kronecker_inverse_G[matrix_type].Resize(Gave.Height(), Gave.Width());
    auto& Ginv = m_kronecker_inverse_G[matrix_type];
    auto& GLinv = this->get_workspace_matrix(
        std::string("GLinv_"+mname),
        Gave.Height(), Gave.Height());
    kfac_util::get_matrix_inverse(
        Ginv, GLinv, Gave, comm->am_trainer_master() && print_time,
        DataType(damping_err/pi), 0,
        false, sync_info);

    // Compute preconditioned gradients
    El::Matrix<DataType, Device> gradients_mat;
    get_gradient_matrix(matrix_type, gradients_mat);
    auto& Gg = this->get_workspace_matrix(
        std::string("Gg_")+mname,
        Ginv.Height(),
        gradients_mat.Width());
    El::Gemm(
        El::NORMAL, El::NORMAL, // gradient matrices are in row-major
        El::TypeTraits<DataType>::One(), Ginv, gradients_mat,
        El::TypeTraits<DataType>::Zero(), Gg);
    auto& Fgrad = this->get_workspace_matrix(
        std::string("Fgrad_")+mname,
        gradients_mat.Height(), gradients_mat.Width());
    El::Gemm(
        El::NORMAL, El::NORMAL,
        learning_rate_factor, Gg,
        kfac_gru_util::is_matrix_height_hidden(matrix_type) ? Ainv_h : Ainv_x,
        El::TypeTraits<DataType>::Zero(), Fgrad);

    // Dump matrices for debugging
    if(print_matrix) {
      std::cout << std::endl; El::Print(Ginv, std::string("Ginv_")+mname);
      std::cout << std::endl; El::Print(gradients_mat, std::string("grad_")+mname);
      std::cout << std::endl; El::Print(Fgrad, std::string("Finvgrad_")+mname);
      std::cout << std::endl;
    }

    // Dump L2 norm of matrices
    if(print_matrix_summary) {
      El::Matrix<DataType, Device> weights_mat;
      get_weight_matrix(matrix_type, weights_mat);
      std::ostringstream oss;
      oss << "K-FAC callback: L2 norm @ "<< this->m_layer->get_name() << ": "
          << kfac_util::get_matrix_stat((const El::Matrix<DataType, Device>&) weights_mat, mname.c_str())
          << ", " << kfac_util::get_matrix_stat(Ginv, (std::string("Ginv_")+mname).c_str())
          << ", " << kfac_util::get_matrix_stat(gradients_mat, (std::string("grad_")+mname).c_str())
          << ", " << kfac_util::get_matrix_stat(Fgrad, (std::string("Finvgrad_")+mname).c_str())
          << std::endl;
      std::cout << oss.str();
    }

    // Update gradients in the buffer
    El::Matrix<DataType, Device> grad_buffer_mat;
    get_gradient_buffer(matrix_type, grad_buffer_mat);
    assert(Fgrad.Height() == grad_buffer_mat.Height());
    assert(Fgrad.Width() == grad_buffer_mat.Width());
    El::Copy(Fgrad, grad_buffer_mat);
  }

  if(!this->m_has_kronecker_inverse)
    this->m_has_kronecker_inverse = true;
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_gru<Device>::get_preconditioned_grad_buffers() {
  std::vector<El::AbstractMatrix<DataType>*> ret;
  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    auto& buf = m_grad_buffer_G[matrix_type];
    get_gradient_buffer(matrix_type, buf);
    buf.Attach(
        buf.Height()*buf.Width(), 1,
        buf.Buffer(),
        buf.Height()*buf.Width());
    ret.push_back(&buf);
  }
  return ret;
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_gru<Device>::get_internal_matrix_info() const {
  std::vector<std::tuple<std::string, size_t, size_t>> list;
  const auto emplace =
      [&list](const std::string& name,
              const El::Matrix<DataType, Device>& m) {
        list.emplace_back(name, m.Height(), m.Width());
      };
  const auto emplace_if_available =
      [&list, &emplace](
          const std::string& name,
          const std::unordered_map<kfac_gru_util::weight_type,
          El::Matrix<DataType, Device>>& map,
          const kfac_gru_util::weight_type& matrix_type) {
        auto i = map.find(matrix_type);
        if(i != map.end())
          emplace(name, (*i).second);
      };

  emplace("buf_A_h", m_kronecker_factor_buf_A_h);
  emplace("buf_A_x", m_kronecker_factor_buf_A_x);
  emplace("average_A_h", m_kronecker_average_A_h);
  emplace("average_A_x", m_kronecker_average_A_x);
  emplace("inverse_A_h", m_kronecker_inverse_A_h);
  emplace("inverse_A_x", m_kronecker_inverse_A_x);
  for(auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    emplace_if_available("buf_G_"+mname, m_kronecker_factor_buf_G, matrix_type);
    emplace_if_available("average_G_"+mname, m_kronecker_average_G, matrix_type);
    emplace_if_available("inverse_G_"+mname, m_kronecker_inverse_G, matrix_type);
  }
  return list;
}

template <El::Device Device>
void kfac_block_gru<Device>::get_weight_matrix(
    const kfac_gru_util::weight_type matrix_type,
    El::Matrix<DataType, Device>& view) {
  const size_t hidden_size = get_hidden_size();
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
    const kfac_gru_util::weight_type matrix_type,
    El::Matrix<DataType, Device>& view) {
  const size_t hidden_size = get_hidden_size();
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
    const kfac_gru_util::weight_type matrix_type,
    El::Matrix<DataType, Device>& view) {
  const size_t hidden_size = get_hidden_size();
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

std::string kfac_gru_util::get_matrix_type_name(
    const weight_type& matrix_type) {
  if(matrix_type == weight_type::Wr) return "Wr";
  else if(matrix_type == weight_type::Wi) return "Wi";
  else if(matrix_type == weight_type::Wh) return "Wh";
  else if(matrix_type == weight_type::Rr) return "Rr";
  else if(matrix_type == weight_type::Ri) return "Ri";
  else if(matrix_type == weight_type::Rh) return "Rh";
  else if(matrix_type == weight_type::bWr) return "bWr";
  else if(matrix_type == weight_type::bWi) return "bWi";
  else if(matrix_type == weight_type::bWh) return "bWh";
  else if(matrix_type == weight_type::bRr) return "bRr";
  else if(matrix_type == weight_type::bRi) return "bRi";
  else if(matrix_type == weight_type::bRh) return "bRh";
  LBANN_ERROR("Invalid matrix type");
}

bool kfac_gru_util::is_matrix_height_hidden(
    const weight_type& matrix_type) {
  if(matrix_type == weight_type::Wr
     || matrix_type == weight_type::Wi
     || matrix_type == weight_type::Wh) return false;
  else if(matrix_type == weight_type::Rr
          || matrix_type == weight_type::Ri
          || matrix_type == weight_type::Rh) return true;
  LBANN_ERROR("Invalid matrix type");
}

std::pair<int, int> kfac_gru_util::get_gru_weight_offset(
    const weight_type matrix_type) {
  if(matrix_type == weight_type::Wr)       return std::make_pair<int, int>(0, 0);
  else if(matrix_type == weight_type::Wi)  return std::make_pair<int, int>(0, 1);
  else if(matrix_type == weight_type::Wh)  return std::make_pair<int, int>(0, 2);
  else if(matrix_type == weight_type::Rr)  return std::make_pair<int, int>(1, 0);
  else if(matrix_type == weight_type::Ri)  return std::make_pair<int, int>(1, 1);
  else if(matrix_type == weight_type::Rh)  return std::make_pair<int, int>(1, 2);
  else if(matrix_type == weight_type::bWr) return std::make_pair<int, int>(2, 0);
  else if(matrix_type == weight_type::bWi) return std::make_pair<int, int>(2, 1);
  else if(matrix_type == weight_type::bWh) return std::make_pair<int, int>(2, 2);
  else if(matrix_type == weight_type::bRr) return std::make_pair<int, int>(3, 0);
  else if(matrix_type == weight_type::bRi) return std::make_pair<int, int>(3, 1);
  else if(matrix_type == weight_type::bRh) return std::make_pair<int, int>(3, 2);
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
void kfac_gru_util::get_g(
    const El::Matrix<DataType, El::Device::CPU>& h,
    const El::Matrix<DataType, El::Device::CPU>& hprev,
    const El::Matrix<DataType, El::Device::CPU>& dh,
    const El::Matrix<DataType, El::Device::CPU>& hfc,
    const El::Matrix<DataType, El::Device::CPU>& r,
    const El::Matrix<DataType, El::Device::CPU>& i,
    El::Matrix<DataType, El::Device::CPU>& g_Rr,
    El::Matrix<DataType, El::Device::CPU>& g_Ri,
    El::Matrix<DataType, El::Device::CPU>& g_Rh,
    El::Matrix<DataType, El::Device::CPU>& g_Wr,
    El::Matrix<DataType, El::Device::CPU>& g_Wi,
    El::Matrix<DataType, El::Device::CPU>& g_Wh,
    size_t count,
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
