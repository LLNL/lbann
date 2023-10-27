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

#include "lbann/execution_algorithms/kfac/kfac_block_gru.hpp"
#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <typename TensorDataType>
inline TensorDataType sigmoid(const TensorDataType& x)
{
  return (std::tanh(x * 0.5) + 1.0) * 0.5;
}
template <typename TensorDataType>
inline TensorDataType sigmoid_deriv(const TensorDataType& x)
{
  const TensorDataType t = sigmoid(x);
  return t * (1.0 - t);
}
template <typename TensorDataType>
inline TensorDataType sigmoid_inv(const TensorDataType& x)
{
  return std::log(x / (1.0 - x));
}
template <typename TensorDataType>
inline TensorDataType tanh_deriv(const TensorDataType& x)
{
  const TensorDataType t = std::tanh(x);
  return 1.0 - t * t;
}
template <typename TensorDataType>
inline TensorDataType tanh_inv(const TensorDataType& x)
{
  return 0.5 * std::log((1.0 + x) / (1.0 - x));
}

} // namespace

//////////////////////////////////////////////////////////////////////////////////
////  Sub-Grid Parallelism Communication Functions
//////////////////////////////////////////////////////////////////////////////////

template <>
void kfac_block_gru<El::Device::CPU>::send_recv_reserve_space(lbann_comm* comm)
{}

#ifdef LBANN_HAS_GPU
template <>
void kfac_block_gru<El::Device::GPU>::send_recv_reserve_space(lbann_comm* comm)
{
#ifndef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  LBANN_ERROR("GRU not supported on GPU without cuDNN support.");
#else
  // Does not support Model parallel only Data Parallel
  if (this->m_layer->get_data_layout() != data_layout::DATA_PARALLEL)
    LBANN_ERROR("send_recv_reserve_space function in",
                " kfac_block_gru.cpp only supports Data Parallelism");

  const El::mpi::Comm& combined_comm = comm->get_combined_grid_comm();
  const int comm_rank = El::mpi::Rank(comm->get_KFAC_comm());

  std::vector<int> primary_grid_ranks = comm->get_primary_grid_ranks();
  std::vector<int> secondary_grid_ranks = comm->get_secondary_grid_ranks();

  int num_process_primary_grid = (int)primary_grid_ranks.size();
  int num_process_secondary_grid = (int)secondary_grid_ranks.size();

  // Send the size of reserve space to secondary  grid
  if (m_reserve_space_fwd_size == 0) {

    El::Matrix<DataType, El::Device::CPU> reserve_size(1, 1);

    El::SyncInfo<El::Device::CPU> sync_info =
      El::SyncInfoFromMatrix(reserve_size);

    if (comm->get_grid_type() == GridType::PRIMARY_GRID) {
      const auto& reserve_space = get_gru_layer()->get_reserve_space();
      reserve_size(0, 0) = reserve_space.size();
      int num_sends = (int)std::ceil((float)num_process_secondary_grid /
                                     (float)num_process_primary_grid);
      for (int num_send = 0; num_send < num_sends; num_send++) {
        if (comm_rank + num_send * num_process_primary_grid <
            num_process_secondary_grid) {
          int to_send_index = comm_rank + num_send * num_process_primary_grid;
          ::El::mpi::Send((DataType*)reserve_size.Buffer(),
                          1,
                          secondary_grid_ranks[to_send_index],
                          combined_comm,
                          sync_info);
        }
      }
    }

    if (comm->get_grid_type() == GridType::SECONDARY_GRID) {
      int recv_index = comm_rank % num_process_primary_grid;
      ::El::mpi::Recv((DataType*)reserve_size.Buffer(),
                      1,
                      primary_grid_ranks[recv_index],
                      combined_comm,
                      sync_info);
    }
    m_reserve_space_fwd_size = (size_t)reserve_size(0, 0);
  }

  const auto& sync_info = this->get_sync_info();

  // Send reserve space to secondary  grid
  if (comm->get_grid_type() == GridType::PRIMARY_GRID or
      comm->get_KFAC_subgrid_create_two_models()) {
    const auto& reserve_space = get_gru_layer()->get_reserve_space();

    int num_sends = (int)std::ceil((float)num_process_secondary_grid /
                                   (float)num_process_primary_grid);
    for (int num_send = 0; num_send < num_sends; num_send++) {

      if (comm_rank + num_send * num_process_primary_grid <
          num_process_secondary_grid) {
        int to_send_index = comm_rank + num_send * num_process_primary_grid;
        if (comm->enable_subgrid_async_communication() == true) {
          kfac::ReqT send_request;
          m_requests_workspace.push_back(send_request);
          ::Al::NonblockingSend<kfac::BackendT>(
            (El::byte*)reserve_space.data(),
            m_reserve_space_fwd_size,
            secondary_grid_ranks[to_send_index],
            combined_comm.template GetComm<kfac::BackendT>(sync_info),
            m_requests_workspace.back());
        }
        else
          ::Al::Send<kfac::BackendT>(
            (El::byte*)reserve_space.data(),
            m_reserve_space_fwd_size,
            secondary_grid_ranks[to_send_index],
            combined_comm.template GetComm<kfac::BackendT>(sync_info));
      }
    }
  }
  if (comm->get_grid_type() == GridType::SECONDARY_GRID) {
    m_reserve_space_fwd.allocate(m_reserve_space_fwd_size);
    int recv_index = comm_rank % num_process_primary_grid;
    if (comm->enable_subgrid_async_communication() == true) {
      El::Synchronize(sync_info);
      kfac::ReqT recv_request;
      m_requests_workspace.push_back(recv_request);
      ::Al::NonblockingRecv<kfac::BackendT>(
        (El::byte*)m_reserve_space_fwd.data(),
        m_reserve_space_fwd_size,
        primary_grid_ranks[recv_index],
        combined_comm.template GetComm<kfac::BackendT>(sync_info),
        m_requests_workspace.back());
    }
    else {
      ::Al::Recv<kfac::BackendT>(
        (El::byte*)m_reserve_space_fwd.data(),
        m_reserve_space_fwd_size,
        primary_grid_ranks[recv_index],
        combined_comm.template GetComm<kfac::BackendT>(sync_info));
    }
  }
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
}
#endif // LBANN_HAS_GPU

template <>
void kfac_block_gru<El::Device::CPU>::on_forward_prop_end(lbann_comm* comm)
{}

#ifdef LBANN_HAS_GPU
template <>
void kfac_block_gru<El::Device::GPU>::on_forward_prop_end(lbann_comm* comm)
{
#ifndef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  LBANN_ERROR("GRU not supported on GPU without cuDNN support.");
#else

  if (comm->get_grid_type() == GridType::NO_GRID or
      comm->get_KFAC_subgrid_create_two_models()) {
    const auto& reserve_space = get_gru_layer()->get_reserve_space();
    if (m_reserve_space_fwd.size() != reserve_space.size())
      m_reserve_space_fwd.allocate(reserve_space.size());
    const auto& sync_info = this->get_sync_info();
    gpu_lib::mem_copy_async(m_reserve_space_fwd.data(),
                            reserve_space.data(),
                            reserve_space.size(),
                            gpu_lib::GPU_MEMCPY_DEVICE_TO_DEVICE,
                            sync_info.Stream());
    /*CHECK_CUDA(cudaMemcpyAsync(
        m_reserve_space_fwd.data(),
        reserve_space.data(),
        reserve_space.size(),
        cudaMemcpyDeviceToDevice,
        sync_info.Stream()));*/
  }
  else {
    send_recv_reserve_space(comm);
  }
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
}
#endif // LBANN_HAS_GPU

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_gru<Device>::get_local_kronecker_buffers()
{
  std::vector<El::AbstractMatrix<DataType>*> ret = {
    &m_kronecker_factor_buf_A_h,
    &m_kronecker_factor_buf_A_x};
  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES)
    ret.push_back(&m_kronecker_factor_buf_G[matrix_type]);
  return ret;
}

template <El::Device Device>
void kfac_block_gru<Device>::compute_local_kronecker_factors(
  lbann_comm* comm,
  const bool print_matrix,
  const bool print_matrix_summary)
{
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

  const El::Matrix<DataType, Device>& local_inputs =
    this->m_parent_local_activations[0]->LockedMatrix();
  const El::Matrix<DataType, Device>& h0 =
    this->m_parent_local_activations[1]->LockedMatrix();
  const El::Matrix<DataType, Device>& local_outputs =
    this->m_parent_local_activations[2]->LockedMatrix();
  const El::Matrix<DataType, Device>& local_errors =
    this->m_child_local_errors[0]->LockedMatrix();
  const auto local_batch_size = local_inputs.Width();

  auto& A_h = this->get_workspace_matrix("A_h", hidden_size, hidden_size);
  auto& A_x = this->get_workspace_matrix("A_x", input_size, input_size);
  std::unordered_map<kfac_gru_util::weight_type, El::Matrix<DataType, Device>*>
    Gs;
  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    Gs[matrix_type] = &this->get_workspace_matrix(
      std::string("G_") + kfac_gru_util::get_matrix_type_name(matrix_type),
      hidden_size,
      hidden_size);
  }

  // r, i: (hidden_size*local_batch_size) x seq_length
  auto& r =
    this->get_workspace_matrix("r", hidden_size * local_batch_size, seq_length);
  auto& i =
    this->get_workspace_matrix("i", hidden_size * local_batch_size, seq_length);
  auto& biases_ones = this->get_workspace_matrix("b_ones", 1, local_batch_size);
  El::Ones(biases_ones, 1, local_batch_size);

  get_r_i(r,
          i,
          biases_ones,
          local_inputs,
          local_outputs,
          h0,
          local_batch_size,
          sync_info);

  // hfc_t = R_h h_{t-1} + b_Rh : hidden_size x local_batch_size
  // weights_Rh: hidden_size x hidden_size
  // biases_Rh: hidden_size x 1
  auto& hfc = this->get_workspace_matrix("hfc_t",
                                         hidden_size,
                                         local_batch_size * seq_length);

  El::Matrix<DataType, Device> weights_Rh, biases_Rh;
  get_weight_matrix(kfac_gru_util::weight_type::Rh, weights_Rh);
  get_weight_matrix(kfac_gru_util::weight_type::bRh, biases_Rh);

  // Recompute hfc = R_h h_t + b_Rh
  // OPTIMIZE: compute with a single GEMM call
  for (size_t t = 0; t < seq_length; t++) {
    auto hfc_t =
      El::View(hfc,
               El::ALL,
               El::IR(t * local_batch_size, (t + 1) * local_batch_size));
    const auto h_prev =
      (t == 0 ? El::LockedView((El::Matrix<DataType, Device>&)h0)
              : El::LockedView((El::Matrix<DataType, Device>&)local_outputs,
                               El::IR(hidden_size * (t - 1), hidden_size * t),
                               El::ALL));
    El::Gemm(El::NORMAL,
             El::NORMAL, // weight matrices are in row-major
             El::TypeTraits<DataType>::One(),
             weights_Rh,
             h_prev,
             El::TypeTraits<DataType>::Zero(),
             hfc_t);
    El::Gemm(El::NORMAL,
             El::NORMAL,
             El::TypeTraits<DataType>::One(),
             biases_Rh,
             biases_ones,
             El::TypeTraits<DataType>::One(),
             hfc_t);
  }

  std::unordered_map<kfac_gru_util::weight_type, El::Matrix<DataType, Device>*>
    gs;
  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    gs[matrix_type] = &this->get_workspace_matrix(
      std::string("g_") + kfac_gru_util::get_matrix_type_name(matrix_type),
      hidden_size,
      local_batch_size * seq_length);
  }

  kfac_gru_util::get_g((El::Matrix<DataType, Device>&)local_outputs,
                       (El::Matrix<DataType, Device>&)h0,
                       (El::Matrix<DataType, Device>&)local_errors,
                       hfc,
                       r,
                       i,
                       *gs[kfac_gru_util::weight_type::Rr],
                       *gs[kfac_gru_util::weight_type::Ri],
                       *gs[kfac_gru_util::weight_type::Rh],
                       *gs[kfac_gru_util::weight_type::Wr],
                       *gs[kfac_gru_util::weight_type::Wi],
                       *gs[kfac_gru_util::weight_type::Wh],
                       hidden_size,
                       seq_length,
                       local_batch_size,
                       sync_info);

  {
    const DataType alpha = 1.0 / seq_length;
    for (size_t t = 0; t < seq_length; t++) {
      const DataType beta = (t == 0 ? El::TypeTraits<DataType>::Zero()
                                    : El::TypeTraits<DataType>::One());
      const auto x_t =
        El::LockedView((El::Matrix<DataType, Device>&)local_inputs,
                       El::IR(input_size * t, input_size * (t + 1)),
                       El::ALL);
      const auto h_t =
        El::LockedView((El::Matrix<DataType, Device>&)local_outputs,
                       El::IR(hidden_size * t, hidden_size * (t + 1)),
                       El::ALL);
      El::Gemm(El::NORMAL, El::TRANSPOSE, alpha, h_t, h_t, beta, A_h);
      El::Gemm(El::NORMAL, El::TRANSPOSE, alpha, x_t, x_t, beta, A_x);
    }
    for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      auto& g = *gs[matrix_type];
      auto& G = *Gs[matrix_type];
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               alpha,
               g,
               g,
               El::TypeTraits<DataType>::Zero(),
               G);
    }
  }

  m_kronecker_factor_buf_A_h.Resize(A_h.Height() * (A_h.Height() + 1) / 2, 1);
  m_kronecker_factor_buf_A_x.Resize(A_x.Height() * (A_x.Height() + 1) / 2, 1);
  kfac::pack_lower_tri(m_kronecker_factor_buf_A_h, A_h, sync_info);
  kfac::pack_lower_tri(m_kronecker_factor_buf_A_x, A_x, sync_info);
  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    auto& G = *Gs[matrix_type];
    m_kronecker_factor_buf_G[matrix_type].Resize(G.Height() * (G.Height() + 1) /
                                                   2,
                                                 1);
    kfac::pack_lower_tri(m_kronecker_factor_buf_G[matrix_type], G, sync_info);
  }

  // Dump matrices for debugging
  if (comm->am_trainer_master() && print_matrix) {
    for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
      std::cout << std::endl;
      El::Print(*gs[matrix_type], std::string("g_") + mname);
      std::cout << std::endl;
    }
  }

  // Dump L2 norm of matrices
  if (print_matrix_summary) {
    std::ostringstream oss;
    oss
      << "K-FAC: L2 norm @ " << this->m_layer->get_name() << " (process "
      << comm->get_rank_in_trainer() << ")"
      << ": "
      << kfac::get_matrix_stat(
           (const El::Matrix<DataType, Device>&)local_outputs,
           "h")
      << ": "
      << kfac::get_matrix_stat(
           (const El::Matrix<DataType, Device>&)local_errors,
           "dh")
      << ": "
      << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)hfc, "hfc")
      << ", "
      << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)A_h, "A_h")
      << ", "
      << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)A_x, "A_x")
      << ", "
      << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)r, "r")
      << ", "
      << kfac::get_matrix_stat((const El::Matrix<DataType, Device>&)i, "i")
      << std::endl;

    for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
      const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
      oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << " (process "
          << comm->get_rank_in_trainer() << ")"
          << ": "
          << kfac::get_matrix_stat(
               (const El::Matrix<DataType, Device>&)*gs[matrix_type],
               (std::string("g_") + mname).c_str())
          << ", "
          << kfac::get_matrix_stat(
               (const El::Matrix<DataType, Device>&)*Gs[matrix_type],
               (std::string("G") + mname).c_str())
          << std::endl;
    }
    std::cout << oss.str();
  }
}

template <El::Device Device>
void kfac_block_gru<Device>::update_kronecker_average(
  lbann_comm* comm,
  const DataType kronecker_decay,
  const bool print_matrix,
  const bool print_matrix_summary)
{
  const auto& sync_info = this->get_sync_info();
  const size_t hidden_size = get_hidden_size();
  const auto input_dims = this->m_layer->get_input_dims();
  const size_t input_size = get_input_size();

  auto& A_h = this->get_workspace_matrix("Ah", hidden_size, hidden_size);
  auto& A_x = this->get_workspace_matrix("Ax", input_size, input_size);
  kfac::unpack_lower_tri(A_h, m_kronecker_factor_buf_A_h, sync_info);
  kfac::unpack_lower_tri(A_x, m_kronecker_factor_buf_A_x, sync_info);
  if (!this->m_has_kronecker_inverse) {
    El::Copy(A_h, m_kronecker_average_A_h);
    El::Copy(A_x, m_kronecker_average_A_x);
  }
  auto& Aave_h = m_kronecker_average_A_h;
  auto& Aave_x = m_kronecker_average_A_x;
  kfac::update_kronecker_average(Aave_h,
                                 A_h,
                                 A_h.Height() * A_h.Width(),
                                 kronecker_decay,
                                 sync_info);
  kfac::update_kronecker_average(Aave_x,
                                 A_x,
                                 A_x.Height() * A_x.Width(),
                                 kronecker_decay,
                                 sync_info);

  // Dump matrices for debugging
  if (comm->am_trainer_master() && print_matrix) {
    if (comm->am_trainer_master()) {
      std::cout << std::endl;
      El::Print(A_h, "A_h");
      std::cout << std::endl;
      El::Print(A_x, "A_x");
      std::cout << std::endl;
      El::Print(Aave_h, "Aave_h");
      std::cout << std::endl;
      El::Print(Aave_x, "Aave_x");
    }
  }

  // Dump L2 norm of matrices
  if (comm->am_trainer_master() && print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << kfac::get_matrix_stat(Aave_h, "Aave_h") << ", "
        << kfac::get_matrix_stat(Aave_x, "Aave_x");
    oss << std::endl;
    std::cout << oss.str();
  }

  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    const size_t height = kfac_gru_util::is_matrix_height_hidden(matrix_type)
                            ? hidden_size
                            : input_size;
    auto& G =
      this->get_workspace_matrix(std::string("G_") + mname, height, height);
    kfac::unpack_lower_tri(G, m_kronecker_factor_buf_G[matrix_type], sync_info);
    if (!this->m_has_kronecker_inverse)
      El::Copy(G, m_kronecker_average_G[matrix_type]);
    auto& Gave = m_kronecker_average_G[matrix_type];
    kfac::update_kronecker_average(Gave,
                                   G,
                                   G.Height() * G.Width(),
                                   kronecker_decay,
                                   sync_info);

    // Dump matrices for debugging
    if (comm->am_trainer_master() && print_matrix) {
      std::cout << std::endl;
      El::Print(G, std::string("G_") + mname);
      std::cout << std::endl;
      El::Print(Gave, std::string("Gave_") + mname);
      std::cout << std::endl;
    }

    // Dump L2 norm of matrices
    if (comm->am_trainer_master() && print_matrix_summary) {
      std::ostringstream oss;
      oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
          << kfac::get_matrix_stat(G, (std::string("G_") + mname).c_str())
          << ": "
          << kfac::get_matrix_stat(Gave, (std::string("Gave_") + mname).c_str())
          << std::endl;
      std::cout << oss.str();
    }
  }
}

template <El::Device Device>
void kfac_block_gru<Device>::update_kronecker_inverse(
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

  const auto& Aave_h = m_kronecker_average_A_h;
  const auto& Aave_x = m_kronecker_average_A_x;
  if (!this->m_has_kronecker_inverse) {
    m_kronecker_inverse_A_h.Resize(Aave_h.Height(), Aave_h.Width());
    m_kronecker_inverse_A_x.Resize(Aave_x.Height(), Aave_x.Width());
  }

  const DataType pi = 1.0;
  if (use_pi)
    LBANN_ERROR(
      "The GRU K-FAC implementation does not currently support use_pi.");
  auto& Ainv_h = m_kronecker_inverse_A_h;
  auto& Ainv_x = m_kronecker_inverse_A_x;
  auto& ALinv_h =
    this->get_workspace_matrix("ALinv_h", Aave_h.Height(), Aave_h.Height());
  auto& ALinv_x =
    this->get_workspace_matrix("ALinv_x", Aave_x.Height(), Aave_x.Height());

  if (use_eigen_decomposition) {
    kfac::get_matrix_inverse(Ainv_h,
                             ALinv_h,
                             Aave_h,
                             comm->am_trainer_master() && print_time,
                             DataType(damping_act * pi),
                             0,
                             false,
                             sync_info);
    kfac::get_matrix_inverse(Ainv_x,
                             ALinv_x,
                             Aave_x,
                             comm->am_trainer_master() && print_time,
                             DataType(damping_act * pi),
                             0,
                             false,
                             sync_info);
  }
  else {
    kfac::get_matrix_inverse_eigen(Ainv_h,
                                   ALinv_h,
                                   Aave_h,
                                   comm->am_trainer_master() && print_time,
                                   DataType(damping_act * pi),
                                   0,
                                   false,
                                   sync_info);
    kfac::get_matrix_inverse_eigen(Ainv_x,
                                   ALinv_x,
                                   Aave_x,
                                   comm->am_trainer_master() && print_time,
                                   DataType(damping_act * pi),
                                   0,
                                   false,
                                   sync_info);
  }

  // Dump matrices for debugging
  if (print_matrix) {
    std::cout << std::endl;
    El::Print(Ainv_h, "Ainv_h");
    std::cout << std::endl;
    El::Print(Ainv_x, "Ainv_x");
    std::cout << std::endl;
  }

  // Dump L2 norm of matrices
  if (print_matrix_summary) {
    std::ostringstream oss;
    oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
        << kfac::get_matrix_stat(Ainv_h, "Ainv_h") << ", "
        << kfac::get_matrix_stat(Ainv_x, "Ainv_x") << std::endl;
    std::cout << oss.str();
  }

  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    const auto& Gave = m_kronecker_average_G[matrix_type];
    if (!this->m_has_kronecker_inverse)
      m_kronecker_inverse_G[matrix_type].Resize(Gave.Height(), Gave.Width());
    auto& Ginv = m_kronecker_inverse_G[matrix_type];
    auto& GLinv = this->get_workspace_matrix(std::string("GLinv_" + mname),
                                             Gave.Height(),
                                             Gave.Height());
    kfac::get_matrix_inverse(Ginv,
                             GLinv,
                             Gave,
                             comm->am_trainer_master() && print_time,
                             DataType(damping_err / pi),
                             0,
                             false,
                             sync_info);
  }

  if (!this->m_has_kronecker_inverse)
    this->m_has_kronecker_inverse = true;
}

template <El::Device Device>
void kfac_block_gru<Device>::compute_preconditioned_gradients(
  lbann_comm* comm,
  DataType learning_rate_factor,
  bool print_matrix,
  bool print_matrix_summary,
  bool print_time)
{

  auto& Ainv_h = m_kronecker_inverse_A_h;
  auto& Ainv_x = m_kronecker_inverse_A_x;

  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    auto& Ginv = m_kronecker_inverse_G[matrix_type];

    // Compute preconditioned gradients
    El::Matrix<DataType, Device> gradients_mat;
    get_gradient_matrix(matrix_type, gradients_mat);
    auto& Gg = this->get_workspace_matrix(std::string("Gg_") + mname,
                                          gradients_mat.Height(),
                                          Ginv.Width());
    El::Gemm(El::NORMAL,
             El::NORMAL, // gradient matrices are in row-major
             El::TypeTraits<DataType>::One(),
             gradients_mat,
             Ginv,
             El::TypeTraits<DataType>::Zero(),
             Gg);
    auto& Fgrad = this->get_workspace_matrix(std::string("Fgrad_") + mname,
                                             gradients_mat.Height(),
                                             gradients_mat.Width());
    El::Gemm(El::NORMAL,
             El::NORMAL,
             learning_rate_factor,
             Gg,
             kfac_gru_util::is_matrix_height_hidden(matrix_type) ? Ainv_h
                                                                 : Ainv_x,
             El::TypeTraits<DataType>::Zero(),
             Fgrad);

    // Dump matrices for debugging
    if (print_matrix) {
      std::cout << std::endl;
      El::Print(Ginv, std::string("Ginv_") + mname);
      std::cout << std::endl;
      El::Print(gradients_mat, std::string("grad_") + mname);
      std::cout << std::endl;
      El::Print(Fgrad, std::string("Finvgrad_") + mname);
      std::cout << std::endl;
    }

    // Dump L2 norm of matrices
    if (print_matrix_summary) {
      El::Matrix<DataType, Device> weights_mat;
      get_weight_matrix(matrix_type, weights_mat);
      std::ostringstream oss;
      oss << "K-FAC: L2 norm @ " << this->m_layer->get_name() << ": "
          << kfac::get_matrix_stat(
               (const El::Matrix<DataType, Device>&)weights_mat,
               mname.c_str())
          << ", "
          << kfac::get_matrix_stat(Ginv, (std::string("Ginv_") + mname).c_str())
          << ", "
          << kfac::get_matrix_stat(gradients_mat,
                                   (std::string("grad_") + mname).c_str())
          << ", "
          << kfac::get_matrix_stat(Fgrad,
                                   (std::string("Finvgrad_") + mname).c_str())
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
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block_gru<Device>::get_preconditioned_grad_buffers()
{
  std::vector<El::AbstractMatrix<DataType>*> ret;
  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    auto& buf = m_grad_buffer_G[matrix_type];
    get_gradient_buffer(matrix_type, buf);
    buf.Attach(buf.Height() * buf.Width(),
               1,
               buf.Buffer(),
               buf.Height() * buf.Width());
    ret.push_back(&buf);
  }
  return ret;
}

template <El::Device Device>
int kfac_block_gru<Device>::get_inverse_matrices(
  El::Matrix<DataType, Device>& output,
  int offset)
{
  const size_t input_size = get_input_size();
  const size_t hidden_size = get_hidden_size();

  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(m_kronecker_inverse_G[matrix_type]);
    const size_t height = m_kronecker_inverse_G[matrix_type].Height();
    const size_t size_inv = height * height;

    El::copy::util::InterleaveMatrix(
      size_inv,
      1,
      m_kronecker_inverse_G[matrix_type].LockedBuffer(),
      1,
      size_inv,
      output.Buffer(offset, 0),
      1,
      size_inv,
      sync_info);
    offset += size_inv;
  }

  {
    const size_t size_inv = hidden_size * hidden_size;
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(m_kronecker_inverse_A_h);
    El::copy::util::InterleaveMatrix(size_inv,
                                     1,
                                     m_kronecker_inverse_A_h.LockedBuffer(),
                                     1,
                                     size_inv,
                                     output.Buffer(offset, 0),
                                     1,
                                     size_inv,
                                     sync_info);
    offset += size_inv;
  }

  {
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(m_kronecker_inverse_A_x);
    const size_t size_inv = input_size * input_size;
    El::copy::util::InterleaveMatrix(size_inv,
                                     1,
                                     m_kronecker_inverse_A_x.LockedBuffer(),
                                     1,
                                     size_inv,
                                     output.Buffer(offset, 0),
                                     1,
                                     size_inv,
                                     sync_info);
    offset += size_inv;
  }

  return offset;
}

template <El::Device Device>
int kfac_block_gru<Device>::set_inverse_matrices(
  El::Matrix<DataType, Device>& workspace,
  int offset,
  lbann_comm* comm)
{
  const size_t input_size = get_input_size();
  const size_t hidden_size = get_hidden_size();

  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(m_kronecker_inverse_G[matrix_type]);
    const size_t height = kfac_gru_util::is_matrix_height_hidden(matrix_type)
                            ? hidden_size
                            : input_size;
    const size_t size_inv = height * height;

    if (m_kronecker_inverse_G[matrix_type].Height() == 0)
      m_kronecker_inverse_G[matrix_type].Resize(height, height);

    El::copy::util::InterleaveMatrix(
      size_inv,
      1,
      workspace.LockedBuffer(offset, 0),
      1,
      size_inv,
      m_kronecker_inverse_G[matrix_type].Buffer(),
      1,
      size_inv,
      sync_info);
    offset += size_inv;
  }

  {
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(m_kronecker_inverse_A_h);
    const size_t size_inv = hidden_size * hidden_size;
    if (m_kronecker_inverse_A_h.Height() == 0)
      m_kronecker_inverse_A_h.Resize(hidden_size, hidden_size);
    El::copy::util::InterleaveMatrix(size_inv,
                                     1,
                                     workspace.LockedBuffer(offset, 0),
                                     1,
                                     size_inv,
                                     m_kronecker_inverse_A_h.Buffer(),
                                     1,
                                     size_inv,
                                     sync_info);
    offset += size_inv;
  }

  {
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(m_kronecker_inverse_A_x);
    const size_t size_inv = input_size * input_size;
    if (m_kronecker_inverse_A_x.Height() == 0)
      m_kronecker_inverse_A_x.Resize(input_size, input_size);
    El::copy::util::InterleaveMatrix(size_inv,
                                     1,
                                     workspace.LockedBuffer(offset, 0),
                                     1,
                                     size_inv,
                                     m_kronecker_inverse_A_x.Buffer(),
                                     1,
                                     size_inv,
                                     sync_info);
    offset += size_inv;
  }

  return offset;
}

template <El::Device Device>
int kfac_block_gru<Device>::get_inverse_matrices_size(lbann_comm* comm)
{
  const size_t input_size = get_input_size();
  const size_t hidden_size = get_hidden_size();

  int inverse_size = 0;

  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const size_t height = kfac_gru_util::is_matrix_height_hidden(matrix_type)
                            ? hidden_size
                            : input_size;
    const size_t size_inv = height * height;
    inverse_size += size_inv;
  }

  inverse_size += input_size * input_size + hidden_size * hidden_size;

  return inverse_size;
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block_gru<Device>::get_internal_matrix_info() const
{
  std::vector<std::tuple<std::string, size_t, size_t>> list;
  const auto emplace = [&list](const std::string& name,
                               const El::Matrix<DataType, Device>& m) {
    list.emplace_back(name, m.Height(), m.Width());
  };
  const auto emplace_if_available =
    [&emplace](const std::string& name,
               const std::unordered_map<kfac_gru_util::weight_type,
                                        El::Matrix<DataType, Device>>& map,
               const kfac_gru_util::weight_type& matrix_type) {
      auto i = map.find(matrix_type);
      if (i != map.end())
        emplace(name, (*i).second);
    };

  emplace("buf_A_h", m_kronecker_factor_buf_A_h);
  emplace("buf_A_x", m_kronecker_factor_buf_A_x);
  emplace("average_A_h", m_kronecker_average_A_h);
  emplace("average_A_x", m_kronecker_average_A_x);
  emplace("inverse_A_h", m_kronecker_inverse_A_h);
  emplace("inverse_A_x", m_kronecker_inverse_A_x);
  for (auto& matrix_type : kfac_gru_util::LEARNABLE_MATRICES) {
    const auto mname = kfac_gru_util::get_matrix_type_name(matrix_type);
    emplace_if_available("buf_G_" + mname,
                         m_kronecker_factor_buf_G,
                         matrix_type);
    emplace_if_available("average_G_" + mname,
                         m_kronecker_average_G,
                         matrix_type);
    emplace_if_available("inverse_G_" + mname,
                         m_kronecker_inverse_G,
                         matrix_type);
  }
  return list;
}

template <>
void kfac_block_gru<El::Device::CPU>::check_dnn_lib_spec() const
{}

#ifdef LBANN_HAS_GPU
template <>
void kfac_block_gru<El::Device::GPU>::check_dnn_lib_spec() const
{
#if defined(LBANN_GRU_LAYER_CUDNN_SUPPORTED) && defined(LBANN_HAS_DNN_LIB)
  const auto math_type = dnn_lib::get_default_convolution_math_type();
  if (math_type != dnn_lib::DNN_DEFAULT_MATH) {
    std::stringstream ss;
    ss << "The structure of cuDNN's reserve space might not be"
       << " what the GRU K-FAC implementation expects when Tensor Cores are "
          "enabled.";
    LBANN_WARNING(ss.str());
  }
#else  // defined(LBANN_GRU_LAYER_CUDNN_SUPPORTED) && defined(LBANN_HAS_DNN_LIB)
  LBANN_ERROR("cuDNN should be enabled to use K-FAC with GPUs.");
#endif // defined(LBANN_GRU_LAYER_CUDNN_SUPPORTED) && defined(LBANN_HAS_DNN_LIB)
}
#endif // LBANN_HAS_GPU

template <>
void kfac_block_gru<El::Device::CPU>::get_r_i(
  El::Matrix<DataType, El::Device::CPU>& r,
  El::Matrix<DataType, El::Device::CPU>& i,
  const El::Matrix<DataType, El::Device::CPU>& biases_ones,
  const El::Matrix<DataType, El::Device::CPU>& local_inputs,
  const El::Matrix<DataType, El::Device::CPU>& local_outputs,
  const El::Matrix<DataType, El::Device::CPU>& h0,
  const size_t local_batch_size,
  const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const size_t input_size = get_input_size();
  const size_t hidden_size = get_hidden_size();
  const size_t seq_length = get_seq_length();
  El::Matrix<DataType, El::Device::CPU> weights_Wi, weights_Ri, biases_Wi,
    biases_Ri, weights_Wr, weights_Rr, biases_Wr, biases_Rr;
  get_weight_matrix(kfac_gru_util::weight_type::Wi, weights_Wi);
  get_weight_matrix(kfac_gru_util::weight_type::Ri, weights_Ri);
  get_weight_matrix(kfac_gru_util::weight_type::bWi, biases_Wi);
  get_weight_matrix(kfac_gru_util::weight_type::bRi, biases_Ri);
  get_weight_matrix(kfac_gru_util::weight_type::Wr, weights_Wr);
  get_weight_matrix(kfac_gru_util::weight_type::Rr, weights_Rr);
  get_weight_matrix(kfac_gru_util::weight_type::bWr, biases_Wr);
  get_weight_matrix(kfac_gru_util::weight_type::bRr, biases_Rr);
  for (size_t t = 0; t < seq_length; t++) {
    auto i_t = El::View(i, El::ALL, El::IR(t, t + 1));
    auto r_t = El::View(r, El::ALL, El::IR(t, t + 1));
    i_t.Attach(hidden_size, local_batch_size, i_t.Buffer(), hidden_size);
    r_t.Attach(hidden_size, local_batch_size, r_t.Buffer(), hidden_size);
    const auto x_t =
      El::LockedView(local_inputs,
                     El::IR(input_size * t, input_size * (t + 1)),
                     El::ALL);
    const auto hprev_t =
      (t == 0 ? El::LockedView(h0)
              : El::LockedView(local_outputs,
                               El::IR(hidden_size * (t - 1), hidden_size * t),
                               El::ALL));
    kfac_gru_util::gru_gate_forward(weights_Wi,
                                    weights_Ri,
                                    biases_Wi,
                                    biases_Ri,
                                    x_t,
                                    hprev_t,
                                    biases_ones,
                                    i_t);
    kfac_gru_util::gru_gate_forward(weights_Wr,
                                    weights_Rr,
                                    biases_Wr,
                                    biases_Rr,
                                    x_t,
                                    hprev_t,
                                    biases_ones,
                                    r_t);
  }
}

#ifdef LBANN_HAS_GPU
template <>
void kfac_block_gru<El::Device::GPU>::get_r_i(
  El::Matrix<DataType, El::Device::GPU>& r,
  El::Matrix<DataType, El::Device::GPU>& i,
  const El::Matrix<DataType, El::Device::GPU>& biases_ones,
  const El::Matrix<DataType, El::Device::GPU>& local_inputs,
  const El::Matrix<DataType, El::Device::GPU>& local_outputs,
  const El::Matrix<DataType, El::Device::GPU>& h0,
  const size_t local_batch_size,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
#ifndef LBANN_GRU_LAYER_CUDNN_SUPPORTED
  LBANN_ERROR("cuDNN should be enabled to use K-FAC with GPUs.");
#else
  kfac_gru_util::unpack_reserve_space(
    (const DataType*)m_reserve_space_fwd.data(),
    r,
    i,
    get_hidden_size(),
    get_seq_length(),
    local_batch_size,
    sync_info);
#endif // LBANN_GRU_LAYER_CUDNN_SUPPORTED
}
#endif // LBANN_HAS_GPU

template <El::Device Device>
void kfac_block_gru<Device>::get_weight_matrix(
  const kfac_gru_util::weight_type matrix_type,
  El::Matrix<DataType, Device>& view)
{
  const size_t hidden_size = get_hidden_size();
  const auto ids = kfac_gru_util::get_gru_weight_offset(matrix_type);
  auto weight_ptr = dynamic_cast<
    const El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
    &(*this->m_weight_values[ids.first]));
  const auto& weights_mat = El::LockedView(
    weight_ptr->LockedMatrix(),
    El::IR(hidden_size * ids.second, hidden_size * (ids.second + 1)),
    El::ALL);
  El::LockedView(view, weights_mat);
}

template <El::Device Device>
void kfac_block_gru<Device>::get_gradient_matrix(
  const kfac_gru_util::weight_type matrix_type,
  El::Matrix<DataType, Device>& view)
{
  const size_t hidden_size = get_hidden_size();
  const auto ids = kfac_gru_util::get_gru_weight_offset(matrix_type);
  auto& weights = this->m_layer->get_weights(ids.first);
  optimizer* opt = weights.get_optimizer();
  auto* dto = dynamic_cast<data_type_optimizer<DataType>*>(opt);
  auto grad = dto->get_gradient();
  const auto& gradients = grad->LockedMatrix();
  LBANN_ASSERT(static_cast<size_t>(gradients.Height()) == hidden_size * 3);
  const auto gradients_mat = El::LockedView(
    (El::Matrix<DataType, Device>&)gradients,
    El::IR(hidden_size * ids.second, hidden_size * (ids.second + 1)),
    El::ALL);
  El::LockedView(view, gradients_mat);
}

template <El::Device Device>
void kfac_block_gru<Device>::get_gradient_buffer(
  const kfac_gru_util::weight_type matrix_type,
  El::Matrix<DataType, Device>& view)
{
  const size_t hidden_size = get_hidden_size();
  DataType dst_scale = El::TypeTraits<DataType>::Zero(),
           gradient_scale = El::TypeTraits<DataType>::One();
  const auto ids = kfac_gru_util::get_gru_weight_offset(matrix_type);
  auto& weights = this->m_layer->get_weights(ids.first);
  optimizer* opt = weights.get_optimizer();
  auto& grad_buffer =
    opt->get_gradient_buffer(dst_scale, gradient_scale, false).Matrix();
  LBANN_ASSERT(static_cast<size_t>(grad_buffer.Height()) == hidden_size * 3);
  auto grad_buffer_mat =
    El::View((El::Matrix<DataType, Device>&)grad_buffer,
             El::IR(hidden_size * ids.second, hidden_size * (ids.second + 1)),
             El::ALL);
  El::View(view, grad_buffer_mat);
}

std::string kfac_gru_util::get_matrix_type_name(const weight_type& matrix_type)
{
  if (matrix_type == weight_type::Wr)
    return "Wr";
  else if (matrix_type == weight_type::Wi)
    return "Wi";
  else if (matrix_type == weight_type::Wh)
    return "Wh";
  else if (matrix_type == weight_type::Rr)
    return "Rr";
  else if (matrix_type == weight_type::Ri)
    return "Ri";
  else if (matrix_type == weight_type::Rh)
    return "Rh";
  else if (matrix_type == weight_type::bWr)
    return "bWr";
  else if (matrix_type == weight_type::bWi)
    return "bWi";
  else if (matrix_type == weight_type::bWh)
    return "bWh";
  else if (matrix_type == weight_type::bRr)
    return "bRr";
  else if (matrix_type == weight_type::bRi)
    return "bRi";
  else if (matrix_type == weight_type::bRh)
    return "bRh";
  LBANN_ERROR("Invalid matrix type");
}

bool kfac_gru_util::is_matrix_height_hidden(const weight_type& matrix_type)
{
  if (matrix_type == weight_type::Wr || matrix_type == weight_type::Wi ||
      matrix_type == weight_type::Wh)
    return false;
  else if (matrix_type == weight_type::Rr || matrix_type == weight_type::Ri ||
           matrix_type == weight_type::Rh)
    return true;
  LBANN_ERROR("Invalid matrix type");
}

std::pair<int, int>
kfac_gru_util::get_gru_weight_offset(const weight_type matrix_type)
{
  if (matrix_type == weight_type::Wr)
    return std::make_pair<int, int>(0, 0);
  else if (matrix_type == weight_type::Wi)
    return std::make_pair<int, int>(0, 1);
  else if (matrix_type == weight_type::Wh)
    return std::make_pair<int, int>(0, 2);
  else if (matrix_type == weight_type::Rr)
    return std::make_pair<int, int>(1, 0);
  else if (matrix_type == weight_type::Ri)
    return std::make_pair<int, int>(1, 1);
  else if (matrix_type == weight_type::Rh)
    return std::make_pair<int, int>(1, 2);
  else if (matrix_type == weight_type::bWr)
    return std::make_pair<int, int>(2, 0);
  else if (matrix_type == weight_type::bWi)
    return std::make_pair<int, int>(2, 1);
  else if (matrix_type == weight_type::bWh)
    return std::make_pair<int, int>(2, 2);
  else if (matrix_type == weight_type::bRr)
    return std::make_pair<int, int>(3, 0);
  else if (matrix_type == weight_type::bRi)
    return std::make_pair<int, int>(3, 1);
  else if (matrix_type == weight_type::bRh)
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
  const El::SyncInfo<El::Device::CPU>& sync_info)
{
  LBANN_ERROR("This function shouldn't be called because oneDNN does not have "
              "cuDNN's reserve space.");
}

template <El::Device Device>
void kfac_gru_util::gru_gate_forward(
  const El::Matrix<DataType, Device>& W_y,
  const El::Matrix<DataType, Device>& R_y,
  const El::Matrix<DataType, Device>& b_Wy,
  const El::Matrix<DataType, Device>& b_Ry,
  const El::Matrix<DataType, Device>& x_t,
  const El::Matrix<DataType, Device>& hprev_t,
  const El::Matrix<DataType, Device>& biases_ones,
  El::Matrix<DataType, Device>& y_t)
{
  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           W_y,
           x_t,
           El::TypeTraits<DataType>::Zero(),
           y_t);
  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           R_y,
           hprev_t,
           El::TypeTraits<DataType>::One(),
           y_t);
  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           b_Wy,
           biases_ones,
           El::TypeTraits<DataType>::One(),
           y_t);
  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           b_Ry,
           biases_ones,
           El::TypeTraits<DataType>::One(),
           y_t);
#pragma omp parallel for
  for (El::AbstractMatrix<DataType>::size_type j = 0;
       j < y_t.Height() * y_t.Width();
       j++)
    y_t.Buffer()[j] = sigmoid(y_t.Buffer()[j]);
}

template <>
void kfac_gru_util::get_g(const El::Matrix<DataType, El::Device::CPU>& h,
                          const El::Matrix<DataType, El::Device::CPU>& h0,
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
                          const size_t hidden_size,
                          const size_t seq_length,
                          const size_t local_batch_size,
                          const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const DataType* h_buf = h.LockedBuffer();
  const DataType* h0_buf = h0.LockedBuffer();
  const DataType* dh_buf = dh.LockedBuffer();
  const DataType* hfc_buf = hfc.LockedBuffer();
  const DataType* r_buf = r.LockedBuffer();
  const DataType* i_buf = i.LockedBuffer();
  DataType* g_Wr_buf = g_Wr.Buffer();
  DataType* g_Wi_buf = g_Wi.Buffer();
  DataType* g_Wh_buf = g_Wh.Buffer();
  DataType* g_Rr_buf = g_Rr.Buffer();
  DataType* g_Ri_buf = g_Ri.Buffer();
  DataType* g_Rh_buf = g_Rh.Buffer();

#pragma omp parallel for
  for (size_t gid = 0; gid < hidden_size * seq_length * local_batch_size;
       gid++) {
    const size_t i_hidden = gid % hidden_size;
    const size_t i_seq = (gid / hidden_size) % seq_length;
    const size_t i_batch = gid / hidden_size / seq_length;
    const size_t i_hsl =
      i_hidden + i_seq * hidden_size + i_batch * hidden_size * seq_length;
    const size_t i_hl = i_hidden + i_batch * hidden_size;
    const size_t i_hls =
      i_hidden + i_batch * hidden_size + i_seq * hidden_size * local_batch_size;

    const DataType r_val = r_buf[i_hls];
    const DataType i_val = i_buf[i_hls];
    const DataType dh_val = dh_buf[i_hsl];

    const DataType hprev =
      (i_seq == 0 ? h0_buf[i_hl] : h_buf[i_hsl - hidden_size]);
    const DataType hd = (h_buf[i_hsl] - i_val * hprev) / (1.0 - i_val);
    const DataType dhdhd = 1.0 - i_val;
    const DataType dhdr = dhdhd * tanh_deriv(tanh_inv(hd)) * hfc_buf[i_hls];
    const DataType dhdi = hprev - hd;

    const DataType drdg_Rr = sigmoid_deriv(sigmoid_inv(r_val));
    const DataType didg_Ri = sigmoid_deriv(sigmoid_inv(i_val));
    const DataType dhddg_Wh = tanh_deriv(tanh_inv(hd));
    const DataType dhddg_Rh = dhddg_Wh * r_val;

    g_Wr_buf[i_hls] = g_Rr_buf[i_hls] = dh_val * dhdr * drdg_Rr;
    g_Wi_buf[i_hls] = g_Ri_buf[i_hls] = dh_val * dhdi * didg_Ri;
    g_Wh_buf[i_hls] = dh_val * dhdhd * dhddg_Wh;
    g_Rh_buf[i_hls] = dh_val * dhdhd * dhddg_Rh;
  }
}

template class kfac_block_gru<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac_block_gru<El::Device::GPU>;
#endif // LBANN_HAS_GPU

//////////////////////////////////////////////////////////////////////////////////
////  Sub-Grid Parallelism Functions
//////////////////////////////////////////////////////////////////////////////////

template <El::Device Device>
void kfac_block_gru<Device>::start_communication_forward_end(lbann_comm* comm)
{

  int num_local_activations = 3, num_weights = 4;

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto parent_h0 = this->m_layer->get_parent_layers()[1];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_parent_h0 =
    dynamic_cast<const data_type_layer<DataType>&>(*parent_h0);
  const auto& dtl =
    dynamic_cast<const data_type_layer<DataType>&>(*this->m_layer);
  const auto& local_inputs = dtl_parent.get_activations();
  const auto& h0 = dtl_parent_h0.get_activations();
  const auto& local_outputs = dtl.get_activations();

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
          input = std::make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            this->m_subset_matrix.push_back(
              std::make_unique<
                El::
                  DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
                comm->get_subset_grid(),
                0));
        }
        else { // Model-Parallel Layout
          input = std::make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            LBANN_ERROR("Async prgoress is not supported for model-parallel "
                        "layer layout in sub-grid parallelism");
        }
      }

      for (auto& weight : this->m_weight_values) {
        weight = std::make_unique<
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

      const auto h0_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(h0));
      auto local_activations1 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_parent_local_activations[1]));
      auto subset1 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[1]));

      const auto local_outputs_vc = dynamic_cast<
        const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(local_outputs));
      auto local_activations2 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_parent_local_activations[2]));
      auto subset2 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[2]));

      kfac::TranslateBetweenGridsVCAsync(*local_inputs_vc,
                                         *local_activations0,
                                         *subset0,
                                         this->m_requests_forward_end);
      kfac::TranslateBetweenGridsVCAsync(*h0_vc,
                                         *local_activations1,
                                         *subset1,
                                         this->m_requests_forward_end);
      kfac::TranslateBetweenGridsVCAsync(*local_outputs_vc,
                                         *local_activations2,
                                         *subset2,
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
        kfac::TranslateBetweenGridsSTARAsync(*weight_ptr,
                                             *weight_input_ptr,
                                             this->m_requests_forward_end);
        iter++;
      }
    }
    else {
      El::Copy(local_inputs, *this->m_parent_local_activations[0]);
      El::Copy(h0, *this->m_parent_local_activations[1]);
      El::Copy(local_outputs, *this->m_parent_local_activations[2]);

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

    for (auto& weight : this->m_weight_values) {
      weight = std::make_unique<
        El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
        comm->get_trainer_grid(),
        0);
    }

    El::LockedView(*(this->m_parent_local_activations[0]), local_inputs);
    El::LockedView(*(this->m_parent_local_activations[1]), h0);
    El::LockedView(*(this->m_parent_local_activations[2]), local_outputs);

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
void kfac_block_gru<Device>::end_communication_forward_end(lbann_comm* comm)
{
  if ((comm->get_grid_type() == GridType::SECONDARY_GRID or
       comm->get_grid_type() == GridType::PRIMARY_GRID) and
      comm->enable_subgrid_async_communication() and
      comm->get_KFAC_subgrid_create_two_models() == false) {
    int num_local_activations = 3;
    auto primary_grid_ranks = comm->get_primary_grid_ranks();
    auto secondary_grid_ranks = comm->get_secondary_grid_ranks();

    for (auto& req : this->m_requests_forward_end) {
      ::Al::Wait<kfac::BackendT>(req);
    }
    for (auto& req : m_requests_workspace) {
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
    m_requests_workspace.clear();
  }
}

template <El::Device Device>
void kfac_block_gru<Device>::start_communication_backward_end(lbann_comm* comm)
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
          error = std::make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
          if (comm->enable_subgrid_async_communication())
            this->m_subset_matrix.push_back(
              std::make_unique<
                El::
                  DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
                comm->get_subset_grid(),
                0));
        }
        else {
          error = std::make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
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
      auto subset1 = dynamic_cast<
        El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>*>(
        &(*this->m_subset_matrix[3]));

      kfac::TranslateBetweenGridsVCAsync(*local_errors_vc,
                                         *local_errors0,
                                         *subset1,
                                         this->m_requests_backward_end);
    } // Async progress
  }

  if (comm->get_grid_type() == GridType::NO_GRID or
      comm->get_KFAC_subgrid_create_two_models()) {
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
    El::LockedView(*(this->m_child_local_errors[0]), local_errors);
  }
}

template <El::Device Device>
void kfac_block_gru<Device>::end_communication_backward_end(lbann_comm* comm)
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
        &(*this->m_subset_matrix[3]));
      kfac::TranslateBetweenGridsVC(*subset1, *local_errors0);
    }
    this->m_requests_backward_end.clear();
  }
}

template <El::Device Device>
void kfac_block_gru<Device>::initialize_activations_and_errors(
  lbann_comm* comm,
  int num_local_activations,
  int num_local_errors,
  int num_weights)
{

  num_local_activations = 3;
  num_local_errors = 1;
  num_weights = 4;

  const auto parent = this->m_layer->get_parent_layers()[0];
  const auto parent_h0 = this->m_layer->get_parent_layers()[1];
  const auto child = this->m_layer->get_child_layers()[0];
  const auto& dtl_parent =
    dynamic_cast<const data_type_layer<DataType>&>(*parent);
  const auto& dtl_parent_h0 =
    dynamic_cast<const data_type_layer<DataType>&>(*parent_h0);
  const auto& dtl_child =
    dynamic_cast<const data_type_layer<DataType>&>(*child);
  const auto& dtl =
    dynamic_cast<const data_type_layer<DataType>&>(*this->m_layer);
  const auto& local_inputs = dtl_parent.get_activations();
  const auto& h0 = dtl_parent_h0.get_activations();
  const auto& local_outputs = dtl.get_activations();
  const auto& local_errors = dtl_child.get_error_signals();

  if (comm->get_KFAC_subgrid_create_two_models() or
      comm->get_grid_type() == GridType::NO_GRID) {
    this->m_parent_local_activations.resize(num_local_activations);
    this->m_child_local_errors.resize(num_local_errors);
    this->m_weight_values.resize(num_weights);
  }
  else {
    if (this->m_parent_local_activations.size() == 0) {
      // Resize vectors
      this->m_parent_local_activations.resize(num_local_activations);
      this->m_child_local_errors.resize(num_local_errors);
      this->m_weight_values.resize(num_weights);

      // Initialize Dist Matrices
      for (auto& input : this->m_parent_local_activations) {
        if (dtl_parent.get_data_layout() == data_layout::DATA_PARALLEL)
          input = std::make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
        else
          input = std::make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
      }

      for (auto& error : this->m_child_local_errors) {

        if (dtl_child.get_data_layout() == data_layout::DATA_PARALLEL)
          error = std::make_unique<
            El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
        else
          error = std::make_unique<
            El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, Device>>(
            comm->get_secondary_grid(),
            0);
      }

      for (auto& weight : this->m_weight_values) {
        weight = std::make_unique<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
          comm->get_secondary_grid(),
          0);
      }
    }

    El::Copy(local_inputs, *this->m_parent_local_activations[0]);
    El::Copy(h0, *this->m_parent_local_activations[1]);
    El::Copy(local_outputs, *this->m_parent_local_activations[2]);
    El::Copy(local_errors, *this->m_child_local_errors[0]);
  }

  if (comm->get_grid_type() == GridType::NO_GRID or
      comm->get_KFAC_subgrid_create_two_models()) {

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
    for (auto& weight : this->m_weight_values) {
      weight = std::make_unique<
        El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>>(
        comm->get_trainer_grid(),
        0);
    }

    El::LockedView(*(this->m_parent_local_activations[0]), local_inputs);
    El::LockedView(*(this->m_parent_local_activations[1]), h0);
    El::LockedView(*(this->m_parent_local_activations[2]), local_outputs);
    El::LockedView(*(this->m_child_local_errors[0]), local_errors);

    for (int i = 0; i < num_weights; ++i) {
      auto& weights = this->m_layer->get_weights(i);
      const auto& dtw = dynamic_cast<data_type_weights<DataType>*>(&weights);
      El::LockedView(*this->m_weight_values[i], dtw->get_values_sharded());
    }
  }
}

} // namespace lbann
