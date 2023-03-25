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

#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"

namespace lbann {

template <El::Device Device>
El::Matrix<DataType, Device>&
kfac_block<Device>::get_workspace_matrix(const std::string& key,
                                         const size_t height,
                                         const size_t width)
{
  return m_context->get_workspace_matrix(get_name() + " " + key, height, width);
}

template <>
El::SyncInfo<El::Device::CPU> kfac_block<El::Device::CPU>::get_sync_info()
{
  return El::SyncInfo<El::Device::CPU>{};
}

#ifdef LBANN_HAS_GPU
template <>
El::SyncInfo<El::Device::GPU> kfac_block<El::Device::GPU>::get_sync_info()
{
  return El::gpu::DefaultSyncInfo();
}
#endif // LBANN_HAS_GPU

template <El::Device Device>
void kfac_block<Device>::compute_local_kronecker_factors(lbann_comm* comm,
                                                         bool print_matrix,
                                                         bool print_matrix_summary)
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block<Device>::get_local_kronecker_buffers()
{
  LBANN_ERROR("this function should be called via a sub-class.");
}


template <El::Device Device>
void kfac_block<Device>::update_kronecker_average(lbann_comm* comm,
                                                  DataType kronecker_decay,
                                                  bool print_matrix,
                                                  bool print_matrix_summary)
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template <El::Device Device>
void kfac_block<Device>::update_kronecker_inverse(lbann_comm* comm,
                                                  bool use_pi,
                                                  DataType damping_act,
                                                  DataType damping_err,
                                                  DataType learning_rate_factor,
                                                  bool use_eigen_decomposition,
                                                  bool print_matrix,
                                                  bool print_matrix_summary,
                                                  bool print_time)
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template <El::Device Device>
void kfac_block<Device>::compute_preconditioned_gradients(lbann_comm* comm,
                                                          DataType learning_rate_factor,
                                                          bool print_matrix,
                                                          bool print_matrix_summary,
                                                          bool print_time)
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template <El::Device Device>
void kfac_block<Device>::initialize_activations_and_errors(lbann_comm* comm,
                                                           int num_local_activations,
                                                           int num_local_errors,
                                                           int num_weights)
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template <El::Device Device>
const std::vector<El::AbstractMatrix<DataType>*>
kfac_block<Device>::get_preconditioned_grad_buffers()
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template <El::Device Device>
std::vector<std::tuple<std::string, size_t, size_t>>
kfac_block<Device>::get_internal_matrix_info() const
{
  LBANN_ERROR("this function should be called via a sub-class.");
}

template class kfac_block<El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class kfac_block<El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
