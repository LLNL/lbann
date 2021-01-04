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
// kfac_util .hpp .cpp - Utility (GPU) functions for the K-FAC callback.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_UTIL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_UTIL_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/callbacks/kfac/kfac.hpp"

namespace lbann {
namespace callback {
namespace kfac_util {

/** @brief Gets the inverse matrix of A. **/
template <El::Device Device>
void get_matrix_inverse(
    El::AbstractMatrix<DataType>& Ainv,
    El::AbstractMatrix<DataType>& Linv,
    const El::AbstractMatrix<DataType>& A,
    bool report_time,
    DataType damping,
    DataType damping_bn_err,
    bool is_bn,
    const El::SyncInfo<Device>& sync_info);

/** @brief Gets statistics of a given matrix. **/
template <El::Device Device>
std::string get_matrix_stat(
    const El::Matrix<DataType, Device>& X,
    const char *name);

/** @brief Perform all-reduce on the lower triangular of a symmetric matrix. **/
template <El::Device Device>
void allreduce_lower_tri(
    El::AbstractMatrix<DataType>& A,
    El::AbstractMatrix<DataType>& AL,
    lbann_comm *comm,
    const El::SyncInfo<Device>& sync_info);

/** @brief Get whether a global buffer is needed. **/
bool is_reduce_scatter_buffer_required(kfac_reduce_scatter_mode mode);

/** @brief Perform reduce-scatter on one or more blocks. **/
template <El::Device Device>
void reduce_scatter_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, Device>& global_buffer,
    lbann_comm *comm,
    kfac_reduce_scatter_mode mode);

/** @brief Get whether local and global buffers are needed. **/
std::pair<bool, bool> is_allgather_buffer_required(kfac_allgather_mode mode);

/** @brief Perform reduce-scatter on one or more blocks. **/
template <El::Device Device>
void allgather_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, Device>& send_buffer,
    El::Matrix<DataType, Device>& recv_buffer,
    lbann_comm *comm,
    kfac_allgather_mode mode);

/** @brief Add the damping value to the diagonal elements of A. **/
template <El::Device Device>
void add_to_diagonal(
    El::Matrix<DataType, Device>& A,
    DataType value,
    DataType value_bn_err,
    bool is_bn,
    const El::SyncInfo<Device>& sync_info);

/** @brief Fill the upper trianglar with the lower trianglar. **/
template <El::Device Device>
void fill_upper_tri(
    El::Matrix<DataType, Device>& A,
    const El::SyncInfo<Device>& sync_info);

/** @brief Update a Kronecker factor matrix using decay.
 *
 * Aave = Aave * decay + A * (1-decay) **/
template <El::Device Device>
void update_kronecker_average(
    El::Matrix<DataType, Device>& Aave,
    const El::Matrix<DataType, Device>& A,
    size_t count,
    double decay,
    const El::SyncInfo<Device>& sync_info);

/** @brief Substitute the identity matrix.
 *  TODO: Replace with El::Identity<El::Device::GPU>
 *   once it gets supported. **/
template <El::Device Device>
void identity(
    El::Matrix<DataType, Device>& A,
    const El::SyncInfo<Device>& sync_info);

/** @brief Pack the lower triangular of a symmetric matrix. **/
template <El::Device Device>
void pack_lower_tri(
    El::Matrix<DataType, Device>& L,
    const El::Matrix<DataType, Device>& A,
    const El::SyncInfo<Device>& sync_info);

/** @brief Unpack the lower triangular of a symmetric matrix. **/
template <El::Device Device>
void unpack_lower_tri(
    El::Matrix<DataType, Device>& A,
    const El::Matrix<DataType, Device>& L,
    const El::SyncInfo<Device>& sync_info);

/** @brief Wrappers to call Aluminum with suitable communication types. **/
template <El::Device Device>
void reduce_block_device(
    El::Matrix<DataType, Device>& block,
    const size_t count,
    const size_t root,
    const El::mpi::Comm& trainer_comm,
    const El::SyncInfo<Device>& sync_info);
template <El::Device Device>
void reduce_scatter_v_blocks_device(
    El::Matrix<DataType, Device>& blocks,
    const std::vector<size_t>& recv_sizes,
    const El::mpi::Comm& trainer_comm,
    const El::SyncInfo<Device>& sync_info);
template <El::Device Device>
void allgather_v_blocks_device(
    const El::Matrix<DataType, Device>& send_block,
    El::Matrix<DataType, Device>& recv_blocks,
    const std::vector<size_t>& recv_sizes,
    const std::vector<size_t>& recv_offsets,
    const El::mpi::Comm& trainer_comm,
    const El::SyncInfo<Device>& sync_info);

} // namespace kfac_util
} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_UTIL_HPP_INCLUDED
