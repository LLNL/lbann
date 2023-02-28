////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_UTIL_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_UTIL_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/gru.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block.hpp"

// Forward declarations
namespace lbann {
class KFAC;
template <El::Device Device>
class kfac_block;
}

namespace lbann {

namespace kfac {

#if defined AL_HAS_NCCL
using BackendT = ::Al::NCCLBackend;
#elif defined AL_HAS_HOST_TRANSFER
using BackendT = ::Al::HostTransferBackend;
#else
using BackendT = ::Al::MPIBackend;
#endif

using ReqT = typename BackendT::req_type;

enum class kfac_inverse_strategy {
  ALL,  // Apply round-robin assingment to all of the layers. may cause load imbalance.
  EACH, // Apply round-robin assingment to every type of layers. may
  // not work well for small networks.
  ROOT, // Use only the root GPU. This is only for testing.
};

enum class kfac_reduce_scatter_mode {
  ALLREDUCE, // Use lbann_comm::allreduce
  REDUCE_SCATTER, // Use El::ReduceScatter
  REDUCE, // Use El::Reduce for each block
};

enum class kfac_allgather_mode {
  ALLREDUCE, // Use lbann_comm::allreduce
  ALLGATHER, // Use El::ReduceScatter
  BROADCAST // Use El::Broadcast for each block
};

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

/** @brief Gets the inverse matrix of A using Eigen Value Decomposition. **/
template <El::Device Device>
void get_matrix_inverse_eigen(
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

/** @brief Perform allgather for inverse matrices **/
template <El::Device Device>
void allgather_inverse_matrices(
    const std::vector<std::shared_ptr<kfac_block<Device>>>& blocks,
    El::Matrix<DataType, Device>& global_buffer,
    lbann_comm *comm);

/** @brief Perform allgather for inverse matrices size**/
template <El::Device Device>
void allgather_inverse_matrices_sizes(
    const std::vector<std::shared_ptr<kfac_block<Device>>>& blocks,
    El::Matrix<double, El::Device::CPU>& global_buffer,
    lbann_comm *comm);

/** @brief Add the damping value to the diagonal elements of A. **/
template <El::Device Device>
void add_to_diagonal(
    El::Matrix<DataType, Device>& A,
    DataType value,
    DataType value_bn_err,
    bool is_bn,
    const El::SyncInfo<Device>& sync_info);

/** @brief Add the damping value to the diagonal elements of A from B. **/
template <El::Device Device>
void make_diagonal(
    El::Matrix<DataType, Device>& A,
    El::Matrix<DataType, Device>& B,
    DataType value,
    DataType value_bn_err,
    bool is_bn,
    const El::SyncInfo<Device>& sync_info);

/** @brief Add the damping value to the diagonal elements of A. **/
template <El::Device Device>
void get_matrix_entrywise_inverse(
    El::Matrix<DataType, Device>& input,
    El::Matrix<DataType, Device>& output,
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

template<typename T, El::Device Device>
void TranslateBetweenGridsVCAsync
( const El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& A,
  El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& B,
  El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& subset,
  std::vector<ReqT>& Requests);

template<typename T, El::Device Device>
void TranslateBetweenGridsVCAsyncDirect
( const El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& A,
  El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& B,
  El::Int featureSize,
  El::Int currentBatchSize,
  std::vector<ReqT>& Requests);

template<typename T, El::Device Device>
void TranslateBetweenGridsSTARAsync
(const El::DistMatrix<T,El::STAR,El::STAR,El::ELEMENT,Device>& A,
  El::DistMatrix<T,El::STAR,El::STAR,El::ELEMENT,Device>& B,
  std::vector<ReqT>& Requests);

template<typename T, El::Device Device>
void TranslateBetweenGridsKFACAsync
(const El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& A,
  El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& B,
  std::vector<ReqT>& Requests);

template<typename T, El::Device Device>
void TranslateBetweenGridsVC
(El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device> const& A,
  El::DistMatrix<T,El::STAR,El::VC,El::ELEMENT,Device>& B);


} // namespace kfac
} // namespace lbann

#endif  // LBANN_EXECUTION_ALGORITHMS_KFAC_KFAC_UTIL_HPP_INCLUDED
