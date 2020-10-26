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

#ifdef LBANN_HAS_GPU

/** @brief Gets the inverse matrix of A. **/
void get_matrix_inverse(
    El::Matrix<DataType, El::Device::GPU>& Ainv,
    El::Matrix<DataType, El::Device::GPU>& Linv,
    const El::Matrix<DataType, El::Device::GPU>& A,
    bool report_time,
    DataType damping,
    DataType damping_bn_err,
    bool is_bn,
    const cudaStream_t& stream);

/** @brief Gets statistics of a given matrix. **/
std::string get_matrix_stat(
    const El::Matrix<DataType, El::Device::GPU>& X,
    const char *name);

/** @brief Perform all-reduce on the lower triangular of a symmetric matrix. **/
void allreduce_lower_tri(
    El::Matrix<DataType, El::Device::GPU>& A,
    El::Matrix<DataType, El::Device::GPU>& AL,
    lbann_comm *comm,
    const cudaStream_t& stream);

/** @brief Get whether a global buffer is needed. **/
bool is_reduce_scatter_buffer_required(kfac_reduce_scatter_mode mode);

/** @brief Perform reduce-scatter on one or more blocks. **/
void reduce_scatter_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, El::Device::GPU>& global_buffer,
    lbann_comm *comm,
    kfac_reduce_scatter_mode mode);

/** @brief Get whether local and global buffers are needed. **/
std::pair<bool, bool> is_allgather_buffer_required(kfac_allgather_mode mode);

/** @brief Perform reduce-scatter on one or more blocks. **/
void allgather_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, El::Device::GPU>& send_buffer,
    El::Matrix<DataType, El::Device::GPU>& recv_buffer,
    lbann_comm *comm,
    kfac_allgather_mode mode);

/** @brief Add the damping value to the diagonal elements of A. **/
template <typename TensorDataType>
void add_to_diagonal(
    TensorDataType * __restrict__ A,
    size_t height,
    TensorDataType value,
    TensorDataType value_bn_err,
    bool is_bn,
    const cudaStream_t& stream);

/** @brief Fill the upper trianglar with the lower trianglar. **/
template <typename TensorDataType>
void fill_upper_tri(
    TensorDataType * __restrict__ A,
    size_t height,
    const cudaStream_t& stream);

/** @brief Update a Kronecker factor matrix using decay.
 *
 * Aave = Aave * decay + A * (1-decay) **/
template <typename TensorDataType>
void update_kronecker_average(
    TensorDataType * __restrict__ Aave,
    const TensorDataType * __restrict__ A,
    size_t count,
    double decay,
    const cudaStream_t& stream);

/** @brief Substitute the identity matrix.
 *  TODO: Replace with El::Identity<El::Device::GPU>
 *   once it gets supported. **/
template <typename TensorDataType>
void identity(
    TensorDataType * __restrict__ A,
    size_t height,
    const cudaStream_t& stream);

/** @brief Pack the lower triangular of a symmetric matrix. **/
template <typename TensorDataType>
void pack_lower_tri(
    TensorDataType * __restrict__ L,
    const TensorDataType * __restrict__ A,
    size_t height,
    const cudaStream_t& stream);

/** @brief Unpack the lower triangular of a symmetric matrix. **/
template <typename TensorDataType>
void unpack_lower_tri(
    TensorDataType * __restrict__ A,
    const TensorDataType * __restrict__ L,
    size_t height,
    const cudaStream_t& stream);

#endif // LBANN_HAS_GPU

} // namespace kfac_util
} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_UTIL_HPP_INCLUDED
