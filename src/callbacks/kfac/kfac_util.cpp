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

#include "lbann/callbacks/kfac/kfac_util.hpp"
#include "lbann/base.hpp"
#include "lbann/utils/timer.hpp"

#include <cassert>
#include <core/imports/mpi.hpp>
#include <iomanip>
#include <iterator>

namespace lbann {
namespace callback {
namespace kfac_util {
namespace {
std::vector<int> intify_size_t_vector(std::vector<size_t> const& in_sizes)
{
  std::vector<int> out;
  out.reserve(in_sizes.size());
  std::transform(cbegin(in_sizes),
                 cend(in_sizes),
                 std::back_inserter(out),
                 [](size_t const& x) {
                   int out = static_cast<int>(x);
                   if ((out < 0) || (static_cast<size_t>(out) != x))
                     throw std::runtime_error(
                       "MPI size not in dynamic range of int");
                   return out;
                 });
  return out;
}

template <El::Device Device>
void reduce_block_device(El::Matrix<DataType, Device>& block,
                         const size_t count,
                         const size_t root,
                         const El::mpi::Comm& trainer_comm,
                         const El::SyncInfo<Device>& sync_info)
{
  El::mpi::Reduce(block.Buffer(),
                  count,
                  El::mpi::SUM,
                  root,
                  trainer_comm,
                  sync_info);
}

template <El::Device Device>
void reduce_scatter_v_blocks_device(El::Matrix<DataType, Device>& blocks,
                                    const std::vector<size_t>& recv_sizes,
                                    const El::mpi::Comm& trainer_comm,
                                    const El::SyncInfo<Device>& sync_info)
{
  El::mpi::ReduceScatter(blocks.LockedBuffer(),
                         blocks.Buffer(),
                         intify_size_t_vector(recv_sizes).data(),
                         trainer_comm,
                         sync_info);
}

template <El::Device Device>
void allgather_v_blocks_device(const El::Matrix<DataType, Device>& send_block,
                               El::Matrix<DataType, Device>& recv_blocks,
                               const std::vector<size_t>& recv_sizes,
                               const std::vector<size_t>& recv_offsets,
                               const El::mpi::Comm& trainer_comm,
                               const El::SyncInfo<Device>& sync_info)
{
  auto const int_sizes = intify_size_t_vector(recv_sizes);
  auto const int_offsets = intify_size_t_vector(recv_offsets);
  El::mpi::AllGather(send_block.LockedBuffer(),
                     int_sizes[trainer_comm.Rank()],
                     recv_blocks.Buffer(),
                     int_sizes.data(),
                     int_offsets.data(),
                     trainer_comm,
                     sync_info);
}

#if defined LBANN_HAS_GPU && defined LBANN_HAS_ALUMINUM
template <>
void reduce_block_device<El::Device::GPU>(
  El::Matrix<DataType, El::Device::GPU>& block,
  const size_t count,
  const size_t root,
  const El::mpi::Comm& trainer_comm,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
  ::Al::Reduce<::Al::NCCLBackend>(
    block.Buffer(),
    count,
    ::Al::ReductionOperator::sum,
    root,
    trainer_comm.template GetComm<::Al::NCCLBackend>(sync_info));
}

template <>
void reduce_scatter_v_blocks_device(
  El::Matrix<DataType, El::Device::GPU>& blocks,
  const std::vector<size_t>& recv_sizes,
  const El::mpi::Comm& trainer_comm,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
  ::Al::Reduce_scatterv<::Al::NCCLBackend>(
    blocks.Buffer(),
    recv_sizes,
    ::Al::ReductionOperator::sum,
    trainer_comm.template GetComm<::Al::NCCLBackend>(sync_info));
}

template <>
void allgather_v_blocks_device(
  const El::Matrix<DataType, El::Device::GPU>& send_block,
  El::Matrix<DataType, El::Device::GPU>& recv_blocks,
  const std::vector<size_t>& recv_sizes,
  const std::vector<size_t>& recv_offsets,
  const El::mpi::Comm& trainer_comm,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
  ::Al::Allgatherv<::Al::NCCLBackend>(
    send_block.LockedBuffer(),
    recv_blocks.Buffer(),
    recv_sizes,
    recv_offsets,
    trainer_comm.template GetComm<::Al::NCCLBackend>(sync_info));
}
#endif // defined LBANN_HAS_GPU && defined LBANN_HAS_ALUMINUM

} // namespace

template <El::Device Device>
void get_matrix_inverse(
    El::AbstractMatrix<DataType>& Ainv,
    El::AbstractMatrix<DataType>& Linv,
    const El::AbstractMatrix<DataType>& A,
    const bool report_time,
    const DataType damping,
    const DataType damping_bn_err,
    const bool is_bn,
    const El::SyncInfo<Device>& sync_info) {
  assert(A.Height() == A.Width());
  assert(Ainv.Height() == A.Height());
  assert(Ainv.Width() == A.Height());
  El::Copy(A, Ainv);

  const double t_start = get_time();

  if(damping > 0 || damping_bn_err > 0)
    add_to_diagonal<Device>(
        Ainv,
        damping, damping_bn_err,
        is_bn,
        sync_info);

  const double t_damping = get_time();

  const auto uplo = El::UpperOrLowerNS::LOWER;
  El::Cholesky(
      uplo,
      (El::AbstractMatrix<DataType> &) Ainv);

  const double t_spotrf = get_time();

  assert(Linv.Height() == Ainv.Height());
  assert(Linv.Width() == Ainv.Height());
  identity<Device>(Linv, sync_info);
  El::Trsm(
      El::LeftOrRightNS::LEFT,
      uplo,
      El::OrientationNS::NORMAL,
      El::UnitOrNonUnitNS::NON_UNIT,
      El::TypeTraits<DataType>::One(),
      (const El::AbstractMatrix<DataType> &) Ainv,
      (El::AbstractMatrix<DataType> &) Linv,
      true);
  El::Gemm(
      El::TRANSPOSE, El::NORMAL,
      El::TypeTraits<DataType>::One(), Linv, Linv,
      El::TypeTraits<DataType>::Zero(), Ainv);

  const double t_spotri = get_time();

  // TRSM+GEMM is equivalent to POTRI+fill_upper_tri.
  // fill_upper_tri(Ainv.Buffer(), Ainv.Height());

  const double t_fill = get_time();

  if(report_time) {
    std::cout << "K-FAC callback: get_matrix_inverse of"
              << " " << A.Height() << "x" << A.Width()
              << " using Hydrogen"
              << " (damping=" << damping << "): "
              << " t_damping=" << (t_damping-t_start)
              << ", t_spotrf=" << (t_spotrf-t_damping)
              << ", t_spotri=" << (t_spotri-t_spotrf)
              << ", t_fill=" << (t_fill-t_spotri)
              << std::endl;
  }

  // TODO: Check whether this is actually needed.
  El::Synchronize(sync_info);
}

template <El::Device Device>
std::string get_matrix_stat(const El::Matrix<DataType, Device>& X,
                            const char *name) {
  El::Matrix<DataType> XCPU(X);
  const auto nrm2 = El::Nrm2(El::Reshape(XCPU.Height()*XCPU.Width(), 1, XCPU));
  std::ostringstream oss;
  oss << name
      << "("
      << X.Height()
      << "x"
      << X.Width()
      << ")="
      << std::setprecision(2)
      << std::scientific
      << nrm2;
  return oss.str();
}

template <El::Device Device>
void allreduce_lower_tri(El::AbstractMatrix<DataType>& A,
                         El::AbstractMatrix<DataType>& AL,
                         lbann_comm *comm,
                         const El::SyncInfo<Device>& sync_info) {
  assert(A.Height() == A.Width());
  assert(AL.Height() == A.Height()*(A.Height()+1)/2);
  assert(AL.Width() == 1);
  pack_lower_tri<Device>(AL, A, sync_info);
  comm->allreduce(AL, comm->get_trainer_comm());
  unpack_lower_tri<Device>(A, AL, sync_info);
}

bool is_reduce_scatter_buffer_required(const kfac_reduce_scatter_mode mode) {
  if(mode == kfac_reduce_scatter_mode::ALLREDUCE)
    return true;
  else if(mode == kfac_reduce_scatter_mode::REDUCE_SCATTER)
    return true;
  else if(mode == kfac_reduce_scatter_mode::REDUCE)
    return false;
  LBANN_ERROR("Invalid reduce-scatter mode");
}

template <El::Device Device>
void reduce_scatter_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, Device>& global_buffer,
    lbann_comm *comm,
    const kfac_reduce_scatter_mode mode) {

  if (mode == kfac_reduce_scatter_mode::REDUCE) {
    for (auto& [block_root, block_mat] : blocks) {
      auto& blk = dynamic_cast<El::Matrix<DataType, Device>&>(*block_mat);
      reduce_block_device(blk,
                          blk.Height(),
                          block_root,
                          comm->get_trainer_comm(),
                          El::SyncInfoFromMatrix(global_buffer));
    }
    return;
  }

  // Sort blocks so that received blocks per process become contiguous.
  std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> sorted_blocks(blocks.size());
  std::copy(blocks.begin(), blocks.end(), sorted_blocks.begin());
  if(mode == kfac_reduce_scatter_mode::REDUCE_SCATTER)
    std::stable_sort(
        sorted_blocks.begin(), sorted_blocks.end(),
        [](const std::pair<size_t, El::AbstractMatrix<DataType>*>& lhs,
           const std::pair<size_t, El::AbstractMatrix<DataType>*>& rhs) {
          return lhs.first < rhs.first;
        });

  // Copy blocks to the send buffer.
  {
    size_t offset = 0;
    for(auto& block : sorted_blocks) {
      auto view = El::View(global_buffer, El::IR(offset, offset+block.second->Height()), El::ALL);
      El::Copy(*block.second, view);
      offset += block.second->Height();
    }
  }

  if(mode == kfac_reduce_scatter_mode::ALLREDUCE) {
    comm->allreduce(
        (El::AbstractMatrix<DataType>&) global_buffer,
        comm->get_trainer_comm());
  } else {
    std::vector<size_t> recv_sizes;
    recv_sizes.resize(comm->get_procs_per_trainer());
    for(auto& block : sorted_blocks)
      recv_sizes[block.first] += block.second->Height();
    reduce_scatter_v_blocks_device(
        global_buffer,
        recv_sizes,
        comm->get_trainer_comm(),
        El::SyncInfoFromMatrix(global_buffer));
  }

  // Apply aggregated Kronecker factros to each block.
  {
    size_t offset = 0;
    for(auto& block : sorted_blocks) {
      const bool is_my_block = (block.first == (size_t) comm->get_rank_in_trainer());
      if(is_my_block) {
        const auto view = El::LockedView(global_buffer, El::IR(offset, offset+block.second->Height()), El::ALL);
        El::Copy(view, *block.second);
      }
      if(mode == kfac_reduce_scatter_mode::ALLREDUCE || is_my_block) {
        offset += block.second->Height();
      }
    }
  }
}

/** @brief Get whether local and global buffers are needed. **/
std::pair<bool, bool> is_allgather_buffer_required(const kfac_allgather_mode mode) {
  if(mode == kfac_allgather_mode::ALLREDUCE)
    return std::make_pair(false, true);
  else if(mode == kfac_allgather_mode::ALLGATHER)
    return std::make_pair(true, true);
  else if(mode == kfac_allgather_mode::BROADCAST)
    return std::make_pair(false, false);
  LBANN_ERROR("Invalid allgather mode");
}

template <El::Device Device>
void allgather_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, Device>& local_buffer,
    El::Matrix<DataType, Device>& global_buffer,
    lbann_comm *comm,
    const kfac_allgather_mode mode) {

  if(mode == kfac_allgather_mode::BROADCAST) {
    for(auto& block : blocks)
      El::Broadcast(
          *block.second, comm->get_trainer_comm(),
          block.first);
    return;
  }

  // Sort blocks so that received blocks per process become
  // contiguous.
  std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> sorted_blocks(blocks.size());
  std::copy(blocks.begin(), blocks.end(), sorted_blocks.begin());
  if(mode == kfac_allgather_mode::ALLGATHER)
    std::stable_sort(
        sorted_blocks.begin(), sorted_blocks.end(),
        [](const std::pair<size_t, El::AbstractMatrix<DataType>*>& lhs,
           const std::pair<size_t, El::AbstractMatrix<DataType>*>& rhs) {
          return lhs.first < rhs.first;
        });

  // Copy blocks to the send buffer.
  {
    El::Matrix<DataType, Device>& buffer =
        (mode == kfac_allgather_mode::ALLREDUCE ? global_buffer : local_buffer);
    if(mode == kfac_allgather_mode::ALLREDUCE)
      El::Zeros(buffer, buffer.Height(), buffer.Width());
    size_t offset = 0;
    for(auto& block : sorted_blocks) {
      const bool is_my_block = (block.first == (size_t) comm->get_rank_in_trainer());
      if(is_my_block) {
        auto view = El::View(buffer, El::IR(offset, offset+block.second->Height()), El::ALL);
        El::Copy(*block.second, view);
      }
      if(is_my_block || mode == kfac_allgather_mode::ALLREDUCE)
        offset += block.second->Height();
    }
  }

  if(mode == kfac_allgather_mode::ALLREDUCE) {
    comm->allreduce(
        (El::AbstractMatrix<DataType>&) global_buffer,
        comm->get_trainer_comm());
  }
  else {
    std::vector<size_t> recv_sizes;
    recv_sizes.resize(comm->get_procs_per_trainer());
    for(auto& block : sorted_blocks)
      recv_sizes[block.first] += block.second->Height();
    std::vector<size_t> recv_offsets;
    recv_offsets.resize(recv_sizes.size()+1);
    for(size_t i = 0; i <= recv_sizes.size(); i++)
      recv_offsets[i] = (i > 0 ? recv_offsets[i-1]+recv_sizes[i-1] : 0);
    allgather_v_blocks_device(
        local_buffer,
        global_buffer,
        recv_sizes,
        recv_offsets,
        comm->get_trainer_comm(),
        El::SyncInfoFromMatrix(local_buffer));
  }

  // Copy blocks from the buffer.
  {
    size_t offset = 0;
    for(auto& block : sorted_blocks) {
      if(block.first != (size_t) comm->get_rank_in_trainer()) {
        const auto view = El::LockedView(global_buffer, El::IR(offset, offset+block.second->Height()), El::ALL);
        El::Copy(view, *block.second);
      }
      offset += block.second->Height();
    }
  }
}

template <>
void add_to_diagonal(
    El::Matrix<DataType, El::Device::CPU>& A,
    const DataType damping,
    const DataType damping_bn_err,
    const bool is_bn,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  const auto height = A.Height();
#pragma omp parallel for
  for(int i = 0; i < height; i++)
    A(i, i) += (is_bn && i >= A.Height()/2 ? damping_bn_err : damping);
}

template <>
void fill_upper_tri(
    El::Matrix<DataType, El::Device::CPU>& A,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  const auto height = A.Height();
#pragma omp parallel for
  for(int col = 0; col < height; col++)
    for(int row = 0; row < height; row++)
      if(row < col)
        A(row, col) += A(col, row);
}

// TODO: Do not define count but use A.Height()*A.Height()
template <>
void update_kronecker_average(
    El::Matrix<DataType, El::Device::CPU>& Aave,
    const El::Matrix<DataType, El::Device::CPU>& A,
    const size_t count, const double decay,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  assert(count == (size_t) (A.Height()*A.Height()));
  const auto height = A.Height();
#pragma omp parallel for
  for(int col = 0; col < height; col++)
    for(int row = 0; row < height; row++)
      Aave(row, col) = Aave(row, col)*decay + A(row, col)*(1.0-decay);
}

template <>
void identity(
    El::Matrix<DataType, El::Device::CPU>& A,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  El::Identity(A, A.Height(), A.Height());
}

template <>
void pack_lower_tri(
    El::Matrix<DataType, El::Device::CPU>& L,
    const El::Matrix<DataType, El::Device::CPU>& A,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  const auto height = A.Height();
#pragma omp parallel for
  for(int col = 0; col < height; col++)
    for(int row = 0; row < height; row++)
      if(row >= col)
        L(row+(2*height-(col-1))*col/2-col, 0) = A(row+col*height, 0);
}

template <>
void unpack_lower_tri(
    El::Matrix<DataType, El::Device::CPU>& A,
    const El::Matrix<DataType, El::Device::CPU>& L,
    const El::SyncInfo<El::Device::CPU>& sync_info) {
  const auto height = A.Height();
#pragma omp parallel for
  for(int col = 0; col < height; col++)
    for(int row = 0; row < height; row++)
      if(row >= col)
        A(row+col*height, 0)
            = A(col+row*height, 0)
            = L(row+(2*height-(col-1))*col/2-col, 0);
}

#define PROTO_DEVICE(T, Device)                 \
  template void get_matrix_inverse(             \
      El::AbstractMatrix<T>& Ainv,              \
      El::AbstractMatrix<T>& Linv,              \
      const El::AbstractMatrix<T>& A,           \
      bool report_time,                         \
      T damping,                                \
      T damping_bn_err,                         \
      bool is_bn,                               \
      const El::SyncInfo<Device>& sync_info);   \
  template std::string get_matrix_stat(         \
      const El::Matrix<T, Device>& X,           \
      const char *name);                        \
  template void allreduce_lower_tri(            \
      El::AbstractMatrix<T>& A,                 \
      El::AbstractMatrix<T>& AL,                \
      lbann_comm *comm,                         \
      const El::SyncInfo<Device>& sync_info);   \
  template void reduce_scatter_blocks(          \
      const std::vector<std::pair<size_t,       \
      El::AbstractMatrix<T>*>>& blocks,         \
      El::Matrix<T, Device>& global_buffer,     \
      lbann_comm *comm,                         \
      const kfac_reduce_scatter_mode mode);     \
  template void allgather_blocks(               \
      const std::vector<std::pair<size_t,       \
      El::AbstractMatrix<T>*>>& blocks,         \
      El::Matrix<T, Device>& local_buffer,      \
      El::Matrix<T, Device>& global_buffer,     \
      lbann_comm *comm,                         \
      const kfac_allgather_mode mode);

PROTO_DEVICE(DataType, El::Device::CPU);
#ifdef LBANN_HAS_GPU
PROTO_DEVICE(DataType, El::Device::GPU);
#endif // LBANN_HAS_GPU

} // namespace kfac_util
} // namespace callback
} // namespace lbann
