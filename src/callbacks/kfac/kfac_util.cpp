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
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/timer.hpp"

#include <cassert>
#include <iomanip>

namespace lbann {
namespace callback {
namespace kfac_util {

#ifdef LBANN_HAS_GPU

void get_matrix_inverse(
    El::Matrix<DataType, El::Device::GPU>& Ainv,
    El::Matrix<DataType, El::Device::GPU>& Linv,
    const El::Matrix<DataType, El::Device::GPU>& A,
    const bool report_time,
    const DataType damping,
    const DataType damping_bn_err,
    const bool is_bn,
    const cudaStream_t& stream) {
  assert(A.Height() == A.Width());
  assert(Ainv.Height() == A.Height());
  assert(Ainv.Width() == A.Height());
  El::Copy(A, Ainv);

  const double t_start = get_time();

  if(damping > 0 || damping_bn_err > 0)
    add_to_diagonal(
        Ainv.Buffer(), Ainv.Height(),
        damping, damping_bn_err,
        is_bn,
        stream);

  const double t_damping = get_time();

  const auto uplo = El::UpperOrLowerNS::LOWER;
  El::Cholesky(
      uplo,
      (El::AbstractMatrix<DataType> &) Ainv);

  const double t_spotrf = get_time();

  assert(Linv.Height() == Ainv.Height());
  assert(Linv.Width() == Ainv.Height());
  identity(Linv.Buffer(), Linv.Height(), stream);
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
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

std::string get_matrix_stat(const El::Matrix<DataType, El::Device::GPU>& X,
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

void allreduce_lower_tri(El::Matrix<DataType, El::Device::GPU>& A,
                         El::Matrix<DataType, El::Device::GPU>& AL,
                         lbann_comm *comm,
                         const cudaStream_t& stream) {
  assert(A.Height() == A.Width());
  assert(AL.Height() == A.Height()*(A.Height()+1)/2);
  assert(AL.Width() == 1);
  pack_lower_tri(AL.Buffer(), A.LockedBuffer(), A.Height(), stream);
  comm->allreduce((El::AbstractMatrix<DataType>&) AL,
                  comm->get_trainer_comm());
  unpack_lower_tri(A.Buffer(), AL.Buffer(), A.Height(), stream);
}

void reduce_scatter_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, El::Device::GPU>& global_buffer,
    lbann_comm *comm,
    const kfac_reduce_scatter_mode mode) {

  if(mode == kfac_reduce_scatter_mode::REDUCE) {
    for(auto& block : blocks)
      ::Al::Reduce<::Al::NCCLBackend>(
           block.second->Buffer(),
           block.second->Height(),
           ::Al::ReductionOperator::sum,
           block.first,
           comm->get_trainer_comm().template GetComm<::Al::NCCLBackend>(El::SyncInfoFromMatrix(global_buffer)));
    return;
  }

  // Sort blocks so that received blocks per process become contiguous.
  std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> sorted_blocks(blocks.size());
  std::copy(blocks.begin(), blocks.end(), sorted_blocks.begin());
  if(mode == kfac_reduce_scatter_mode::REDUCE_SCATTER)
    std::sort(sorted_blocks.begin(), sorted_blocks.end(),
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
    ::Al::Reduce_scatterv<::Al::NCCLBackend>(
         global_buffer.Buffer(),
         recv_sizes,
         ::Al::ReductionOperator::sum,
         comm->get_trainer_comm().template GetComm<::Al::NCCLBackend>(El::SyncInfoFromMatrix(global_buffer)));
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

void allgather_blocks(
    const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
    El::Matrix<DataType, El::Device::GPU>& local_buffer,
    El::Matrix<DataType, El::Device::GPU>& global_buffer,
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
    std::sort(sorted_blocks.begin(), sorted_blocks.end(),
              [](const std::pair<size_t, El::AbstractMatrix<DataType>*>& lhs,
                 const std::pair<size_t, El::AbstractMatrix<DataType>*>& rhs) {
                return lhs.first < rhs.first;
              });

  // Copy blocks to the send buffer.
  {
    El::Matrix<DataType, El::Device::GPU>& buffer =
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
  } else {
    std::vector<size_t> recv_sizes;
    recv_sizes.resize(comm->get_procs_per_trainer());
    for(auto& block : sorted_blocks)
      recv_sizes[block.first] += block.second->Height();
    std::vector<size_t> recv_offsets;
    recv_offsets.resize(recv_sizes.size()+1);
    for(size_t i = 0; i <= recv_sizes.size(); i++)
      recv_offsets[i] = (i > 0 ? recv_offsets[i-1]+recv_sizes[i-1] : 0);

    ::Al::Allgatherv<::Al::NCCLBackend>(
         local_buffer.LockedBuffer(), global_buffer.Buffer(),
         recv_sizes, recv_offsets,
         comm->get_trainer_comm().template GetComm<::Al::NCCLBackend>(El::SyncInfoFromMatrix(local_buffer)));
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

#endif // LBANN_HAS_GPU

} // namespace kfac_util
} // namespace callback
} // namespace lbann
