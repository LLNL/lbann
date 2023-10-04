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

#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/base.hpp"
#include "lbann/utils/timer.hpp"

#include "lbann/utils/entrywise_operator.hpp"
#include "lbann/utils/gpu/gpu_lib.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include <cassert>
#include <core/imports/mpi.hpp>
#include <iomanip>
#include <iterator>

namespace lbann {
namespace kfac {
namespace {

std::vector<int> intify_size_t_vector(std::vector<size_t> const& in_sizes)
{
  std::vector<int> out;
  out.reserve(in_sizes.size());
  std::transform(cbegin(in_sizes),
                 cend(in_sizes),
                 std::back_inserter(out),
                 [](size_t const& x) {
                   int val = static_cast<int>(x);
                   if ((val < 0) || (static_cast<size_t>(val) != x))
                     throw std::runtime_error(
                       "MPI size not in dynamic range of int");
                   return val;
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
void reduce_block_device(El::Matrix<DataType, El::Device::GPU>& block,
                         const size_t count,
                         const size_t root,
                         const El::mpi::Comm& trainer_comm,
                         const El::SyncInfo<El::Device::GPU>& sync_info)
{
  ::Al::Reduce<BackendT>(block.Buffer(),
                         count,
                         ::Al::ReductionOperator::sum,
                         root,
                         trainer_comm.template GetComm<BackendT>(sync_info));
}

template <>
void reduce_scatter_v_blocks_device(
  El::Matrix<DataType, El::Device::GPU>& blocks,
  const std::vector<size_t>& recv_sizes,
  const El::mpi::Comm& trainer_comm,
  const El::SyncInfo<El::Device::GPU>& sync_info)
{
  ::Al::Reduce_scatterv<BackendT>(
    blocks.Buffer(),
    recv_sizes,
    ::Al::ReductionOperator::sum,
    trainer_comm.template GetComm<BackendT>(sync_info));
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
  ::Al::Allgatherv<BackendT>(
    send_block.LockedBuffer(),
    recv_blocks.Buffer(),
    recv_sizes,
    recv_offsets,
    trainer_comm.template GetComm<BackendT>(sync_info));
}
#endif // defined LBANN_HAS_GPU && defined LBANN_HAS_ALUMINUM

} // namespace

/** Entry-wise operator for CPU */
template <typename TensorDataType>
struct inverse_op_cpu
{
  inline TensorDataType operator()(const TensorDataType& x) const
  {
    return TensorDataType(1) / (x);
  }
};

template <>
void get_matrix_entrywise_inverse(
  El::Matrix<DataType, El::Device::CPU>& input,
  El::Matrix<DataType, El::Device::CPU>& output,
  const El::SyncInfo<El::Device::CPU>& sync_info)
{

  apply_entrywise_unary_operator<inverse_op_cpu, DataType>(input, output);
}

// #ifdef LBANN_HAS_GPU
// /** Entry-wise operator. */
// template <typename TensorDataType>
// struct inverse_op_gpu {
//   inline __device__ TensorDataType operator()(const TensorDataType& x) const
//   {
//     return TensorDataType(1)/x;
//   }
// };

// template<>
// void get_matrix_entrywise_inverse(
//     El::Matrix<DataType,El::Device::GPU>& input,
//     El::Matrix<DataType,El::Device::GPU>& output){

//   ::lbann::gpu_lib::apply_entrywise_unary_operator<inverse_op_gpu,
//   DataType>(input,output);
// }
// #endif //LBANN_HAS_GPU

template <El::Device Device>
void get_matrix_inverse(El::AbstractMatrix<DataType>& Ainv,
                        El::AbstractMatrix<DataType>& Linv,
                        const El::AbstractMatrix<DataType>& A,
                        const bool report_time,
                        const DataType damping,
                        const DataType damping_bn_err,
                        const bool is_bn,
                        const El::SyncInfo<Device>& sync_info)
{
  assert(A.Height() == A.Width());
  assert(Ainv.Height() == A.Height());
  assert(Ainv.Width() == A.Height());
  El::Copy(A, Ainv);

  const double t_start = get_time();

  if (damping > 0 || damping_bn_err > 0)
    add_to_diagonal<Device>(Ainv, damping, damping_bn_err, is_bn, sync_info);

  const double t_damping = get_time();

  const auto uplo = El::UpperOrLowerNS::LOWER;
  El::Cholesky(uplo, (El::AbstractMatrix<DataType>&)Ainv);

  const double t_spotrf = get_time();

  assert(Linv.Height() == Ainv.Height());
  assert(Linv.Width() == Ainv.Height());
  identity<Device>(Linv, sync_info);
  El::Trsm(El::LeftOrRightNS::LEFT,
           uplo,
           El::OrientationNS::NORMAL,
           El::UnitOrNonUnitNS::NON_UNIT,
           El::TypeTraits<DataType>::One(),
           (const El::AbstractMatrix<DataType>&)Ainv,
           (El::AbstractMatrix<DataType>&)Linv,
           true);
  El::Gemm(El::TRANSPOSE,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           Linv,
           Linv,
           El::TypeTraits<DataType>::Zero(),
           Ainv);

  const double t_spotri = get_time();

  // TRSM+GEMM is equivalent to POTRI+fill_upper_tri.
  // fill_upper_tri(Ainv.Buffer(), Ainv.Height());

  const double t_fill = get_time();

  if (report_time) {
    std::cout << "K-FAC: get_matrix_inverse of"
              << " " << A.Height() << "x" << A.Width() << " using Hydrogen"
              << " (damping=" << damping << "): "
              << " t_damping=" << (t_damping - t_start)
              << ", t_spotrf=" << (t_spotrf - t_damping)
              << ", t_spotri=" << (t_spotri - t_spotrf)
              << ", t_fill=" << (t_fill - t_spotri) << std::endl;
  }

  // TODO: Check whether this is actually needed.
  El::Synchronize(sync_info);
}

template <El::Device Device>
void get_matrix_inverse_eigen(El::AbstractMatrix<DataType>& Ainv,
                              El::AbstractMatrix<DataType>& Linv,
                              const El::AbstractMatrix<DataType>& A,
                              const bool report_time,
                              const DataType damping,
                              const DataType damping_bn_err,
                              const bool is_bn,
                              const El::SyncInfo<Device>& sync_info)
{

  assert(A.Height() == A.Width());
  assert(Ainv.Height() == A.Height());
  assert(Ainv.Width() == A.Height());
  El::Copy(A, Ainv);

  El::HermitianEigCtrl<DataType> ctrl;
  // BVE FIXME unused variable
  // typedef El::Base<DataType> Real;
  El::Matrix<DataType, Device> w;
  El::Matrix<DataType, Device> Q, diag(Ainv.Height(), Ainv.Width()),
    diag_out(Ainv.Height(), Ainv.Width());
  identity<Device>(diag, sync_info);
  Zeros(Q, Ainv.Height(), Ainv.Width());

  const auto uplo = El::UpperOrLowerNS::LOWER;

  El::HermitianEig(uplo, (El::Matrix<DataType, Device>&)Ainv, w, Q, ctrl);

  const double t_start = get_time();

  const double t_damping = get_time();

  const double t_spotrf = get_time();

  // assert(Linv.Height() == Ainv.Height());
  // assert(Linv.Width() == Ainv.Height());
  El::Zeros(diag, Ainv.Height(), Ainv.Width());

  make_diagonal<Device>(diag, w, damping, damping_bn_err, is_bn, sync_info);

  El::Gemm(El::NORMAL,
           El::NORMAL,
           El::TypeTraits<DataType>::One(),
           Q,
           diag,
           El::TypeTraits<DataType>::Zero(),
           diag_out);

  El::Gemm(El::NORMAL,
           El::TRANSPOSE,
           El::TypeTraits<DataType>::One(),
           diag_out,
           Q,
           El::TypeTraits<DataType>::Zero(),
           Ainv);

  const double t_spotri = get_time();

  // TRSM+GEMM is equivalent to POTRI+fill_upper_tri.
  // fill_upper_tri(Ainv.Buffer(), Ainv.Height());

  const double t_fill = get_time();

  if (report_time) {
    std::cout << "K-FAC: get_matrix_inverse of"
              << " " << A.Height() << "x" << A.Width() << " using Hydrogen"
              << " (damping=" << damping << "): "
              << " t_damping=" << (t_damping - t_start)
              << ", t_spotrf=" << (t_spotrf - t_damping)
              << ", t_spotri=" << (t_spotri - t_spotrf)
              << ", t_fill=" << (t_fill - t_spotri) << std::endl;
  }

  // TODO: Check whether this is actually needed.
  El::Synchronize(sync_info);
}

template <El::Device Device>
std::string get_matrix_stat(const El::Matrix<DataType, Device>& X,
                            const char* name)
{
  El::Matrix<DataType> XCPU(X);
  const auto nrm2 =
    El::Nrm2(El::Reshape(XCPU.Height() * XCPU.Width(), 1, XCPU));
  std::ostringstream oss;
  oss << name << "(" << X.Height() << "x" << X.Width()
      << ")=" << std::setprecision(2) << std::scientific << nrm2;
  return oss.str();
}

template <El::Device Device>
void allreduce_lower_tri(El::AbstractMatrix<DataType>& A,
                         El::AbstractMatrix<DataType>& AL,
                         lbann_comm* comm,
                         const El::SyncInfo<Device>& sync_info)
{
  assert(A.Height() == A.Width());
  assert(AL.Height() == A.Height() * (A.Height() + 1) / 2);
  assert(AL.Width() == 1);
  pack_lower_tri<Device>(AL, A, sync_info);
  comm->allreduce(AL, comm->get_KFAC_comm());
  unpack_lower_tri<Device>(A, AL, sync_info);
}

bool is_reduce_scatter_buffer_required(const kfac_reduce_scatter_mode mode)
{
  if (mode == kfac_reduce_scatter_mode::ALLREDUCE)
    return true;
  else if (mode == kfac_reduce_scatter_mode::REDUCE_SCATTER)
    return true;
  else if (mode == kfac_reduce_scatter_mode::REDUCE)
    return false;
  LBANN_ERROR("Invalid reduce-scatter mode");
}

template <El::Device Device>
void reduce_scatter_blocks(
  const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
  El::Matrix<DataType, Device>& global_buffer,
  lbann_comm* comm,
  const kfac_reduce_scatter_mode mode)
{

  if (mode == kfac_reduce_scatter_mode::REDUCE) {
    for (auto& [block_root, block_mat] : blocks) {
      auto& blk = dynamic_cast<El::Matrix<DataType, Device>&>(*block_mat);
      reduce_block_device(blk,
                          blk.Height(),
                          block_root,
                          comm->get_KFAC_comm(),
                          El::SyncInfoFromMatrix(global_buffer));
    }
    return;
  }

  // Sort blocks so that received blocks per process become contiguous.
  std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> sorted_blocks(
    blocks.size());
  std::copy(blocks.begin(), blocks.end(), sorted_blocks.begin());
  if (mode == kfac_reduce_scatter_mode::REDUCE_SCATTER)
    std::stable_sort(
      sorted_blocks.begin(),
      sorted_blocks.end(),
      [](const std::pair<size_t, El::AbstractMatrix<DataType>*>& lhs,
         const std::pair<size_t, El::AbstractMatrix<DataType>*>& rhs) {
        return lhs.first < rhs.first;
      });

  // Copy blocks to the send buffer.
  {
    size_t offset = 0;
    for (auto& block : sorted_blocks) {
      auto view = El::View(global_buffer,
                           El::IR(offset, offset + block.second->Height()),
                           El::ALL);
      El::Copy(*block.second, view);
      offset += block.second->Height();
    }
  }

  if (mode == kfac_reduce_scatter_mode::ALLREDUCE) {
    comm->allreduce((El::AbstractMatrix<DataType>&)global_buffer,
                    comm->get_KFAC_comm());
  }
  else {
    std::vector<size_t> recv_sizes;
    recv_sizes.resize(comm->get_procs_per_trainer());
    for (auto& block : sorted_blocks)
      recv_sizes[block.first] += block.second->Height();
    reduce_scatter_v_blocks_device(global_buffer,
                                   recv_sizes,
                                   comm->get_KFAC_comm(),
                                   El::SyncInfoFromMatrix(global_buffer));
  }

  // Apply aggregated Kronecker factros to each block.
  {
    size_t offset = 0;
    for (auto& block : sorted_blocks) {
      const bool is_my_block =
        (block.first == (size_t)comm->get_rank_in_trainer());
      if (is_my_block) {
        const auto view =
          El::LockedView(global_buffer,
                         El::IR(offset, offset + block.second->Height()),
                         El::ALL);
        El::Copy(view, *block.second);
      }
      if (mode == kfac_reduce_scatter_mode::ALLREDUCE || is_my_block) {
        offset += block.second->Height();
      }
    }
  }
}

/** @brief Get whether local and global buffers are needed. **/
std::pair<bool, bool>
is_allgather_buffer_required(const kfac_allgather_mode mode)
{
  if (mode == kfac_allgather_mode::ALLREDUCE)
    return std::make_pair(false, true);
  else if (mode == kfac_allgather_mode::ALLGATHER)
    return std::make_pair(true, true);
  else if (mode == kfac_allgather_mode::BROADCAST)
    return std::make_pair(false, false);
  LBANN_ERROR("Invalid allgather mode");
}

template <El::Device Device>
void allgather_blocks(
  const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
  El::Matrix<DataType, Device>& local_buffer,
  El::Matrix<DataType, Device>& global_buffer,
  lbann_comm* comm,
  const kfac_allgather_mode mode)
{
  if (mode == kfac_allgather_mode::BROADCAST) {
    for (auto& block : blocks)
      El::Broadcast(*block.second, comm->get_KFAC_comm(), block.first);
    return;
  }

  // Sort blocks so that received blocks per process become
  // contiguous.
  std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> sorted_blocks(
    blocks.size());
  std::copy(blocks.begin(), blocks.end(), sorted_blocks.begin());
  if (mode == kfac_allgather_mode::ALLGATHER)
    std::stable_sort(
      sorted_blocks.begin(),
      sorted_blocks.end(),
      [](const std::pair<size_t, El::AbstractMatrix<DataType>*>& lhs,
         const std::pair<size_t, El::AbstractMatrix<DataType>*>& rhs) {
        return lhs.first < rhs.first;
      });

  // Copy blocks to the send buffer.
  {
    El::Matrix<DataType, Device>& buffer =
      (mode == kfac_allgather_mode::ALLREDUCE ? global_buffer : local_buffer);
    if (mode == kfac_allgather_mode::ALLREDUCE)
      El::Zeros(buffer, buffer.Height(), buffer.Width());
    size_t offset = 0;
    for (auto& block : sorted_blocks) {
      const bool is_my_block =
        (block.first == (size_t)comm->get_rank_in_trainer());
      if (is_my_block) {
        auto view = El::View(buffer,
                             El::IR(offset, offset + block.second->Height()),
                             El::ALL);
        El::Copy(*block.second, view);
      }
      if (is_my_block || mode == kfac_allgather_mode::ALLREDUCE)
        offset += block.second->Height();
    }
  }

  if (mode == kfac_allgather_mode::ALLREDUCE) {
    comm->allreduce((El::AbstractMatrix<DataType>&)global_buffer,
                    comm->get_KFAC_comm());
  }
  else {
    std::vector<size_t> recv_sizes;
    recv_sizes.resize(comm->get_procs_per_trainer());
    for (auto& block : sorted_blocks)
      recv_sizes[block.first] += block.second->Height();
    std::vector<size_t> recv_offsets;
    recv_offsets.resize(recv_sizes.size() + 1);
    for (size_t i = 0; i <= recv_sizes.size(); i++)
      recv_offsets[i] = (i > 0 ? recv_offsets[i - 1] + recv_sizes[i - 1] : 0);
    allgather_v_blocks_device(local_buffer,
                              global_buffer,
                              recv_sizes,
                              recv_offsets,
                              comm->get_KFAC_comm(),
                              El::SyncInfoFromMatrix(local_buffer));
  }

  // Copy blocks from the buffer.
  {
    size_t offset = 0;
    for (auto& block : sorted_blocks) {
      if (block.first != (size_t)comm->get_rank_in_trainer()) {
        const auto view =
          El::LockedView(global_buffer,
                         El::IR(offset, offset + block.second->Height()),
                         El::ALL);
        El::Copy(view, *block.second);
      }
      offset += block.second->Height();
    }
  }
}

template <El::Device Device>
void allgather_inverse_matrices_sizes(
  const std::vector<std::shared_ptr<kfac_block<Device>>>& blocks,
  El::Matrix<double, El::Device::CPU>& global_buffer,
  lbann_comm* comm)
{

  const size_t num_blocks = blocks.size();
  global_buffer.Resize(num_blocks, 4);
  El::Zeros(global_buffer, global_buffer.Height(), global_buffer.Width());

  int iter = 0;
  for (auto& block : blocks) {
    const bool is_my_block =
      (block->get_inverse_proc_rank() == (size_t)comm->get_rank_in_trainer());
    if (is_my_block) {
      auto inverse_size = block->get_inverse_matrices_size_vector(comm);

      for (int i = 0; i < 4; ++i)
        global_buffer(iter, i) = inverse_size[i];
    }
    iter++;
  }

  comm->allreduce((El::AbstractMatrix<double>&)global_buffer,
                  comm->get_KFAC_comm());
}

template <El::Device Device>
void allgather_inverse_matrices(
  const std::vector<std::shared_ptr<kfac_block<Device>>>& blocks,
  El::Matrix<DataType, Device>& global_buffer,
  lbann_comm* comm)
{

  {
    El::Zeros(global_buffer, global_buffer.Height(), global_buffer.Width());
    size_t offset = 0;
    for (auto& block : blocks) {
      const bool is_my_block =
        (block->get_inverse_proc_rank() == (size_t)comm->get_rank_in_trainer());
      if (is_my_block) {
        offset = block->get_inverse_matrices(global_buffer, offset);
      }
      else {
        offset += block->get_inverse_matrices_size(comm);
      }
    }
  }

  comm->allreduce((El::AbstractMatrix<DataType>&)global_buffer,
                  comm->get_KFAC_comm());
  {
    size_t offset = 0;
    for (auto& block : blocks) {
      offset = block->set_inverse_matrices(global_buffer, offset, comm);
    }
  }
}

namespace {
#ifdef LBANN_HAS_GPU
static constexpr auto KFACDevice = El::Device::GPU;
#else
static constexpr auto KFACDevice = El::Device::CPU;
#endif // LBANN_HAS_GPU
} // namespace

template <typename T, El::Device Device>
void TranslateBetweenGridsVC(
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device> const& A,
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B)
{
  LBANN_ERROR("TranslateBetweenGridsVC function is not implemented for this "
              "configuration");
}

template <>
void TranslateBetweenGridsVC(
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice> const& A,
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& B)
{
  int m = A.Height();
  int n = A.Width();
  const int mLocA = A.LocalHeight();
  const int nLocA = A.LocalWidth();
  El::mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  El::mpi::Group owningGroupA = A.Grid().OwningGroup();

  // Compute the number of process rows and columns that each process
  // needs to send to.
  int colStrideA = A.ColStride();
  int rowStrideA = A.RowStride();
  int colAlignA = A.ColAlign();
  const bool inAGrid = A.Participating();

  B.Resize(m, n);
  const int colStrideB = B.ColStride();
  const int rowStrideB = B.RowStride();
  const int colRankB = B.ColRank();
  const int colAlignB = B.ColAlign();
  const bool inBGrid = B.Participating();

  El::SyncInfo<KFACDevice> syncInfoA = El::SyncInfoFromMatrix(A.LockedMatrix());
  El::SyncInfo<KFACDevice> syncInfoB = El::SyncInfoFromMatrix(B.LockedMatrix());

  const int rowGCD = El::GCD(rowStrideB, rowStrideA);
  const int rowLCM = rowStrideB * rowStrideA / rowGCD;
  const int numRowSends = rowLCM / rowStrideA;
  const int numRowRecvs = rowLCM / rowStrideB;
  const int myRankViewing = El::mpi::Rank(viewingCommB);
  const int rankBRecv = El::Mod(B.Grid().VCRank(), rowStrideA);

  // Setup for receiving data in B
  const int sendColOffset = colAlignA;
  const int recvColOffset = El::Mod(colAlignB, colStrideB);
  const int colShift = El::Mod(colRankB - recvColOffset, colStrideB);
  const int numInB = B.Grid().Rank();

  const int firstSendRow = El::Mod(colShift + sendColOffset, colStrideA);

  // Recv data
  // For now, simply receive sequentially. Until we switch to
  // nonblocking recv's, we won't be using much of the
  // recvBuf
  int sendRow = firstSendRow;

  if (!inBGrid && !inAGrid)
    return;

  const int maxSendSize = (n / (rowStrideA * numRowSends) + 1) * (m);

  // Translate the ranks from A's VC communicator to B's viewing so that
  // we can match send/recv communicators. Since A's VC communicator is not
  // necessarily defined on every process, we instead work with A's owning
  // group and account for row-major ordering if necessary.
  const int sizeA = A.Grid().Size();
  std::vector<int> rankMap(sizeA), ranks(sizeA);
  if (A.Grid().Order() == El::COLUMN_MAJOR) {
    for (int j = 0; j < sizeA; ++j)
      ranks[j] = j;
  }
  else {
    // The (i,j) = i + j*colStrideA rank in the column-major ordering is
    // equal to the j + i*rowStrideA rank in a row-major ordering.
    // Since we desire rankMap[i+j*colStrideA] to correspond to process
    // (i,j) in A's grid's rank in this viewing group, ranks[i+j*colStrideA]
    // should correspond to process (i,j) in A's owning group. Since the
    // owning group is ordered row-major in this case, its rank is
    // j+i*rowStrideA. Note that setting
    // ranks[j+i*rowStrideA] = i+j*colStrideA is *NOT* valid.
    for (int i = 0; i < colStrideA; ++i)
      for (int j = 0; j < rowStrideA; ++j)
        ranks[i + j * colStrideA] = j + i * rowStrideA;
  }
  El::mpi::Translate(owningGroupA,
                     sizeA,
                     ranks.data(),
                     viewingCommB,
                     rankMap.data());

  El::simple_buffer<DataType, KFACDevice> send_buf(inAGrid ? maxSendSize : 0,
                                                   syncInfoA);
  El::simple_buffer<DataType, KFACDevice> recv_buf(inBGrid ? maxSendSize : 0,
                                                   syncInfoB);

  DataType* sendBuf = send_buf.data();
  DataType* recvBuf = recv_buf.data();

  // Ranks of processes to send data.
  // Key: Process rank
  // value: column offset
  std::map<int, int> sendProcessRanks;
  std::map<int, int> recvProcessRanks;
  for (int rowSend = 0; rowSend < numRowSends; rowSend++) {
    const int recvVCRank =
      El::Mod(A.Grid().Rank() + rowSend * rowStrideA, rowStrideB);
    const int recvViewingRank = B.Grid().VCToViewing(recvVCRank);
    sendProcessRanks.insert(std::pair<int, int>(recvViewingRank, rowSend));
  }

  sendRow = 0;

  for (int rowRecv = 0; rowRecv < numRowRecvs; rowRecv++) {
    const int sendVCRank = El::Mod((sendRow + rankBRecv), rowStrideA);
    recvProcessRanks.insert(std::pair<int, int>(rankMap[sendVCRank], rowRecv));
    sendRow = El::Mod(sendRow + rowStrideB, rowStrideA);
  }

  // Checking if process are in both A and B grids
  for (int rowSend = 0; rowSend < numRowSends; rowSend++) {
    const int recvVCRank =
      El::Mod(A.Grid().Rank() + rowSend * rowStrideA, rowStrideB);
    const int recvViewingRank = B.Grid().VCToViewing(recvVCRank);

    if (recvViewingRank == myRankViewing) {
      int sendWidth = El::Length(nLocA, rowSend, numRowSends);

      int rowRecv = 0;

      for (rowRecv = 0; rowRecv < numRowRecvs; ++rowRecv) {
        const int sendVCRank = El::Mod((sendRow + rankBRecv), rowStrideA);
        sendRow = El::Mod(sendRow + rowStrideB, rowStrideA);
        if (rankMap[sendVCRank] == myRankViewing)
          break;
      }

      El::copy::util::InterleaveMatrix(mLocA,
                                       sendWidth,
                                       A.LockedBuffer(0, rowSend),
                                       1,
                                       numRowSends * A.LDim(),
                                       B.Buffer(0, rowRecv),
                                       1,
                                       (numRowRecvs)*B.LDim(),
                                       syncInfoB);
      El::Synchronize(syncInfoA);
      El::Synchronize(syncInfoB);
    }
  }

  std::map<int, int>::iterator sendRankItr, recvRankItr;
  sendRankItr = sendProcessRanks.begin();
  recvRankItr = recvProcessRanks.begin();
  for (int numOp = 0; numOp < numRowRecvs + numRowSends; numOp++) {
    if (recvRankItr != recvProcessRanks.end()) {
      if (recvRankItr->first < myRankViewing ||
          (sendRankItr == sendProcessRanks.end() &&
           recvRankItr->first > myRankViewing)) {
        // Post recv operation

        if (inBGrid) {
          const int sendWidth =
            ((recvRankItr->second * rowStrideB + numInB) >= El::Mod(n, rowLCM))
              ? floor(n / rowLCM)
              : floor(n / rowLCM) + 1;

          El::mpi::Recv(recvBuf,
                        m * sendWidth,
                        recvRankItr->first,
                        viewingCommB,
                        syncInfoB);

          // Unpack the data
          El::copy::util::InterleaveMatrix(m,
                                           sendWidth,
                                           recvBuf,
                                           1,
                                           m,
                                           B.Buffer(0, recvRankItr->second),
                                           1,
                                           (numRowRecvs)*B.LDim(),
                                           syncInfoB);
        }
        recvRankItr++;
      }
      else if (recvRankItr->first != myRankViewing &&
               sendRankItr != sendProcessRanks.end()) {
        // Post send operation if not done already
        // Pack Data
        if (sendRankItr->first != myRankViewing && inAGrid) {

          int sendWidth = El::Length(nLocA, sendRankItr->second, numRowSends);
          El::copy::util::InterleaveMatrix(
            mLocA,
            sendWidth,
            A.LockedBuffer(0, sendRankItr->second),
            1,
            numRowSends * A.LDim(),
            sendBuf,
            1,
            mLocA,
            syncInfoA);

          El::mpi::Send(sendBuf,
                        mLocA * sendWidth,
                        sendRankItr->first,
                        viewingCommB,
                        syncInfoA);
        }
        sendRankItr++;
      }
      else {
        recvRankItr++;
      }
    } // only send operations are left
    else {
      // Post send operation if not done already
      // Pack Data
      if (sendRankItr->first != myRankViewing && inAGrid) {
        int sendWidth = El::Length(nLocA, sendRankItr->second, numRowSends);
        El::copy::util::InterleaveMatrix(mLocA,
                                         sendWidth,
                                         A.LockedBuffer(0, sendRankItr->second),
                                         1,
                                         numRowSends * A.LDim(),
                                         sendBuf,
                                         1,
                                         mLocA,
                                         syncInfoA);
        // Al hangs in GetComm function as it is a universal operation
        //  ::Al::Send<::Al::MPIBackend>(sendBuf, mLocA*sendWidth,
        //  sendRankItr->first,
        //    viewingCommB.template GetComm<::Al::MPIBackend>(syncInfoA));
        El::mpi::Send(sendBuf,
                      mLocA * sendWidth,
                      sendRankItr->first,
                      viewingCommB,
                      syncInfoA);
      }
      sendRankItr++;
    }
  }
}

template <typename T, El::Device Device>
void TranslateBetweenGridsVCAsync(
  const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B,
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& subset,
  std::vector<ReqT>& Requests)
{
  LBANN_ERROR("TranslateBetweenGridsVCAsync function is not implemented for "
              "this configuration");
}

template <>
void TranslateBetweenGridsVCAsync(
  const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& A,
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& B,
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& subset,
  std::vector<ReqT>& Requests)
{
  int height = A.Height();
  int width = A.Width();
  const bool inBGrid = B.Participating();
  const bool inAGrid = A.Participating();
  // Assumes that viewing comm of A and B is same
  El::mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
  El::mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  const int commSizeA = A.Grid().VCSize();
  const int commSizeB = B.Grid().VCSize();
  El::SyncInfo<KFACDevice> syncInfoA = El::SyncInfoFromMatrix(A.LockedMatrix());
  El::SyncInfo<KFACDevice> syncInfoB = El::SyncInfoFromMatrix(B.LockedMatrix());

  El::SyncInfo<KFACDevice> syncGeneral = El::SyncInfo<KFACDevice>();

  El::Int recvMetaData[2], metaData[2];

  if (inAGrid) {
    metaData[0] = height;
    metaData[1] = width;
  }
  else {
    metaData[0] = 0;
    metaData[1] = 0;
  }

  El::mpi::AllReduce(metaData,
                     recvMetaData,
                     2,
                     El::mpi::MAX,
                     viewingCommB,
                     syncGeneral);
  El::Synchronize(syncGeneral);

  height = recvMetaData[0];
  width = recvMetaData[1];

  B.Resize(height, width);
  subset.Resize(height, width);

  if (inAGrid == inBGrid) {
    LBANN_ERROR("TranslateBetweenGridsAsync: A rank cannnot be the part of "
                "both grids or it must be the part of one grid");
  }

  El::mpi::Comm const& activeCommB = B.Grid().ViewingComm();

  if (!El::mpi::Congruent(viewingCommA, viewingCommB))
    LBANN_ERROR("communicators were not congruent");

  const int rankB = B.Grid().VCRank();

  if (commSizeA > commSizeB)
    TranslateBetweenGridsVC(A, subset);

  BackendT::comm_type& backend_commB =
    activeCommB.template GetComm<BackendT>(syncInfoB);
  BackendT::comm_type& backend_commA =
    activeCommB.template GetComm<BackendT>(syncInfoA);

  if (inAGrid) {
    const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>&
      send_mat = commSizeA > commSizeB ? subset : A;
    bool in_send_grid = send_mat.Participating();

    const int transferSize = commSizeA > commSizeB
                               ? subset.LocalHeight() * subset.LocalWidth()
                               : A.LocalHeight() * A.LocalWidth();

    if (in_send_grid) {
      kfac::ReqT sendRequest;
      Requests.push_back(sendRequest);
      int to_send_index = send_mat.Grid().VCRank();
      const int sendViewingRank = B.Grid().VCToViewing(to_send_index);
      ::Al::NonblockingSend<BackendT>((DataType*)send_mat.LockedBuffer(),
                                      transferSize,
                                      sendViewingRank,
                                      backend_commA,
                                      Requests.back());
    }
  }

  if (inBGrid) {
    kfac::ReqT recv_request;
    El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>&
      recv_mat = commSizeA < commSizeB ? subset : B;

    int comm_size_recv_mat = recv_mat.Grid().VCSize();
    int recv_index = rankB % comm_size_recv_mat;
    const int recvViewingRank = A.Grid().VCToViewing(recv_index);
    bool in_recv_grid = recv_mat.Participating();

    const int transferSize = commSizeA < commSizeB
                               ? subset.LocalHeight() * subset.LocalWidth()
                               : B.LocalHeight() * B.LocalWidth();
    if (in_recv_grid) {
      Requests.push_back(recv_request);
      ::Al::NonblockingRecv<BackendT>((DataType*)recv_mat.Buffer(),
                                      transferSize,
                                      recvViewingRank,
                                      backend_commB,
                                      Requests.back());
    }
  }
}

template <typename T, El::Device Device>
void TranslateBetweenGridsVCAsyncDirect(
  const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B,
  El::Int featureSize,
  El::Int currentBatchSize,
  std::vector<ReqT>& Requests)
{
  LBANN_ERROR("TranslateBetweenGridsVCAsyncDirect function is not implemented "
              "for this configuration");
}

template <>
void TranslateBetweenGridsVCAsyncDirect(
  const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& A,
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& B,
  El::Int featureSize,
  El::Int currentBatchSize,
  std::vector<ReqT>& Requests)
{
  int height = A.Height();
  int width = A.Width();
  const bool inBGrid = B.Participating();
  const bool inAGrid = A.Participating();
  // Assumes that viewing comm of A and B is same
  El::mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
  El::mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  const int commSizeA = A.Grid().VCSize();
  const int commSizeB = B.Grid().VCSize();
  El::SyncInfo<KFACDevice> syncInfoA = El::SyncInfoFromMatrix(A.LockedMatrix());
  El::SyncInfo<KFACDevice> syncInfoB = El::SyncInfoFromMatrix(B.LockedMatrix());

  height = featureSize;
  width = currentBatchSize;

  B.Resize(height, width);

  if (inAGrid == inBGrid) {
    LBANN_ERROR("TranslateBetweenGridsVCAsyncDirect: A rank cannnot be the "
                "part of both grids or it must be the part of one grid");
  }

  El::mpi::Comm const& activeCommB = B.Grid().ViewingComm();

  if (!El::mpi::Congruent(viewingCommA, viewingCommB))
    LBANN_ERROR("communicators were not congruent");

  // BVE FIXME unused variable
  // const int rankB = B.Grid().VCRank();

  std::vector<int> columns_per_rankA, columns_per_rankB;
  std::vector<std::vector<int>> to_send_ranks, to_recv_ranks, num_cols_to_send,
    num_cols_to_recv;

  to_send_ranks.resize(commSizeA);
  num_cols_to_send.resize(commSizeA);

  to_recv_ranks.resize(commSizeB);
  num_cols_to_recv.resize(commSizeB);

  // Generating number of columns required in A and B
  int max_cols_A = int(std::ceil(float(width) / commSizeA));
  int min_cols_A = int(std::floor(float(width) / commSizeA));
  int max_cols_B = int(std::ceil(float(width) / commSizeB));
  int min_cols_B = int(std::floor(float(width) / commSizeB));

  // Generating number of columns in A
  for (int i = 0; i < commSizeA; ++i) {
    if (width % commSizeA > i)
      columns_per_rankA.push_back(max_cols_A);
    else
      columns_per_rankA.push_back(min_cols_A);
  }
  // Generating number of columns in B
  for (int i = 0; i < commSizeB; ++i) {
    if (width % commSizeB > i)
      columns_per_rankB.push_back(max_cols_B);
    else
      columns_per_rankB.push_back(min_cols_B);
  }

  if (commSizeA != commSizeB) {

    std::vector<int> columns_left_per_rankA(columns_per_rankA),
      columns_left_per_rankB(columns_per_rankB);

    for (int i = 0; i < std::max(commSizeA, commSizeB); ++i) {
      int index_a = i % commSizeA;
      int index_b = i % commSizeB;
      int min_transfer = std::min(columns_left_per_rankA[index_a],
                                  columns_left_per_rankB[index_b]);

      if (min_transfer > 0) {
        columns_left_per_rankA[index_a] =
          columns_left_per_rankA[index_a] - min_transfer;
        columns_left_per_rankB[index_b] =
          columns_left_per_rankB[index_b] - min_transfer;
        to_send_ranks[index_a].push_back(index_b);
        to_recv_ranks[index_b].push_back(index_a);

        num_cols_to_send[index_a].push_back(min_transfer);
        num_cols_to_recv[index_b].push_back(min_transfer);
      }
    }

    // Checking if any column is left
    if (std::accumulate(columns_left_per_rankA.begin(),
                        columns_left_per_rankA.end(),
                        0) > 0) {
      for (int i = 0; i < commSizeA; ++i) {
        for (int j = 0; j < commSizeB; ++j) {
          if (columns_per_rankA[i] == 0)
            break;
          int min_transfer =
            std::min(columns_left_per_rankA[i], columns_left_per_rankB[j]);
          if (min_transfer > 0) {

            columns_left_per_rankA[i] =
              columns_left_per_rankA[i] - min_transfer;
            columns_left_per_rankB[j] =
              columns_left_per_rankB[j] - min_transfer;
            to_send_ranks[i].push_back(j);
            to_recv_ranks[j].push_back(i);

            num_cols_to_send[i].push_back(min_transfer);
            num_cols_to_recv[j].push_back(min_transfer);
          }
        }
      }
    }
  }
  else {
    for (int i = 0; i < commSizeA; ++i) {
      to_send_ranks[i].push_back(i);
      to_recv_ranks[i].push_back(i);

      num_cols_to_send[i].push_back(columns_per_rankA[i]);
      num_cols_to_recv[i].push_back(columns_per_rankA[i]);
    }
  }

  BackendT::comm_type& backend_commB =
    activeCommB.template GetComm<BackendT>(syncInfoB);
  BackendT::comm_type& backend_commA =
    activeCommB.template GetComm<BackendT>(syncInfoA);

  if (inAGrid) {
    // const El::DistMatrix<DataType,El::STAR,El::VC,El::ELEMENT,KFACDevice>&
    // send_mat = commSizeA > commSizeB ? subset : A; bool in_send_grid =
    // send_mat.Participating();

    int start_col = 0;
    int my_rank = A.Grid().VCRank();

    for (int rank_to_send_index = 0;
         rank_to_send_index < int(to_send_ranks[my_rank].size());
         ++rank_to_send_index) {
      kfac::ReqT sendRequest;
      Requests.push_back(sendRequest);
      const int transferSize =
        A.LocalHeight() * num_cols_to_send[my_rank][rank_to_send_index];
      const int sendViewingRank =
        B.Grid().VCToViewing(to_send_ranks[my_rank][rank_to_send_index]);
      ::Al::NonblockingSend<BackendT>((DataType*)A.LockedBuffer(0, start_col),
                                      transferSize,
                                      sendViewingRank,
                                      backend_commA,
                                      Requests.back());

      start_col += num_cols_to_send[my_rank][rank_to_send_index];
    }
  }

  if (inBGrid) {
    // ReqT recv_request;
    // El::DistMatrix<DataType,El::STAR,El::VC,El::ELEMENT,KFACDevice>& recv_mat
    // = commSizeA < commSizeB ? subset : B;

    int start_col = 0;
    int my_rank = B.Grid().VCRank();

    for (int rank_to_recv_index = 0;
         rank_to_recv_index < int(to_recv_ranks[my_rank].size());
         ++rank_to_recv_index) {

      kfac::ReqT recv_request;
      Requests.push_back(recv_request);
      const int recvViewingRank =
        A.Grid().VCToViewing(to_recv_ranks[my_rank][rank_to_recv_index]);
      const int transferSize =
        B.LocalHeight() * num_cols_to_recv[my_rank][rank_to_recv_index];
      ::Al::NonblockingRecv<BackendT>((DataType*)B.Buffer(0, start_col),
                                      transferSize,
                                      recvViewingRank,
                                      backend_commB,
                                      Requests.back());

      start_col += num_cols_to_recv[my_rank][rank_to_recv_index];
    }
  }
}

template <typename T, El::Device Device>
void TranslateBetweenGridsSTARAsync(
  const El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, Device>& A,
  El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, Device>& B,
  std::vector<ReqT>& Requests)
{
  LBANN_ERROR("TranslateBetweenGridsSTARAsync function is not implemented for "
              "this configuration");
}

template <>
void TranslateBetweenGridsSTARAsync(
  const El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, KFACDevice>&
    A,
  El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, KFACDevice>& B,
  std::vector<ReqT>& Requests)
{
  const int height = A.Height();
  const int width = A.Width();
  B.Resize(height, width);

  // Assumes that viewing comm of A and B is same
  El::mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
  El::mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  const int commSizeA = A.Grid().VCSize();
  const int commSizeB = B.Grid().VCSize();
  const bool inBGrid = B.Participating();
  const bool inAGrid = A.Participating();
  const int transferSize = A.Height() * A.Width();
  El::SyncInfo<KFACDevice> syncInfoA = El::SyncInfoFromMatrix(A.LockedMatrix());
  // BVE FIXME unused variable
  // El::SyncInfo<KFACDevice> syncInfoB =
  // El::SyncInfoFromMatrix(B.LockedMatrix());

  if (inAGrid == inBGrid) {
    LBANN_ERROR("TranslateBetweenGridsAsync: A rank cannnot be the part of "
                "both grids or it must be the part of one grid");
  }

  El::mpi::Comm const& activeCommB = B.Grid().ViewingComm();

  if (!El::mpi::Congruent(viewingCommA, viewingCommB))
    LBANN_ERROR("communicators were not congruent");

  const int rankA = A.Grid().VCRank();
  const int rankB = B.Grid().VCRank();

  // BVE FIXME unused variable
  // BackendT::comm_type& backend_commB = activeCommB.template
  // GetComm<BackendT>(syncInfoB);
  BackendT::comm_type& backend_commA =
    activeCommB.template GetComm<BackendT>(syncInfoA);

  if (inAGrid) {
    int num_sends = (int)std::ceil((float)commSizeB / (float)commSizeA);

    for (int num_send = 0; num_send < num_sends; num_send++) {
      if (rankA + num_send * commSizeA < commSizeB) {
        ReqT sendRequest;
        Requests.push_back(sendRequest);
        int to_send_index = rankA + num_send * commSizeA;
        const int sendViewingRank = B.Grid().VCToViewing(to_send_index);
        ::Al::NonblockingSend<BackendT>((DataType*)A.LockedBuffer(),
                                        transferSize,
                                        sendViewingRank,
                                        backend_commA,
                                        Requests.back());
      }
    }
  }

  if (inBGrid) {
    ReqT recvRequest;
    Requests.push_back(recvRequest);
    int recv_index = rankB % commSizeA;
    const int recvViewingRank = A.Grid().VCToViewing(recv_index);

    ::Al::NonblockingRecv<BackendT>((DataType*)B.Buffer(),
                                    transferSize,
                                    recvViewingRank,
                                    backend_commA,
                                    Requests.back());
  }
}

template <typename T, El::Device Device>
void TranslateBetweenGridsKFACAsync(
  const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B,
  std::vector<ReqT>& Requests)
{
  LBANN_ERROR("TranslateBetweenGridsKFACAsync function is not implemented for "
              "this configuration");
}

template <>
void TranslateBetweenGridsKFACAsync(
  const El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& A,
  El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, KFACDevice>& B,
  std::vector<ReqT>& Requests)
{
  // Transfers matrix A to B without keeping the order of columns
  // It should be used when column order does not matter
  //  like each column corresponds to different samples in DNN training
  const int height = A.Height();
  const int width = A.Width();
  B.Resize(height, width);

  // Assumes that viewing comm of A and B is same
  El::mpi::Comm const& viewingCommA = A.Grid().ViewingComm();
  El::mpi::Comm const& viewingCommB = B.Grid().ViewingComm();
  const int commSizeA = A.Grid().VCSize();
  const int commSizeB = B.Grid().VCSize();
  const bool inBGrid = B.Participating();
  const bool inAGrid = A.Participating();
  int transferSize = A.Height() * A.Width();
  const int rankA = A.Grid().VCRank();
  const int rankB = B.Grid().VCRank();
  El::SyncInfo<KFACDevice> syncInfoA = El::SyncInfoFromMatrix(A.LockedMatrix());
  El::SyncInfo<KFACDevice> syncInfoB = El::SyncInfoFromMatrix(B.LockedMatrix());

  if (inAGrid == inBGrid) {
    LBANN_ERROR("TranslateBetweenGridsAsync: A rank cannnot be the part of "
                "both grids or it must be the part of one grid");
  }

  std::vector<int> numCollumnsPerProcessA, numCollumnsPerProcessB;
  numCollumnsPerProcessA.resize(
    commSizeA,
    (int)std::floor((float)width / (float)commSizeA));
  numCollumnsPerProcessB.resize(
    commSizeB,
    (int)std::floor((float)width / (float)commSizeB));

  for (int i = 0; i < width % commSizeA; ++i)
    numCollumnsPerProcessA[i] = numCollumnsPerProcessA[i] + 1;
  for (int i = 0; i < width % commSizeB; ++i)
    numCollumnsPerProcessB[i] = numCollumnsPerProcessB[i] + 1;

  std::vector<int> ranksFromRecvData, ranksToSendData, dataToRecvRanks,
    dataToSendRanks;

  // Determine ranks to send/recv  data

  if (inAGrid) {
    // myEndingindex is the index of last data to send
    int myStartingIndex = 0, myEndingIndex;
    for (int i = 0; i < rankA; ++i)
      myStartingIndex += numCollumnsPerProcessA[i];
    myEndingIndex = myStartingIndex + numCollumnsPerProcessA[rankA] - 1;

    int startingIndexB = 0, endingIndexB = -1;
    for (int i = 0; i < commSizeB; ++i) {
      bool updated_flag = false;
      if (startingIndexB >= myStartingIndex and
          startingIndexB <= myEndingIndex) {
        updated_flag = true;
        ranksToSendData.push_back(i);
        if (numCollumnsPerProcessB[i] + startingIndexB > myEndingIndex) {
          dataToSendRanks.push_back(myEndingIndex - startingIndexB + 1);
        }
        else {
          dataToSendRanks.push_back(numCollumnsPerProcessB[i]);
        }
      }
      endingIndexB = startingIndexB + numCollumnsPerProcessB[i] - 1;
      if (endingIndexB >= myStartingIndex and endingIndexB <= myEndingIndex and
          updated_flag == false) {
        updated_flag = true;
        dataToSendRanks.push_back(endingIndexB - myStartingIndex + 1);
        ranksToSendData.push_back(i);
      }
      if (startingIndexB < myStartingIndex and endingIndexB > myEndingIndex) {
        updated_flag = true;
        ranksToSendData.push_back(i);
        dataToSendRanks.push_back(numCollumnsPerProcessA[rankA]);
      }
      startingIndexB += numCollumnsPerProcessB[i];
    }
  }
  if (inBGrid) {
    int myStartingIndex = 0, myEndingIndex;
    for (int i = 0; i < rankB; ++i)
      myStartingIndex += numCollumnsPerProcessB[i];
    myEndingIndex = myStartingIndex + numCollumnsPerProcessB[rankB] - 1;

    int startingIndexA = 0, endingIndexA = -1;
    for (int i = 0; i < commSizeA; ++i) {
      bool updated_flag = false;
      if (startingIndexA >= myStartingIndex and
          startingIndexA <= myEndingIndex) {
        updated_flag = true;
        ranksFromRecvData.push_back(i);
        if (numCollumnsPerProcessA[i] + startingIndexA > myEndingIndex) {
          dataToRecvRanks.push_back(myEndingIndex - startingIndexA + 1);
        }
        else {
          dataToRecvRanks.push_back(numCollumnsPerProcessA[i]);
        }
      }
      endingIndexA = startingIndexA + numCollumnsPerProcessA[i] - 1;
      if (endingIndexA >= myStartingIndex and endingIndexA <= myEndingIndex and
          updated_flag == false) {
        updated_flag = true;
        ranksFromRecvData.push_back(i);
        dataToRecvRanks.push_back(endingIndexA - myStartingIndex + 1);
      }
      if (startingIndexA < myStartingIndex and endingIndexA > myEndingIndex) {
        updated_flag = true;
        ranksFromRecvData.push_back(i);
        dataToRecvRanks.push_back(numCollumnsPerProcessB[rankB]);
      }
      startingIndexA += numCollumnsPerProcessA[i];
    }
  }

  El::mpi::Comm const& activeCommB = B.Grid().ViewingComm();

  if (!El::mpi::Congruent(viewingCommA, viewingCommB))
    LBANN_ERROR("communicators were not congruent");

  int initialIndex = 0;
  if (inAGrid) {
    for (int num_send = 0; num_send < (int)ranksToSendData.size(); num_send++) {
      ReqT sendRequest;
      Requests.push_back(sendRequest);
      const int sendViewingRank =
        B.Grid().VCToViewing(ranksToSendData[num_send]);
      transferSize = height * dataToSendRanks[num_send];
      ::Al::NonblockingSend<BackendT>(
        (DataType*)A.LockedBuffer(0, initialIndex),
        transferSize,
        sendViewingRank,
        activeCommB.template GetComm<BackendT>(syncInfoA),
        Requests.back());
      initialIndex += dataToSendRanks[num_send];
    }
  }

  if (inBGrid) {
    for (int num_recv = 0; num_recv < (int)ranksFromRecvData.size();
         num_recv++) {
      ReqT recvRequest;
      Requests.push_back(recvRequest);

      const int recvViewingRank =
        A.Grid().VCToViewing(ranksFromRecvData[num_recv]);
      transferSize = height * dataToRecvRanks[num_recv];

      ::Al::NonblockingRecv<BackendT>(
        (DataType*)B.Buffer(0, initialIndex),
        transferSize,
        recvViewingRank,
        activeCommB.template GetComm<BackendT>(syncInfoB),
        Requests.back());

      initialIndex += dataToRecvRanks[num_recv];
    }
  }
}

template <>
void add_to_diagonal(El::Matrix<DataType, El::Device::CPU>& A,
                     const DataType damping,
                     const DataType damping_bn_err,
                     const bool is_bn,
                     const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const auto height = A.Height();
#pragma omp parallel for
  for (int i = 0; i < height; i++)
    A(i, i) += (is_bn && i >= A.Height() / 2 ? damping_bn_err : damping);
}

template <>
void make_diagonal(El::Matrix<DataType, El::Device::CPU>& A,
                   El::Matrix<DataType, El::Device::CPU>& B,
                   const DataType damping,
                   const DataType damping_bn_err,
                   const bool is_bn,
                   const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const auto height = A.Height();
#pragma omp parallel for
  for (int i = 0; i < height; i++)
    A(i, i) =
      DataType(1) /
      (B(i) + (is_bn && i >= A.Height() / 2 ? damping_bn_err : damping));
}

template <>
void fill_upper_tri(El::Matrix<DataType, El::Device::CPU>& A,
                    const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const auto height = A.Height();
#pragma omp parallel for
  for (int col = 0; col < height; col++)
    for (int row = 0; row < height; row++)
      if (row < col)
        A(row, col) += A(col, row);
}

// TODO: Do not define count but use A.Height()*A.Height()
template <>
void update_kronecker_average(El::Matrix<DataType, El::Device::CPU>& Aave,
                              const El::Matrix<DataType, El::Device::CPU>& A,
                              const size_t count,
                              const double decay,
                              const El::SyncInfo<El::Device::CPU>& sync_info)
{
  assert(count == (size_t)(A.Height() * A.Height()));
  const auto height = A.Height();
#pragma omp parallel for
  for (int col = 0; col < height; col++)
    for (int row = 0; row < height; row++)
      Aave(row, col) = Aave(row, col) * decay + A(row, col) * (1.0 - decay);
}

template <>
void identity(El::Matrix<DataType, El::Device::CPU>& A,
              const El::SyncInfo<El::Device::CPU>& sync_info)
{
  El::Identity(A, A.Height(), A.Height());
}

template <>
void pack_lower_tri(El::Matrix<DataType, El::Device::CPU>& L,
                    const El::Matrix<DataType, El::Device::CPU>& A,
                    const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const auto height = A.Height();
#pragma omp parallel for
  for (int col = 0; col < height; col++)
    for (int row = 0; row < height; row++)
      if (row >= col)
        L(row + (2 * height - (col - 1)) * col / 2 - col, 0) =
          A(row + col * height, 0);
}

template <>
void unpack_lower_tri(El::Matrix<DataType, El::Device::CPU>& A,
                      const El::Matrix<DataType, El::Device::CPU>& L,
                      const El::SyncInfo<El::Device::CPU>& sync_info)
{
  const auto height = A.Height();
#pragma omp parallel for
  for (int col = 0; col < height; col++)
    for (int row = 0; row < height; row++)
      if (row >= col)
        A(row + col * height, 0) = A(col + row * height, 0) =
          L(row + (2 * height - (col - 1)) * col / 2 - col, 0);
}

#define PROTO_DEVICE(T, Device)                                                \
  template void get_matrix_inverse(El::AbstractMatrix<T>& Ainv,                \
                                   El::AbstractMatrix<T>& Linv,                \
                                   const El::AbstractMatrix<T>& A,             \
                                   bool report_time,                           \
                                   T damping,                                  \
                                   T damping_bn_err,                           \
                                   bool is_bn,                                 \
                                   const El::SyncInfo<Device>& sync_info);     \
  template void get_matrix_inverse_eigen(                                      \
    El::AbstractMatrix<T>& Ainv,                                               \
    El::AbstractMatrix<T>& Linv,                                               \
    const El::AbstractMatrix<T>& A,                                            \
    bool report_time,                                                          \
    T damping,                                                                 \
    T damping_bn_err,                                                          \
    bool is_bn,                                                                \
    const El::SyncInfo<Device>& sync_info);                                    \
  template std::string get_matrix_stat(const El::Matrix<T, Device>& X,         \
                                       const char* name);                      \
  template void allreduce_lower_tri(El::AbstractMatrix<T>& A,                  \
                                    El::AbstractMatrix<T>& AL,                 \
                                    lbann_comm* comm,                          \
                                    const El::SyncInfo<Device>& sync_info);    \
  template void reduce_scatter_blocks(                                         \
    const std::vector<std::pair<size_t, El::AbstractMatrix<T>*>>& blocks,      \
    El::Matrix<T, Device>& global_buffer,                                      \
    lbann_comm* comm,                                                          \
    const kfac_reduce_scatter_mode mode);                                      \
  template void allgather_blocks(                                              \
    const std::vector<std::pair<size_t, El::AbstractMatrix<T>*>>& blocks,      \
    El::Matrix<T, Device>& local_buffer,                                       \
    El::Matrix<T, Device>& global_buffer,                                      \
    lbann_comm* comm,                                                          \
    const kfac_allgather_mode mode);                                           \
  template void allgather_inverse_matrices_sizes(                              \
    const std::vector<std::shared_ptr<kfac_block<Device>>>& blocks,            \
    El::Matrix<double>& global_buffer,                                         \
    lbann_comm* comm);                                                         \
  template void allgather_inverse_matrices(                                    \
    const std::vector<std::shared_ptr<kfac_block<Device>>>& blocks,            \
    El::Matrix<T, Device>& global_buffer,                                      \
    lbann_comm* comm);
#define PROTO_DEVICECOMM(T, Device)                                            \
  template void TranslateBetweenGridsVCAsync(                                  \
    const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,         \
    El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B,               \
    El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& subset,          \
    std::vector<ReqT>& Requests);                                              \
  template void TranslateBetweenGridsVCAsyncDirect(                            \
    const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,         \
    El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B,               \
    El::Int featureSize,                                                       \
    El::Int currentBatchSize,                                                  \
    std::vector<ReqT>& Requests);                                              \
  template void TranslateBetweenGridsKFACAsync(                                \
    const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,         \
    El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B,               \
    std::vector<ReqT>& Requests);                                              \
  template void TranslateBetweenGridsSTARAsync(                                \
    const El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, Device>& A,       \
    El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, Device>& B,             \
    std::vector<ReqT>& Requests);                                              \
  template void TranslateBetweenGridsVC(                                       \
    const El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& A,         \
    El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, Device>& B);

PROTO_DEVICE(DataType, El::Device::CPU);

#ifdef LBANN_HAS_GPU
PROTO_DEVICE(DataType, El::Device::GPU);
// If GPUS are defined then the default case CPU case needs to be instantiated
PROTO_DEVICECOMM(DataType, El::Device::CPU);
#endif // LBANN_HAS_GPU

} // namespace kfac
} // namespace lbann
