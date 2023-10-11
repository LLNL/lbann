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
// lbann_comm .hpp .cpp - LBANN communication utilities
////////////////////////////////////////////////////////////////////////////////

#define LBANN_COMM_INSTANTIATE
#include "lbann/comm_impl.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/timer.hpp"
#include "mpi.h"
#include "omp.h"
#include <sstream>
#include <thread>

namespace lbann {

// Error utility macro
#ifdef LBANN_DEBUG
#define checkMPI(mpi_call)                                                     \
  {                                                                            \
    const int status = mpi_call;                                               \
    if (status != MPI_SUCCESS) {                                               \
      char error_string[MPI_MAX_ERROR_STRING];                                 \
      int error_string_len;                                                    \
      MPI_Error_string(status, error_string, &error_string_len);               \
      std::cerr << "MPI error: "                                               \
                << std::string(error_string, error_string_len) << "\n"         \
                << "Error at " << __FILE__ << ":" << __LINE__ << "\n";         \
      throw lbann_exception("MPI error");                                      \
    }                                                                          \
  }
#else
#define checkMPI(status) status
#endif // #ifdef LBANN_DEBUG

lbann_comm::lbann_comm(int ppm, El::mpi::Comm world)
  : m_world_comm(std::move(world)),
    m_procs_per_trainer(ppm),
    m_num_trainer_barriers(0),
    m_num_intertrainer_barriers(0),
    m_num_global_barriers(0),
    m_bytes_sent(0),
    m_bytes_received(0)
{
#ifdef LBANN_HAS_ALUMINUM
  // Don't have argc/argv here, but MPI should already be init'd.
  int argc_dummy = 0;
  char** argv_dummy = nullptr;
  ::Al::Initialize(argc_dummy, argv_dummy);
#endif
  // Set up the initial trainer split
  split_trainers(m_procs_per_trainer);

  // Initialize node communicators
  setup_node_comm();
  m_procs_per_node = El::mpi::Size(m_node_comm);
  m_rank_in_node = El::mpi::Rank(m_node_comm);

  // Setup threads
  setup_threads();
}

lbann_comm::~lbann_comm()
{
  m_grid.reset();
  El::mpi::Free(m_trainer_comm);
  El::mpi::Free(m_intertrainer_comm);
  El::mpi::Free(m_node_comm);
#ifdef LBANN_HAS_ALUMINUM
  ::Al::Finalize();
#endif
}

void lbann_comm::split_trainers(int procs_per_trainer, int trainer_grid_height)
{
  const int world_size = El::mpi::Size(get_world_comm());
  m_procs_per_trainer = procs_per_trainer;
  if (m_procs_per_trainer <= 0) {
    m_procs_per_trainer = world_size;
  }
  // Check if parameters are valid
  if (m_procs_per_trainer > world_size) {
    LBANN_ERROR(
      "Not enough processes to create one trainer; procs_per_trainer: ",
      m_procs_per_trainer,
      " is larger than world_size: ",
      world_size);
  }
  if (world_size % m_procs_per_trainer != 0) {
    LBANN_ERROR("Procs per trainer does not divide total number of procs; "
                "procs_per_trainer: ",
                m_procs_per_trainer,
                " total number of procs (world size): ",
                world_size);
  }

  m_num_trainers = world_size / m_procs_per_trainer;
  m_trainer_rank = El::mpi::Rank(get_world_comm()) / m_procs_per_trainer;
  m_rank_in_trainer = El::mpi::Rank(get_world_comm()) % m_procs_per_trainer;

  // Initialize trainer and intertrainer communicators
  El::mpi::Split(get_world_comm(),
                 m_trainer_rank,
                 m_rank_in_trainer,
                 m_trainer_comm);
  El::mpi::Split(get_world_comm(),
                 m_rank_in_trainer,
                 m_trainer_rank,
                 m_intertrainer_comm);

  // Initialize Elemental grid for trainer
  m_grid = std::make_unique<El::Grid>(m_trainer_comm.GetMPIComm(),
                                      trainer_grid_height);
}

void lbann_comm::split_trainer_grid(int num_process_primary_grid,
                                    bool create_two_models,
                                    bool enable_async_comm,
                                    bool enable_topo_aware)
{
  const int trainer_size = El::mpi::Size(m_trainer_comm);
  m_create_two_models = create_two_models;
  m_subgrid_async_progress = enable_async_comm;
  bool enable_topology_aware = enable_topo_aware;
  // enable_topology_aware = true;
  std::cout << "Topoaware In comm:" << enable_topo_aware << "\n";

  // If primary grid size is not given then split resources equally between
  // primary and secondary grid
  if (num_process_primary_grid == 0) {
    num_process_primary_grid = trainer_size / 2;
  }

  if (num_process_primary_grid == 0) {
    LBANN_ERROR("Procs for primary grid in a trainer cannot be zero.");
  }

  if (num_process_primary_grid == trainer_size) {
    return;
  }

  std::cout << "Rank:" << m_rank_in_trainer << " split trainer grid\n"
            << std::flush;
  int num_process_secondary_grid = trainer_size - num_process_primary_grid;

  int rank_in_split_comm;
  if (enable_topology_aware == false) {
    if (m_rank_in_trainer < num_process_primary_grid) {
      rank_in_split_comm = m_rank_in_trainer % num_process_primary_grid;
      m_grid_type = GridType::PRIMARY_GRID;
      m_rank_in_trainer = rank_in_split_comm;
      m_procs_per_trainer = num_process_primary_grid;
    }
    else {
      rank_in_split_comm = (m_rank_in_trainer - num_process_primary_grid) %
                           num_process_secondary_grid;
      m_grid_type = GridType::SECONDARY_GRID;
      m_rank_in_trainer = rank_in_split_comm;
      m_procs_per_trainer = num_process_secondary_grid;
    }
    // Update ranks in primary and secondary grids
    for (int rank = 0; rank < num_process_primary_grid; ++rank) {
      m_primary_grid_ranks.push_back(rank);
    }
    for (int rank = num_process_primary_grid;
         rank < num_process_primary_grid + num_process_secondary_grid;
         ++rank) {
      m_secondary_grid_ranks.push_back(rank);
    }
  }
  else { // topology aware

    // Update ranks in primary and secondary grids
    for (int rank = 0;
         rank < num_process_primary_grid + num_process_secondary_grid;
         ++rank) {
      if ((rank % 2 == 0 and rank < 2 * num_process_primary_grid) or
          m_secondary_grid_ranks.size() == (size_t)num_process_secondary_grid)
        m_primary_grid_ranks.push_back(rank);
      else
        m_secondary_grid_ranks.push_back(rank);
    }
    if (std::find(m_primary_grid_ranks.begin(),
                  m_primary_grid_ranks.end(),
                  m_rank_in_trainer) !=
        m_primary_grid_ranks.end()) { // Primary grid

      auto pos = std::find(m_primary_grid_ranks.begin(),
                           m_primary_grid_ranks.end(),
                           m_rank_in_trainer);
      rank_in_split_comm = pos - m_primary_grid_ranks.begin();
      m_grid_type = GridType::PRIMARY_GRID;
      m_rank_in_trainer = rank_in_split_comm;
      m_procs_per_trainer = num_process_primary_grid;
    }
    else {

      auto pos = std::find(m_secondary_grid_ranks.begin(),
                           m_secondary_grid_ranks.end(),
                           m_rank_in_trainer);
      rank_in_split_comm = pos - m_secondary_grid_ranks.begin();
      m_grid_type = GridType::SECONDARY_GRID;
      m_rank_in_trainer = rank_in_split_comm;
      m_procs_per_trainer = num_process_secondary_grid;
    }
  }

  std::cout << "Primary Grid:";
  for (auto it = m_primary_grid_ranks.begin(); it != m_primary_grid_ranks.end();
       it++)
    std::cout << *it << " ";
  std::cout << "\n";

  std::cout << "Secondary Grid:";
  for (auto it = m_secondary_grid_ranks.begin();
       it != m_secondary_grid_ranks.end();
       it++)
    std::cout << *it << " ";
  std::cout << "\n";

  // Create Groups to form communicators
  El::mpi::Group trainer_group, primary_grid_group, secondary_grid_group,
    subset_grid_group;
  El::mpi::CommGroup(m_trainer_comm, trainer_group);
  El::mpi::Incl(trainer_group,
                m_primary_grid_ranks.size(),
                m_primary_grid_ranks.data(),
                primary_grid_group);
  El::mpi::Incl(trainer_group,
                m_secondary_grid_ranks.size(),
                m_secondary_grid_ranks.data(),
                secondary_grid_group);

  // Create communicators (one each for primary and secondary grid)
  El::mpi::Create(m_trainer_comm, primary_grid_group, m_primary_grid_comm);
  El::mpi::Create(m_trainer_comm, secondary_grid_group, m_secondary_grid_comm);

  El::mpi::Dup(m_trainer_comm, m_combined_grid_comm);
  if (m_create_two_models) {
    if (m_grid_type == GridType::PRIMARY_GRID) {
      El::mpi::Dup(m_primary_grid_comm, m_trainer_comm);
    }
    else {
      El::mpi::Dup(m_secondary_grid_comm, m_trainer_comm);
    }
    // Initialize Elemental grid for trainer
    m_grid = std::make_unique<El::Grid>(m_trainer_comm.GetMPIComm(), 1);
  }
  else {
    if (m_grid_type == GridType::PRIMARY_GRID) {
      El::mpi::Dup(m_primary_grid_comm, m_trainer_comm);
    }
    else {
      El::mpi::Dup(m_secondary_grid_comm, m_trainer_comm);
    }
    // Initialize Elemental grid for trainer
    m_grid = std::make_unique<El::Grid>(m_combined_grid_comm.GetMPIComm(),
                                        primary_grid_group,
                                        num_process_primary_grid,
                                        El::COLUMN_MAJOR);

    m_secondary_grid =
      std::make_unique<El::Grid>(m_combined_grid_comm.GetMPIComm(),
                                 secondary_grid_group,
                                 num_process_secondary_grid,
                                 El::COLUMN_MAJOR);
  }

  if (m_subgrid_async_progress) {
    std::vector<int> subset_ranks;

    int subset_grid_size =
      std::min(num_process_primary_grid, num_process_secondary_grid);
    if (num_process_primary_grid > num_process_secondary_grid) {
      for (int i = 0; i < subset_grid_size; ++i) {
        subset_ranks.push_back(m_primary_grid_ranks[i]);
      }
    }

    else {
      for (int i = 0; i < subset_grid_size; ++i) {
        subset_ranks.push_back(m_secondary_grid_ranks[i]);
      }
    }

    El::mpi::Incl(trainer_group,
                  subset_ranks.size(),
                  subset_ranks.data(),
                  subset_grid_group);

    m_subset_grid = make_unique<El::Grid>(m_combined_grid_comm.GetMPIComm(),
                                          subset_grid_group,
                                          subset_grid_size,
                                          El::COLUMN_MAJOR);
  }
}

void lbann_comm::intertrainer_sum_matrix(AbsMat& mat) const
{
  m_bytes_sent += sizeof(DataType) * mat.Height() * mat.Width();
  El::AllReduce(mat, m_intertrainer_comm, El::mpi::SUM);
  m_bytes_received += sizeof(DataType) * mat.Height() * mat.Width();
}

void lbann_comm::intertrainer_sum_matrix(AbsDistMat& mat) const
{
  allreduce(mat, m_intertrainer_comm, El::mpi::SUM);
}

namespace {

template <typename BackendT>
struct BackendTag
{
};

#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)
[[maybe_unused]] auto GetRequest(Al::request& r, BackendTag<Al::dummy_backend>)
  -> typename Al::dummy_backend::req_type
{
  return Al::dummy_backend::null_req;
}

[[maybe_unused]] auto GetRequest(Al::request& r, BackendTag<::Al::MPIBackend>)
  -> typename ::Al::MPIBackend::req_type&
{
  return r.mpi_req;
}
[[maybe_unused]] void
UpdateRequest(typename ::Al::MPIBackend::req_type&,
              El::SyncInfo<El::Device::CPU> const&) noexcept
{}

#ifdef AL_HAS_NCCL
[[maybe_unused]] auto GetRequest(Al::request& r,
                                 BackendTag<::Al::NCCLBackend>) noexcept ->
  typename ::Al::NCCLBackend::req_type&
{
  return r.nccl_req;
}
[[maybe_unused]] void
UpdateRequest(typename ::Al::NCCLBackend::req_type& req,
              El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_NCCL

#ifdef AL_HAS_HOST_TRANSFER
[[maybe_unused]] auto GetRequest(Al::request& r,
                                 BackendTag<::Al::HostTransferBackend>) noexcept
  -> typename ::Al::HostTransferBackend::req_type&
{
  return r.hosttransfer_req;
}
[[maybe_unused]] void
UpdateRequest(typename ::Al::HostTransferBackend::req_type& req,
              El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_HOST_TRANSFER
#endif // defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)

// The best we can do on CPU is exactly the Elemental implementation:
// If the buffer is contiguous, call the El::mpi interface, which will
// dispatch to Aluminum if possible for the type; otherwise,
// pack-allreduce-unpack.
//
// Likewise, if we don't have Aluminum, this is the best we can do on GPU.
//
// If we DO have Aluminum, the compiler should select that overload
// for GPUs as it is "more specialized" than this template. If that's
// not what's happening, there's a compiler bug.
template <typename T, El::Device D>
void allreduce_impl(El::Matrix<T, D>& m,
                    const El::mpi::Comm& c,
                    El::mpi::Op const& op)
{
  return El::AllReduce(m, c, op);
}

template <typename T>
void nb_allreduce_impl(El::Matrix<T, El::Device::CPU>& m,
                       const El::mpi::Comm& c,
                       Al::request& req,
                       El::mpi::Op const& op)
{
  if (m.Height() == m.LDim() || m.Width() == 1) {
    auto const count = m.Height() * m.Width();
    MPI_Iallreduce(MPI_IN_PLACE,
                   m.Buffer(),
                   count,
                   El::mpi::TypeMap<T>(),
                   op.op,
                   c.GetMPIComm(),
                   &(req.raw_mpi_req));
  }
  else {
    return El::AllReduce(m, c, op);
  }
}

#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)

template <typename T,
          typename BackendT,
          El::EnableWhen<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                        const El::mpi::Comm& c,
                        El::mpi::Op const& op,
                        BackendTag<BackendT>,
                        typename BackendT::allreduce_algo_type algo =
                          BackendT::allreduce_algo_type::automatic)
{
  const auto local_size = m.Height() * m.Width();
  ::Al::Allreduce<BackendT>(
    m.Buffer(),
    local_size,
    mpi_op_to_al_op(op),
    c.template GetComm<BackendT>(El::SyncInfoFromMatrix(m)),
    algo);
}

template <typename T,
          typename BackendT,
          El::EnableWhen<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void nb_allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                           const El::mpi::Comm& c,
                           Al::request& req,
                           El::mpi::Op const& op,
                           BackendTag<BackendT> const& tag,
                           typename BackendT::allreduce_algo_type algo =
                             BackendT::allreduce_algo_type::automatic)
{
  const auto local_size = m.Height() * m.Width();
  const auto& syncinfo = El::SyncInfoFromMatrix(m);
  auto& request = GetRequest(req, tag);
  ::Al::NonblockingAllreduce<BackendT>(m.Buffer(),
                                       local_size,
                                       mpi_op_to_al_op(op),
                                       c.template GetComm<BackendT>(syncinfo),
                                       request,
                                       algo);
  UpdateRequest(request, syncinfo);
}

template <typename T,
          typename BackendT,
          El::EnableUnless<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void nb_allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                           const El::mpi::Comm& c,
                           Al::request& req,
                           El::mpi::Op const& op,
                           BackendTag<BackendT> const& tag,
                           typename BackendT::allreduce_algo_type algo =
                             BackendT::allreduce_algo_type::automatic)
{
  El::AllReduce(m, c, op);
}

template <typename T,
          typename BackendT,
          El::EnableUnless<
            El::AluminumSupportsBackendAndCollective<T,
                                                     El::Collective::ALLREDUCE,
                                                     BackendT>,
            int> = 0>
void allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                        const El::mpi::Comm& c,
                        El::mpi::Op const& op,
                        BackendTag<BackendT>,
                        typename BackendT::allreduce_algo_type =
                          BackendT::allreduce_algo_type::automatic)
{
  // We cannot dispatch with this backend directly to Aluminum. Let
  // Elemental handle it.
  El::AllReduce(m, c, op);
}

template <typename T>
void allreduce_impl(El::Matrix<T, El::Device::GPU>& m,
                    El::mpi::Comm const& c,
                    El::mpi::Op const& op)
{
  return El::AllReduce(m, c, op);
}

template <typename T>
void nb_allreduce_impl(El::Matrix<T, El::Device::GPU>& m,
                       El::mpi::Comm const& c,
                       Al::request& req,
                       El::mpi::Op const& op)
{
  if (m.Width() > 1 && m.Height() != m.LDim()) {
    // Aluminum doesn't do allreduces on strided matrices
    return El::AllReduce(m, c, op);
  }

#if defined(AL_HAS_NCCL)
  return nb_allreduce_aluminum(m, c, req, op, BackendTag<::Al::NCCLBackend>{});
#elif defined(AL_HAS_HOST_TRANSFER)
  return nb_allreduce_aluminum(
    m,
    c,
    req,
    op,
    BackendTag<::Al::HostTransferBackend>{},
    ::Al::HostTransferBackend::allreduce_algo_type::host_transfer);
#else
  // At this point just call Elemental again
  return El::AllReduce(m, c, op);
#endif
}

#endif // defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)
} // namespace

template <typename TensorDataType>
void lbann_comm::allreduce(El::AbstractMatrix<TensorDataType>& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) const
{
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }

  const int local_size = m.Height() * m.Width();
  m_bytes_sent += sizeof(DataType) * local_size;
  m_bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (m.GetDevice()) {
  case El::Device::CPU:
    return allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(m),
      c,
      op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m),
      c,
      op);
#endif // LBANN_HAS_GPU
  }
}

template <typename TensorDataType>
void lbann_comm::allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) const
{
  allreduce(m.Matrix(), c, op);
}

template <typename TensorDataType>
void lbann_comm::nb_allreduce(El::AbstractMatrix<TensorDataType>& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) const
{
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }

  const int local_size = m.Height() * m.Width();
  m_bytes_sent += sizeof(DataType) * local_size;
  m_bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (m.GetDevice()) {
  case El::Device::CPU:
    return nb_allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(m),
      c,
      req,
      op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return nb_allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m),
      c,
      req,
      op);
#endif // LBANN_HAS_GPU
  }
}

template <typename TensorDataType>
void lbann_comm::nb_allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) const
{
  nb_allreduce(m.Matrix(), c, req, op);
}

template <typename T,
          typename BackendT,
          El::EnableWhen<El::AluminumSupportsBackendAndCollective<
                           T,
                           El::Collective::REDUCESCATTER,
                           BackendT>,
                         int> = 0>
void nb_reduce_scatter_aluminum(const El::Matrix<T, El::Device::GPU>& src,
                                El::Matrix<T, El::Device::GPU>& dst,
                                const El::mpi::Comm& c,
                                Al::request& req,
                                El::mpi::Op const& op,
                                BackendTag<BackendT> const& tag)
{
  const auto local_size = src.Height() * src.Width();
  const auto& syncinfo = El::SyncInfoFromMatrix(dst);
  auto& request = GetRequest(req, tag);
  ::Al::NonblockingReduce_scatter<BackendT>(
    src.LockedBuffer(),
    dst.Buffer(),
    local_size,
    mpi_op_to_al_op(op),
    c.template GetComm<BackendT>(syncinfo),
    request);
  UpdateRequest(request, syncinfo);
}

template <typename T,
          typename BackendT,
          El::EnableUnless<El::AluminumSupportsBackendAndCollective<
                             T,
                             El::Collective::REDUCESCATTER,
                             BackendT>,
                           int> = 0>
void nb_reduce_scatter_aluminum(const El::Matrix<T, El::Device::GPU>& src,
                                El::Matrix<T, El::Device::GPU>& dst,
                                const El::mpi::Comm& c,
                                Al::request& req,
                                El::mpi::Op const& op,
                                BackendTag<BackendT> const& tag)
{
  const auto& syncinfo = El::SyncInfoFromMatrix(dst);
  return El::mpi::ReduceScatter(src.LockedBuffer(),
                                dst.Buffer(),
                                dst.LocalWidth(),
                                op,
                                c,
                                syncinfo);
}

template <typename T>
void nb_reduce_scatter_impl(const El::Matrix<T, El::Device::GPU>& src,
                            El::Matrix<T, El::Device::GPU>& dst,
                            El::mpi::Comm const& c,
                            Al::request& req,
                            El::mpi::Op const& op)
{
  if (dst.Width() > 1 && dst.Height() != dst.LDim()) {
    // Aluminum doesn't do reducescatter on strided matrices
    const auto& syncinfo = El::SyncInfoFromMatrix(dst);
    return El::mpi::ReduceScatter(src.LockedBuffer(),
                                  dst.Buffer(),
                                  dst.Width(),
                                  op,
                                  c,
                                  syncinfo);
  }

#if defined(AL_HAS_NCCL)
  return nb_reduce_scatter_aluminum(src,
                                    dst,
                                    c,
                                    req,
                                    op,
                                    BackendTag<::Al::NCCLBackend>{});
#elif defined(AL_HAS_HOST_TRANSFER)
  return nb_reduce_scatter_aluminum(src,
                                    dst,
                                    c,
                                    req,
                                    op,
                                    BackendTag<::Al::HostTransferBackend>{});
#else
  // At this point just call Elemental again
  return El::ReduceScatter(src, dst, c, op);
#endif
}

template <typename T>
void nb_reduce_scatter_impl(const El::Matrix<T, El::Device::CPU>& src,
                            El::Matrix<T, El::Device::CPU>& dst,
                            const El::mpi::Comm& c,
                            Al::request& req,
                            El::mpi::Op const& op)
{
  if (dst.Height() == dst.LDim() || dst.Width() == 1) {
    auto const count = src.Height() * src.Width();
    int counts[1] = {static_cast<int>(count)};
    MPI_Ireduce_scatter(src.LockedBuffer(),
                        dst.Buffer(),
                        counts,
                        El::mpi::TypeMap<T>(),
                        op.op,
                        c.GetMPIComm(),
                        &(req.raw_mpi_req));
  }
  else {
    const auto& syncinfo = El::SyncInfoFromMatrix(dst);
    return El::mpi::ReduceScatter(src.LockedBuffer(),
                                  dst.Buffer(),
                                  dst.Width(),
                                  op,
                                  c,
                                  syncinfo);
  }
}

template <typename TensorDataType>
void lbann_comm::nb_reduce_scatter(
  const El::AbstractMatrix<TensorDataType>& src,
  El::AbstractMatrix<TensorDataType>& dst,
  const El::mpi::Comm& c,
  Al::request& req,
  El::mpi::Op op) const
{
  if (El::mpi::Size(c) == 1 || dst.Height() < 1 || dst.Width() < 1) {
    return;
  }

  const int local_size = src.Height() * src.Width();
  m_bytes_sent += sizeof(DataType) * local_size;
  m_bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (dst.GetDevice()) {
  case El::Device::CPU:
    return nb_reduce_scatter_impl(
      static_cast<const El::Matrix<TensorDataType, El::Device::CPU>&>(src),
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(dst),
      c,
      req,
      op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return nb_reduce_scatter_impl(
      static_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(src),
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(dst),
      c,
      req,
      op);
#endif // LBANN_HAS_GPU
  }
}

template <typename TensorDataType>
void lbann_comm::nb_reduce_scatter(
  const El::AbstractDistMatrix<TensorDataType>& src,
  El::AbstractDistMatrix<TensorDataType>& dst,
  const El::mpi::Comm& c,
  Al::request& req,
  El::mpi::Op op) const
{
  nb_reduce_scatter(src.LockedMatrix(), dst.Matrix(), c, req, op);
}

void lbann_comm::wait(Al::request& req) const
{
#ifdef LBANN_HAS_ALUMINUM
  if (req.mpi_req != Al::mpi_null_req) {
    ::Al::Wait<::Al::MPIBackend>(req.mpi_req);
  }
#ifdef AL_HAS_NCCL
  if (req.nccl_req != Al::nccl_null_req) {
    // Note this does not block the host.
    ::Al::Wait<::Al::NCCLBackend>(req.nccl_req);
  }
#endif // AL_HAS_NCCL
#ifdef AL_HAS_HOST_TRANSFER
  if (req.hosttransfer_req != Al::hosttransfer_null_req) {
    // Note this does not block the host.
    ::Al::Wait<::Al::HostTransferBackend>(req.hosttransfer_req);
  }
#endif // AL_HAS_HOST_TRANSFER
#endif // LBANN_HAS_ALUMINUM
  if (req.raw_mpi_req != MPI_REQUEST_NULL) {
    MPI_Wait(&(req.raw_mpi_req), MPI_STATUS_IGNORE);
    ;
  }
}

bool lbann_comm::test(Al::request& req) const
{
  bool req_test = true;
#ifdef LBANN_HAS_ALUMINUM
  if (req.mpi_req != Al::mpi_null_req) {
    req_test = req_test && ::Al::Test<::Al::MPIBackend>(req.mpi_req);
  }
#ifdef AL_HAS_NCCL
  if (req.nccl_req != Al::nccl_null_req) {
    req_test = req_test && ::Al::Test<::Al::NCCLBackend>(req.nccl_req);
  }
#endif // AL_HAS_NCCL
#ifdef AL_HAS_HOST_TRANSFER
  if (req.hosttransfer_req != Al::hosttransfer_null_req) {
    req_test =
      req_test && ::Al::Test<::Al::HostTransferBackend>(req.hosttransfer_req);
  }
#endif // AL_HAS_HOST_TRANSFER
#endif // LBANN_HAS_ALUMINUM
  if (req.raw_mpi_req != MPI_REQUEST_NULL) {
    int flag = 0;
    MPI_Test(&(req.raw_mpi_req), &flag, MPI_STATUS_IGNORE);
    req_test = flag;
  }
  return req_test;
}

void lbann_comm::intertrainer_broadcast_matrix(AbsMat& mat, int root) const
{
  El::Broadcast(mat, m_intertrainer_comm, root);
}

void lbann_comm::intertrainer_broadcast_matrix(AbsDistMat& mat, int root) const
{
  El::Broadcast(mat, m_intertrainer_comm, root);
}

template <>
void lbann_comm::broadcast<std::string>(const int root,
                                        std::string& str,
                                        const El::mpi::Comm& c) const
{
  std::vector<char> data(str.begin(), str.end());
  broadcast(root, data, c);
  str.assign(data.begin(), data.end());
}

void lbann_comm::intertrainer_barrier() const
{
  ++m_num_intertrainer_barriers;
  barrier(m_intertrainer_comm);
}

void lbann_comm::trainer_barrier() const
{
  ++m_num_trainer_barriers;
  barrier(m_trainer_comm);
}

void lbann_comm::global_barrier() const
{
  ++m_num_global_barriers;
  barrier(get_world_comm());
}

void lbann_comm::barrier(const El::mpi::Comm& c) const { El::mpi::Barrier(c); }

void lbann_comm::send(const AbsMat& mat,
                      const int trainer,
                      const int rank) const
{
  El::Send(mat, get_world_comm(), get_world_rank(trainer, rank));
}

void lbann_comm::send(const DistMat& mat,
                      const int trainer,
                      const int rank) const
{
  send(mat.LockedMatrix(), trainer, rank);
}

void lbann_comm::nb_send(const AbsMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_send(mat.LockedBuffer(), mat.Height() * mat.Width(), trainer, rank, req);
}

void lbann_comm::nb_send(const DistMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_send(mat.LockedBuffer(),
          mat.LocalHeight() * mat.LocalWidth(),
          trainer,
          rank,
          req);
}

void lbann_comm::recv(AbsMat& mat, const int trainer, const int rank) const
{
  El::Recv(mat, get_world_comm(), get_world_rank(trainer, rank));
}

void lbann_comm::recv(DistMat& mat, const int trainer, const int rank) const
{
  recv(mat.Matrix(), trainer, rank);
}

void lbann_comm::recv(AbsMat& mat) const
{
  El::Recv(mat, get_world_comm(), El::mpi::ANY_SOURCE);
}

void lbann_comm::recv(DistMat& mat) const { recv(mat.Matrix()); }

void lbann_comm::nb_recv(AbsMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), trainer, rank, req);
}

void lbann_comm::nb_recv(DistMat& mat,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(),
          mat.LocalHeight() * mat.LocalWidth(),
          trainer,
          rank,
          req);
}

void lbann_comm::nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), req);
}

void lbann_comm::nb_recv(DistMat& mat, El::mpi::Request<DataType>& req) const
{
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), req);
}

void lbann_comm::setup_node_comm()
{

  // Get string specifying compute node
  char node_name[MPI_MAX_PROCESSOR_NAME];
  int node_name_len;
  checkMPI(MPI_Get_processor_name(node_name, &node_name_len));
  const std::string node_string(node_name);

  // Hash node names and split MPI processes
  int hash = std::hash<std::string>()(node_string);
  hash = hash >= 0 ? hash : -hash; // Make sure hash is non-negative
  El::mpi::Comm hash_comm;
  El::mpi::Split(get_world_comm(),
                 hash,
                 El::mpi::Rank(get_world_comm()),
                 hash_comm);
  const int hash_comm_size = El::mpi::Size(hash_comm);

  // Compare node names and split MPI processes
  int node_num = El::mpi::Rank(hash_comm);
  {
    std::vector<char> node_name_list(hash_comm_size * MPI_MAX_PROCESSOR_NAME);
    checkMPI(MPI_Allgather(node_name,
                           MPI_MAX_PROCESSOR_NAME,
                           MPI_CHAR,
                           node_name_list.data(),
                           MPI_MAX_PROCESSOR_NAME,
                           MPI_CHAR,
                           hash_comm.GetMPIComm()));
    for (int i = 0; i < hash_comm_size; ++i) {
      const std::string other_node_string(node_name_list.data() +
                                          i * MPI_MAX_PROCESSOR_NAME);
      if (node_string == other_node_string) {
        node_num = i;
        break;
      }
    }
  }
  El::mpi::Split(hash_comm,
                 node_num,
                 El::mpi::Rank(get_world_comm()),
                 m_node_comm);
  El::mpi::Free(hash_comm);

  // Set up list of ranks that are local.
  int node_comm_size = El::mpi::Size(m_node_comm);
  for (int i = 0; i < node_comm_size; ++i) {
    m_world_ranks_on_node.push_back(
      El::mpi::Translate(m_node_comm, i, get_world_comm()));
  }
}

void lbann_comm::setup_threads()
{
  const char* env_num_threads = getenv("OMP_NUM_THREADS");
  if (env_num_threads != nullptr) {
    m_threads_per_proc = std::atoi(env_num_threads);
  }
  else {
    m_threads_per_proc = std::thread::hardware_concurrency() / m_procs_per_node;
  }
  reset_threads();
}

void lbann_comm::reset_threads() const noexcept
{
  if (m_threads_per_proc != omp_get_max_threads()) {
    omp_set_num_threads(m_threads_per_proc);
  }
}

const El::mpi::Comm& lbann_comm::get_packed_group_comm(int num_per_group) const
{
  if (m_group_communicators.count(num_per_group) == 0) {
    // Ensure we can get an even number of groups.
    if (get_procs_in_world() % num_per_group != 0) {
      std::ostringstream err;
      err << "Cannot create a packed group comm with group size "
          << num_per_group << " out of " << get_procs_in_world()
          << " processes";
      LBANN_ERROR(err.str());
    }
    MPI_Comm comm;
    MPI_Comm_split(get_world_comm().GetMPIComm(),
                   get_rank_in_world() / (get_procs_in_world() / num_per_group),
                   0,
                   &comm);
    m_group_communicators.emplace(num_per_group, comm);
    MPI_Comm_free(&comm); // El::mpi::Comm duplicates internally.
  }
  return m_group_communicators[num_per_group];
}

void lbann_comm::lbann_comm_abort(std::string msg) const
{
  throw lbann_exception(msg);
}

#ifdef LBANN_HAS_ALUMINUM
::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op)
{
  if (op == El::mpi::SUM) {
    return ::Al::ReductionOperator::sum;
  }
  else if (op == El::mpi::PROD) {
    return ::Al::ReductionOperator::prod;
  }
  else if (op == El::mpi::MIN) {
    return ::Al::ReductionOperator::min;
  }
  else if (op == El::mpi::MAX) {
    return ::Al::ReductionOperator::max;
  }
  else {
    throw lbann_exception("Reduction operator not supported in Aluminum");
  }
}
#endif

int get_rank_in_world()
{
  int initialized = 0, finalized = 1, rank = -1;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  return rank;
}

#define PROTO(T)                                                               \
  template void lbann_comm::allreduce(El::AbstractMatrix<T>& m,                \
                                      const El::mpi::Comm& c,                  \
                                      El::mpi::Op op) const;                   \
  template void lbann_comm::allreduce(El::AbstractDistMatrix<T>& m,            \
                                      const El::mpi::Comm& c,                  \
                                      El::mpi::Op op) const;                   \
  template void lbann_comm::nb_allreduce(El::AbstractMatrix<T>& m,             \
                                         const El::mpi::Comm& c,               \
                                         Al::request& req,                     \
                                         El::mpi::Op op) const;                \
  template void lbann_comm::nb_allreduce(El::AbstractDistMatrix<T>& m,         \
                                         const El::mpi::Comm& c,               \
                                         Al::request& req,                     \
                                         El::mpi::Op op) const;                \
  template void lbann_comm::nb_reduce_scatter(                                 \
    const El::AbstractMatrix<T>& src,                                          \
    El::AbstractMatrix<T>& dst,                                                \
    const El::mpi::Comm& c,                                                    \
    Al::request& req,                                                          \
    El::mpi::Op op) const;                                                     \
  template void lbann_comm::nb_reduce_scatter(                                 \
    const El::AbstractDistMatrix<T>& src,                                      \
    El::AbstractDistMatrix<T>& dst,                                            \
    const El::mpi::Comm& c,                                                    \
    Al::request& req,                                                          \
    El::mpi::Op op) const
#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
