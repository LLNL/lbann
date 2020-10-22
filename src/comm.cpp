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
// lbann_comm .hpp .cpp - LBANN communication utilities
////////////////////////////////////////////////////////////////////////////////

#define LBANN_COMM_INSTANTIATE
#include "lbann/comm.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "mpi.h"
#include "omp.h"
#include <sstream>
#include <thread>

namespace lbann {

// Error utility macro
#ifdef LBANN_DEBUG
#define checkMPI(mpi_call) {                                            \
    const int status = mpi_call;                                        \
    if(status != MPI_SUCCESS) {                                         \
      char error_string[MPI_MAX_ERROR_STRING];                          \
      int error_string_len;                                             \
      MPI_Error_string(status, error_string, &error_string_len);        \
      std::cerr << "MPI error: " << std::string(error_string, error_string_len) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      throw lbann_exception("MPI error");                        \
    }                                                                   \
  }
#else
#define checkMPI(status) status
#endif // #ifdef LBANN_DEBUG

lbann_comm::lbann_comm(int ppm, El::mpi::Comm world) :
  world_comm(std::move(world)), grid(nullptr), procs_per_trainer(ppm), num_trainer_barriers(0),
  num_intertrainer_barriers(0), num_global_barriers(0), bytes_sent(0),
  bytes_received(0) {
#ifdef LBANN_HAS_ALUMINUM
  // Don't have argc/argv here, but MPI should already be init'd.
  int argc_dummy = 0;
  char** argv_dummy = nullptr;
  ::Al::Initialize(argc_dummy, argv_dummy);
#endif
  // Set up the initial trainer split
  split_trainers(procs_per_trainer);

  // Initialize node communicators
  setup_node_comm();
  procs_per_node = El::mpi::Size(node_comm);
  rank_in_node = El::mpi::Rank(node_comm);

  // Setup threads
  setup_threads();
}

lbann_comm::~lbann_comm() {
  delete grid;
  El::mpi::Free(trainer_comm);
  El::mpi::Free(intertrainer_comm);
  El::mpi::Free(node_comm);
#ifdef LBANN_HAS_ALUMINUM
  ::Al::Finalize();
#endif
}

void lbann_comm::split_trainers(int ppm) {
  int world_size = El::mpi::Size(get_world_comm());
  procs_per_trainer = ppm;
  if (ppm == 0) {
    procs_per_trainer = world_size;
  }
  // Check if parameters are valid
  if (procs_per_trainer > world_size) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: Not enough processes to create one trainer; procs_per_trainer: " +
      std::to_string(procs_per_trainer) + " is larger than world_size: " +
      std::to_string(world_size));
  }
  if (world_size % procs_per_trainer != 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: Procs per trainer does not divide total number of procs; procs_per_trainer: " +
      std::to_string(procs_per_trainer) + " total number of procs (world size): " +
      std::to_string(world_size));
  }

  num_trainers = world_size / procs_per_trainer;
  trainer_rank = El::mpi::Rank(get_world_comm()) / procs_per_trainer;
  rank_in_trainer = El::mpi::Rank(get_world_comm()) % procs_per_trainer;

  // Initialize trainer and intertrainer communicators
  El::mpi::Split(get_world_comm(), trainer_rank, rank_in_trainer, trainer_comm);
  El::mpi::Split(get_world_comm(), rank_in_trainer, trainer_rank,
                 intertrainer_comm);

  // Initialize Elemental grid
  if (grid != nullptr) {
    delete grid;
  }
  grid = new Grid(trainer_comm.GetMPIComm());
}

void lbann_comm::intertrainer_sum_matrix(AbsMat& mat) {
  bytes_sent += sizeof(DataType) * mat.Height() * mat.Width();
  El::AllReduce(mat, intertrainer_comm, El::mpi::SUM);
  bytes_received += sizeof(DataType) * mat.Height() * mat.Width();
}

void lbann_comm::intertrainer_sum_matrix(AbsDistMat& mat) {
  allreduce(mat, intertrainer_comm, El::mpi::SUM);
}

namespace {

template <typename BackendT>
struct BackendTag {};

#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)
auto GetRequest(Al::request& r, BackendTag<Al::dummy_backend>)
    -> typename Al::dummy_backend::req_type
{
    return Al::dummy_backend::null_req;
}

auto GetRequest(Al::request& r, BackendTag<::Al::MPIBackend>)
    -> typename ::Al::MPIBackend::req_type&
{
    return r.mpi_req;
}
void UpdateRequest(typename ::Al::MPIBackend::req_type&,
                   El::SyncInfo<El::Device::CPU> const&) noexcept
{
}

#ifdef AL_HAS_NCCL
auto GetRequest(Al::request& r, BackendTag<::Al::NCCLBackend>) noexcept
    -> typename ::Al::NCCLBackend::req_type&
{
    return r.nccl_req;
}
void UpdateRequest(typename ::Al::NCCLBackend::req_type& req,
                   El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_NCCL

#ifdef AL_HAS_MPI_CUDA
auto GetRequest(Al::request& r, BackendTag<::Al::MPICUDABackend>) noexcept
    -> typename ::Al::MPICUDABackend::req_type&
{
    return r.mpicuda_req;
}
void UpdateRequest(typename ::Al::MPICUDABackend::req_type& req,
                   El::SyncInfo<El::Device::GPU> const& si) noexcept
{
  if (req)
    req->orig_stream = si.Stream();
}
#endif // AL_HAS_MPI_CUDA
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
                    El::mpi::Op const& op) {
  return El::AllReduce(m, c, op);
}

template <typename T, El::Device D>
void nb_allreduce_impl(El::Matrix<T, D>& m,
                       const El::mpi::Comm& c,
                       Al::request&,
                       El::mpi::Op const& op) {
  return El::AllReduce(m, c, op);
}

#if defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)

template <typename T, typename BackendT,
          El::EnableWhen<
            El::AluminumSupportsBackendAndCollective<
              T, El::Collective::ALLREDUCE, BackendT>,
            int> = 0>
void allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                        const El::mpi::Comm& c,
                        El::mpi::Op const& op,
                        BackendTag<BackendT>,
                        typename BackendT::allreduce_algo_type algo
                        = BackendT::allreduce_algo_type::automatic) {
  const auto local_size = m.Height() * m.Width();
  ::Al::Allreduce<BackendT>(
    m.Buffer(),
    local_size,
    mpi_op_to_al_op(op),
    c.template GetComm<BackendT>(El::SyncInfoFromMatrix(m)),
    algo);
}

template <typename T, typename BackendT,
          El::EnableWhen<
            El::AluminumSupportsBackendAndCollective<
              T, El::Collective::ALLREDUCE, BackendT>,
            int> = 0>
void nb_allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                           const El::mpi::Comm& c,
                           Al::request& req,
                           El::mpi::Op const& op,
                           BackendTag<BackendT> const& tag,
                           typename BackendT::allreduce_algo_type algo
                           = BackendT::allreduce_algo_type::automatic) {
  const auto local_size = m.Height() * m.Width();
  const auto& syncinfo = El::SyncInfoFromMatrix(m);
  auto& request = GetRequest(req, tag);
  ::Al::NonblockingAllreduce<BackendT>(
    m.Buffer(),
    local_size,
    mpi_op_to_al_op(op),
    c.template GetComm<BackendT>(syncinfo),
    request,
    algo);
  UpdateRequest(request, syncinfo);
}

template <typename T, typename BackendT,
          El::EnableUnless<
            El::AluminumSupportsBackendAndCollective<
              T, El::Collective::ALLREDUCE, BackendT>,
            int> = 0>
void nb_allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                           const El::mpi::Comm& c,
                           Al::request& req,
                           El::mpi::Op const& op,
                           BackendTag<BackendT> const& tag,
                           typename BackendT::allreduce_algo_type algo
                           = BackendT::allreduce_algo_type::automatic) {
  El::AllReduce(m, c, op);
}

template <typename T, typename BackendT,
          El::EnableUnless<
            El::AluminumSupportsBackendAndCollective<
              T, El::Collective::ALLREDUCE, BackendT>,
            int> = 0>
void allreduce_aluminum(El::Matrix<T, El::Device::GPU>& m,
                        const El::mpi::Comm& c,
                        El::mpi::Op const& op,
                        BackendTag<BackendT>,
                        typename BackendT::allreduce_algo_type
                        = BackendT::allreduce_algo_type::automatic) {
  // We cannot dispatch with this backend directly to Aluminum. Let
  // Elemental handle it.
  El::AllReduce(m, c, op);
}

template <typename T>
void allreduce_impl(El::Matrix<T, El::Device::GPU>& m,
                    El::mpi::Comm const& c,
                    El::mpi::Op const& op) {
  return El::AllReduce(m, c, op);
}

template <typename T>
void nb_allreduce_impl(El::Matrix<T, El::Device::GPU>& m,
                       El::mpi::Comm const& c,
                       Al::request& req,
                       El::mpi::Op const& op) {
  if (m.Width() > 1 && m.Height() != m.LDim()) {
    // Aluminum doesn't do allreduces on strided matrices
    return El::AllReduce(m, c, op);
  }

#if defined(AL_HAS_NCCL)
  return nb_allreduce_aluminum(
    m, c, req, op,
    BackendTag<::Al::NCCLBackend>{});
#elif defined(AL_HAS_MPI_CUDA)
  return nb_allreduce_aluminum(
    m, c, req, op,
    BackendTag<::Al::MPICUDABackend>{},
    ::Al::MPICUDABackend::allreduce_algo_type::host_transfer);
#else
  // At this point just call Elemental again
  return El::AllReduce(m, c, op);
#endif
}

#endif // defined(LBANN_HAS_GPU) && defined(LBANN_HAS_ALUMINUM)
}// namespace <anon>

template <typename TensorDataType>
void lbann_comm::allreduce(El::AbstractMatrix<TensorDataType>& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) {
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }

  const int local_size = m.Height() * m.Width();
  bytes_sent += sizeof(DataType) * local_size;
  bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (m.GetDevice()) {
  case El::Device::CPU:
    return allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(m), c, op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m), c, op);
#endif // LBANN_HAS_GPU
  }

}

template <typename TensorDataType>
void lbann_comm::allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) {
  allreduce(m.Matrix(), c, op);
}

template <typename TensorDataType>
void lbann_comm::nb_allreduce(El::AbstractMatrix<TensorDataType>& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) {
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }

  const int local_size = m.Height() * m.Width();
  bytes_sent += sizeof(DataType) * local_size;
  bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);

  switch (m.GetDevice()) {
  case El::Device::CPU:
    return nb_allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(m), c, req, op);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return nb_allreduce_impl(
      static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(m), c, req, op);
#endif // LBANN_HAS_GPU
  }

}

template <typename TensorDataType>
void lbann_comm::nb_allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) {
  nb_allreduce(m.Matrix(), c, req, op);
}

void lbann_comm::wait(Al::request& req) {
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
#ifdef AL_HAS_MPI_CUDA
  if (req.mpicuda_req != Al::mpicuda_null_req) {
    // Note this does not block the host.
    ::Al::Wait<::Al::MPICUDABackend>(req.mpicuda_req);
  }
#endif  // AL_HAS_MPI_CUDA
#endif // LBANN_HAS_ALUMINUM
}

bool lbann_comm::test(Al::request& req) {
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
#ifdef AL_HAS_MPI_CUDA
  if (req.mpicuda_req != Al::mpicuda_null_req) {
    req_test = req_test && ::Al::Test<::Al::MPICUDABackend>(req.mpicuda_req);
  }
#endif  // AL_HAS_MPI_CUDA
#endif // LBANN_HAS_ALUMINUM
  return req_test;
}

void lbann_comm::intertrainer_broadcast_matrix(AbsMat& mat, int root) {
  El::Broadcast(mat, intertrainer_comm, root);
}

void lbann_comm::intertrainer_broadcast_matrix(AbsDistMat& mat, int root) {
  El::Broadcast(mat, intertrainer_comm, root);
}

template<>
void lbann_comm::broadcast<std::string>(const int root, std::string& str, const El::mpi::Comm& c) {
  std::vector<char> data(str.begin(), str.end());
  broadcast(root, data, c);
  str.assign(data.begin(), data.end());
}

void lbann_comm::intertrainer_barrier() {
  ++num_intertrainer_barriers;
  barrier(intertrainer_comm);
}

void lbann_comm::trainer_barrier() {
  ++num_trainer_barriers;
  barrier(trainer_comm);
}

void lbann_comm::global_barrier() {
  ++num_global_barriers;
  barrier(get_world_comm());
}

void lbann_comm::barrier(const El::mpi::Comm& c) {
  El::mpi::Barrier(c);
}

void lbann_comm::send(const AbsMat& mat, int trainer, int rank) {
  El::Send(mat, get_world_comm(), get_world_rank(trainer, rank));
}

void lbann_comm::send(const DistMat& mat, int trainer, int rank) {
  send(mat.LockedMatrix(), trainer, rank);
}

void lbann_comm::nb_send(const AbsMat& mat, int trainer, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_send(mat.LockedBuffer(), mat.Height() * mat.Width(), trainer, rank, req);
}

void lbann_comm::nb_send(const DistMat& mat, int trainer, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_send(mat.LockedBuffer(), mat.LocalHeight() * mat.LocalWidth(), trainer,
          rank, req);
}

void lbann_comm::recv(AbsMat& mat, int trainer, int rank) {
  El::Recv(mat, get_world_comm(), get_world_rank(trainer, rank));
}

void lbann_comm::recv(DistMat& mat, int trainer, int rank) {
  recv(mat.Matrix(), trainer, rank);
}

void lbann_comm::recv(AbsMat& mat) {
  El::Recv(mat, get_world_comm(), El::mpi::ANY_SOURCE);
}

void lbann_comm::recv(DistMat& mat) {
  recv(mat.Matrix());
}

void lbann_comm::nb_recv(AbsMat& mat, int trainer, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), trainer, rank, req);
}

void lbann_comm::nb_recv(DistMat& mat, int trainer, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), trainer, rank, req);
}

void lbann_comm::nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), req);
}

void lbann_comm::nb_recv(DistMat& mat, El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), req);
}

void lbann_comm::setup_node_comm() {

  // Get string specifying compute node
  char node_name[MPI_MAX_PROCESSOR_NAME];
  int node_name_len;
  checkMPI(MPI_Get_processor_name(node_name, &node_name_len));
  const std::string node_string(node_name);

  // Hash node names and split MPI processes
  int hash = std::hash<std::string>()(node_string);
  hash = hash >= 0 ? hash : -hash;  // Make sure hash is non-negative
  El::mpi::Comm hash_comm;
  El::mpi::Split(get_world_comm(), hash,
                 El::mpi::Rank(get_world_comm()), hash_comm);
  const int hash_comm_size = El::mpi::Size(hash_comm);

  // Compare node names and split MPI processes
  auto *node_name_list = new char[hash_comm_size*MPI_MAX_PROCESSOR_NAME];
  checkMPI(MPI_Allgather(node_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         node_name_list, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         hash_comm.GetMPIComm()));
  int node_num = El::mpi::Rank(hash_comm);
  for(int i=0; i<hash_comm_size; ++i) {
    const std::string other_node_string(node_name_list + i*MPI_MAX_PROCESSOR_NAME);
    if(node_string == other_node_string) {
      node_num = i;
      break;
    }
  }
  delete[] node_name_list;
  El::mpi::Split(hash_comm, node_num, El::mpi::Rank(get_world_comm()),
                 node_comm);
  El::mpi::Free(hash_comm);

  // Set up list of ranks that are local.
  int node_comm_size = El::mpi::Size(node_comm);
  for (int i = 0; i < node_comm_size; ++i) {
    world_ranks_on_node.push_back(
      El::mpi::Translate(node_comm, i, get_world_comm()));
  }
}

void lbann_comm::setup_threads() {
  const char* env_num_threads = getenv("OMP_NUM_THREADS");
  if (env_num_threads != nullptr){
    threads_per_proc = std::atoi(env_num_threads);
  }
  else {
    threads_per_proc = std::thread::hardware_concurrency() / procs_per_node;
  }
  reset_threads();
}

void lbann_comm::reset_threads() {
  if (threads_per_proc != omp_get_max_threads()) {
    omp_set_num_threads(threads_per_proc);
  }
}

const El::mpi::Comm& lbann_comm::get_packed_group_comm(int num_per_group) const {
  if (group_communicators.count(num_per_group) == 0) {
    // Ensure we can get an even number of groups.
    if (get_procs_in_world() % num_per_group != 0) {
      std::stringstream err;
      err << "Cannot create a packed group comm with group size "
          << num_per_group
          << " out of " << get_procs_in_world()
          << " processes";
      LBANN_ERROR(err.str());
    }
    MPI_Comm comm;
    MPI_Comm_split(
      get_world_comm().GetMPIComm(),
      get_rank_in_world() / (get_procs_in_world() / num_per_group),
      0, &comm);
    group_communicators.emplace(num_per_group, comm);
    MPI_Comm_free(&comm);  // El::mpi::Comm duplicates internally.
  }
  return group_communicators[num_per_group];
}

void lbann_comm::lbann_comm_abort(std::string msg) {
  throw lbann_exception(msg);
}

#ifdef LBANN_HAS_ALUMINUM
::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op) {
  if (op == El::mpi::SUM) {
    return ::Al::ReductionOperator::sum;
  } else if (op == El::mpi::PROD) {
    return ::Al::ReductionOperator::prod;
  } else if (op == El::mpi::MIN) {
    return ::Al::ReductionOperator::min;
  } else if (op == El::mpi::MAX) {
    return ::Al::ReductionOperator::max;
  } else {
    throw lbann_exception("Reduction operator not supported in Aluminum");
  }
}
#endif

int get_rank_in_world() {
  int initialized = 0, finalized = 1, rank = -1;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  return rank;
}

#define PROTO(T)                                                                            \
  template void lbann_comm::allreduce<T>(                                                   \
    El::AbstractMatrix<T>& m, const El::mpi::Comm& c, El::mpi::Op op);                      \
  template void lbann_comm::allreduce<T>(                                                   \
    El::AbstractDistMatrix<T>& m, const El::mpi::Comm& c, El::mpi::Op op);                  \
  template void lbann_comm::nb_allreduce<T>(                                                \
    El::AbstractMatrix<T>& m, const El::mpi::Comm& c, Al::request& req, El::mpi::Op op);    \
  template void lbann_comm::nb_allreduce<T>(                                                \
    El::AbstractDistMatrix<T>& m, const El::mpi::Comm& c, Al::request& req, El::mpi::Op op)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
