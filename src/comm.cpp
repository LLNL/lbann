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

#include "lbann/comm.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cuda.hpp"
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

void lbann_comm::allreduce(AbsMat& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) {
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }
  const int local_size = m.Height() * m.Width();
  bytes_sent += sizeof(DataType) * local_size;
#ifdef LBANN_HAS_ALUMINUM
  if (m.Width() > 1 && m.Height() != m.LDim()) {
    std::stringstream err;
    err << "Aluminum does not support allreduces "
        << "on non-contiguous matrices "
        << "(height=" << m.Height() << ", "
        << "width=" << m.Width() << ", "
        << "leading dim=" << m.LDim() << ")";
    LBANN_ERROR(err.str());
  }
  std::type_index t = std::type_index(typeid(::Al::MPIBackend));
#ifdef LBANN_HAS_GPU
  if (m.GetDevice() == El::Device::GPU) {
#ifdef AL_HAS_NCCL
    // We require NCCL for GPU matrices.
    t = std::type_index(typeid(::Al::NCCLBackend));
    // If available, use the MPI-CUDA backend for small matrices.
#ifdef AL_HAS_MPI_CUDA
    // Tuned for Sierra.
    if ((El::mpi::Size(c) >= 64 && local_size <= 4096) ||
        (El::mpi::Size(c) >= 128 && local_size <= 8192) ||
        (El::mpi::Size(c) >= 256 && local_size <= 32768) ||
        (El::mpi::Size(c) >= 512 && local_size <= 65536) ||
        (El::mpi::Size(c) >= 2048 && local_size <= 262144)) {
      t = std::type_index(typeid(::Al::MPICUDABackend));
    }
#endif  // AL_HAS_MPI_CUDA
#elif defined(AL_HAS_MPI_CUDA)
    t = std::type_index(typeid(::Al::MPICUDABackend));
#else
    throw lbann_exception("Allreduce on GPU matrix requires NCCL or MPI-CUDA"
                          " support in Aluminum");
#endif  // AL_HAS_NCCL
  }
#endif  // LBANN_HAS_GPU
  if (t == std::type_index(typeid(::Al::MPIBackend))) {
    ::Al::Allreduce<::Al::MPIBackend>(
      m.Buffer(),
      local_size,
      mpi_op_to_al_op(op),
      c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}));
  }
#ifdef AL_HAS_NCCL
  if (t == std::type_index(typeid(::Al::NCCLBackend))) {
    ::Al::Allreduce<::Al::NCCLBackend>(
      m.Buffer(),
      local_size,
      mpi_op_to_al_op(op),
      c.template GetComm<::Al::NCCLBackend>(
          SyncInfoFromMatrix(
              static_cast<El::Matrix<DataType,El::Device::GPU>&>(m))));
  }
#endif // AL_HAS_NCCL
#ifdef AL_HAS_MPI_CUDA
  if (t == std::type_index(typeid(::Al::MPICUDABackend))) {
    // Force the host-transfer algorithm for now.
    ::Al::Allreduce<::Al::MPICUDABackend>(
      m.Buffer(),
      local_size,
      mpi_op_to_al_op(op),
      c.template GetComm<::Al::MPICUDABackend>(
          SyncInfoFromMatrix(
              static_cast<El::Matrix<DataType,El::Device::GPU>&>(m))),
      ::Al::MPICUDAAllreduceAlgorithm::host_transfer);
  }
#endif  // AL_HAS_MPI_CUDA
#else
  El::AllReduce(m, c, op);
#endif
  bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);
}

void lbann_comm::allreduce(AbsDistMat& m,
                           const El::mpi::Comm& c,
                           El::mpi::Op op) {
  allreduce(m.Matrix(), c, op);
}

void lbann_comm::nb_allreduce(AbsMat& m,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              El::mpi::Op op) {
  if (El::mpi::Size(c) == 1 || m.Height() < 1 || m.Width() < 1) {
    return;
  }
#ifdef LBANN_HAS_ALUMINUM
  const int local_size = m.Height() * m.Width();
  bytes_sent += sizeof(DataType) * local_size;
  if (m.Width() > 1 && m.Height() != m.LDim()) {
    std::stringstream err;
    err << "Aluminum does not support allreduces "
        << "on non-contiguous matrices "
        << "(height=" << m.Height() << ", "
        << "width=" << m.Width() << ", "
        << "leading dim=" << m.LDim() << ")";
    LBANN_ERROR(err.str());
  }
  std::type_index t = std::type_index(typeid(::Al::MPIBackend));
#ifdef LBANN_HAS_GPU
  if (m.GetDevice() == El::Device::GPU) {
#ifdef AL_HAS_NCCL
    // We require NCCL for GPU matrices.
    t = std::type_index(typeid(::Al::NCCLBackend));
    // If available, use the MPI-CUDA backend for small matrices.
#ifdef AL_HAS_MPI_CUDA
    // Tuned for Sierra.
    if ((El::mpi::Size(c) >= 64 && local_size <= 4096) ||
        (El::mpi::Size(c) >= 128 && local_size <= 8192) ||
        (El::mpi::Size(c) >= 256 && local_size <= 32768) ||
        (El::mpi::Size(c) >= 512 && local_size <= 65536) ||
        (El::mpi::Size(c) >= 2048 && local_size <= 262144)) {
      t = std::type_index(typeid(::Al::MPICUDABackend));
    }
#endif  // AL_HAS_MPI_CUDA
#elif defined(AL_HAS_MPI_CUDA)
    t = std::type_index(typeid(::Al::MPICUDABackend));
#else
    throw lbann_exception("Allreduce on GPU matrix requires NCCL or MPI-CUDA"
                          " support in Aluminum");
#endif  // AL_HAS_NCCL
  }
#endif  // LBANN_HAS_GPU
  if (t == std::type_index(typeid(::Al::MPIBackend))) {
    ::Al::NonblockingAllreduce<::Al::MPIBackend>(
      m.Buffer(),
      local_size,
      mpi_op_to_al_op(op),
      c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}),
      req.mpi_req);
  }
  /// @todo MPI-CUDA backend
#ifdef AL_HAS_NCCL
  if (t == std::type_index(typeid(::Al::NCCLBackend))) {
    ::Al::NonblockingAllreduce<::Al::NCCLBackend>(
      m.Buffer(),
      local_size,
      mpi_op_to_al_op(op),
      c.template GetComm<::Al::NCCLBackend>(
          SyncInfoFromMatrix(
              static_cast<El::Matrix<DataType,El::Device::GPU>&>(m))),
      req.nccl_req);
  }
#endif // AL_HAS_NCCL
#ifdef AL_HAS_MPI_CUDA
  if (t == std::type_index(typeid(::Al::MPICUDABackend))) {
    // Force the host-transfer algorithm for now.
    ::Al::NonblockingAllreduce<::Al::MPICUDABackend>(
      m.Buffer(),
      local_size,
      mpi_op_to_al_op(op),
      c.template GetComm<::Al::MPICUDABackend>(
          SyncInfoFromMatrix(
              static_cast<El::Matrix<DataType,El::Device::GPU>&>(m))),
      req.mpicuda_req,
      ::Al::MPICUDAAllreduceAlgorithm::host_transfer);
  }
#endif  // AL_HAS_MPI_CUDA
  bytes_received += sizeof(DataType) * local_size * (El::mpi::Size(c) - 1);
#else
  allreduce(m, c, op);
#endif // LBANN_HAS_ALUMINUM
}

void lbann_comm::nb_allreduce(AbsDistMat& m,
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

#ifdef LBANN_HAS_ALUMINUM
::Al::ReductionOperator lbann_comm::mpi_op_to_al_op(El::mpi::Op op) {
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

void lbann_comm::lbann_comm_abort(std::string msg) {
  throw lbann_exception(msg);
}

int get_rank_in_world() {
  int initialized = 0, finalized = 1, rank = -1;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  return rank;
}

}  // namespace lbann
