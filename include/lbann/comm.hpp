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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_COMM_HPP_INCLUDED
#define LBANN_COMM_HPP_INCLUDED

#include <vector>
#include <map>
#include <typeindex>
#include "base.hpp"
#ifdef LBANN_HAS_CUDA
#include <cuda_runtime.h>
#endif // LBANN_HAS_CUDA
#ifdef LBANN_HAS_ALUMINUM
#include <Al.hpp>
#endif // LBANN_HAS_ALUMINUM
#include "detect_El_mpi.hpp"

namespace lbann {

namespace Al {

/** Dummy Aluminum backend. */
class dummy_backend {
public:
  using req_type = int;
  static constexpr req_type null_req = 0;
};

// Define aliases for Aluminum backends
#ifdef LBANN_HAS_ALUMINUM
using mpi_backend = ::Al::MPIBackend;
#else
using mpi_backend = lbann::Al::dummy_backend;
#endif // LBANN_HAS_ALUMINUM
using mpi_req_type = mpi_backend::req_type;
static const mpi_req_type mpi_null_req = mpi_backend::null_req;
/// @todo MPI-CUDA backend
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)
using nccl_backend = ::Al::NCCLBackend;
// LBANN does its own synchronization on this.
#else
using nccl_backend = lbann::Al::dummy_backend;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)
using nccl_req_type = nccl_backend::req_type;
static const nccl_req_type nccl_null_req = nccl_backend::null_req;
#if defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_MPI_CUDA)
using mpicuda_backend = ::Al::MPICUDABackend;
#else
using mpicuda_backend = lbann::Al::dummy_backend;
#endif  // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_MPI_CUDA)
using mpicuda_req_type = mpicuda_backend::req_type;
static const mpicuda_req_type mpicuda_null_req = mpicuda_backend::null_req;

/** Wrapper for Aluminum non-blocking routine requests. */
struct request {
  mpi_req_type mpi_req = mpi_null_req;
  nccl_req_type nccl_req = nccl_null_req;
  mpicuda_req_type mpicuda_req = mpicuda_null_req;
};

} // namespace Al

/* Notes on Synchronization
 *
 * The updated interface exposes a synchronization handle/device
 * tagging mechanism used by Hydrogen: El::SyncInfo<D>, where D is an
 * El::Device. When operating on Matrix objects, this should be
 * handled automagically, assuming the Matrix is setup properly. Users
 * must be aware of this when making MPI calls through Hydrogen or
 * through lbann_comm with raw data buffers (T[]).
 *
 * When dealing with El::Matrix objects, users should be aware of the
 * following. There is no synchronization for CPU objects
 * (El::SyncInfo<El::Device::CPU> is an empty struct), but GPU Matrix
 * objects now have an associated stream and event. These are
 * GPUManager::Stream() and GPUManager::Event() by default, resp., but
 * can be overriden by a user. Note: the Matrix never owns these; it
 * will not free these resources at destruction. There are many
 * methods in which multiple El::Matrix objects might interact. This
 * should work properly; otherwise, report bugs to benson31.
 *
 * When dealing with raw data (T[]), users should be aware of the
 * following. In the near future, all El::mpi functions will have an
 * El::SyncInfo object as their last parameter, and it will be a
 * required parameter. In lbann_comm, this means that when the call
 * trickles down to an El::mpi function, an appropriate El::SyncInfo
 * must be available. Since many of LBANN's uses of this interface are
 * for communicating CPU buffers, there is "shortcut" API that assumes
 * the data is CPU memory, thus providing the default
 * El::SyncInfo<El::Device::CPU> object to El::mpi. If a user wishes
 * to communicate GPU data, they must use the "full" API, which adds a
 * final El::SyncInfo parameter to the function. This ensures the
 * appropriate synchronization semantics, especially when working with
 * Aluminum as the communication frontend.
 */


/**
 * Manage communication.
 * This supports separate trainers, each of which are split over potentially
 * several processes. Every trainer is split over the same number of processes.
 * The corresponding processes between trainers are on the "inter-trainer
 * communicator".
 * You can also do point-to-point or broadcast communication to arbitrary sets
 * of processes.
 */
class lbann_comm {
 public:
  /**
   * Init communicators for trainers each with procs_per_trainer processes,
   * defaulting to every process in one trainer.
   */
  lbann_comm(int procs_per_trainer = 0,
             El::mpi::Comm world = El::mpi::COMM_WORLD.GetMPIComm());
  /** Don't allow copying; it doesn't make sense for the communicator. */
  lbann_comm(const lbann_comm&) = delete;
  /** Don't allow assignment; it doesn't make sense for the communicator. */
  lbann_comm& operator=(const lbann_comm&) = delete;
  ~lbann_comm();

  /**
   * Split communicators so each trainer has procs_per_trainer processes.
   * If you call this multiple times, it will invalidate existing grids
   * and communicators.
   */
  void split_trainers(int procs_per_trainer);

  /** Get which trainer this process is in. */
  inline int get_trainer_rank() const {
    return trainer_rank;
  }
  /** Get the rank of this process in its trainer. */
  inline int get_rank_in_trainer() const {
    return rank_in_trainer;
  }
  /** Get my rank in COMM_WORLD. */
  inline int get_rank_in_world() const {
    return El::mpi::Rank(get_world_comm());
  }
  /** Return the COMM_WORLD rank of the rank'th processor in trainer. */
  inline int get_world_rank(int trainer, int rank) const {
    return procs_per_trainer * trainer + rank;
  }
  /** Return the rank of the master process in this trainer. */
  inline int get_trainer_master() const {
    return 0;
  }
  /** Return the rank of the inter-trainer master process. */
  inline int get_intertrainer_master() const {
    return 0;
  }
  /** Return the rank of the world master process. */
  inline int get_world_master() const {
    return 0;
  }
  /** Return true if this process is the master process in its trainer. */
  inline bool am_trainer_master() const {
    return get_rank_in_trainer() == get_trainer_master();
  }
  /** Return true if this process is the world master process. */
  inline bool am_world_master() const {
    return get_rank_in_world() == get_world_master();
  }
  /** Return a grid to use for this trainer. */
  inline Grid& get_trainer_grid() {
    return *grid;
  }
  /** Return a read-only grid to use for this trainer. */
  inline const Grid& get_trainer_grid() const {
    return *grid;
  }
  /** Return the total number of trainers. */
  inline int get_num_trainers() const {
    return num_trainers;
  }
  /* Return the number of processes in a trainer. */
  inline int get_procs_per_trainer() const {
    return procs_per_trainer;
  }
  /** Return the number of processes in a compute node. */
  inline int get_procs_per_node() const {
    return procs_per_node;
  }
  /** Return the total number of ranks. */
  inline int get_procs_in_world() const {
    return El::mpi::Size(get_world_comm());
  }
  /** Return the rank of this process within its compute node. */
  inline int get_rank_in_node() const {
    return rank_in_node;
  }
  /** Return true if rank (in COMM_WORLD) is on this compute node. */
  inline bool is_world_rank_on_node(int rank) const {
    return std::find(world_ranks_on_node.begin(),
                     world_ranks_on_node.end(),
                     rank) != world_ranks_on_node.end();
  }

  /** Get default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided.
   */
  inline int get_default_threads_per_proc() const {
    return threads_per_proc;
  }

  /** Reset the number of threads per process to the default. */
  void reset_threads();

  /** Perform a sum reduction of mat over the inter-trainer communicator. */
  void intertrainer_sum_matrix(AbsMat& mat);
  void intertrainer_sum_matrix(AbsDistMat& mat);
  /** Broadcast mat over the inter-trainer communicator starting from root. */
  void intertrainer_broadcast_matrix(AbsMat& mat, int root);
  void intertrainer_broadcast_matrix(AbsDistMat& mat, int root);

  /// Broadcast a scalar value over an arbitrary communicator
  template < typename T, bool S = is_instantiated_El_mpi_type<T>::value >
  void broadcast(int root, T& val, const El::mpi::Comm& c);

  template <typename T>
  void broadcast_custom(int root, T& val, const El::mpi::Comm& c) const;
  template <typename T>
  void broadcast_native(int root, T& val, const El::mpi::Comm& c) const;

  /// World broadcast of a scalar.
  template <typename T>
  void world_broadcast(int root, T& val) {
    broadcast(root, val, get_world_comm());
  }
  /// Inter-trainer broadcast of a scalar.
  template <typename T>
  void intertrainer_broadcast(int root, T& val) {
    broadcast(root, val, get_intertrainer_comm());
  }
  /// Within-trainer broadcast of a scalar.
  template <typename T>
  void trainer_broadcast(int root, T& val) {
    broadcast(root, val, get_trainer_comm());
  }

  /**
   * Broadcast a buffer over an arbitrary communicator assuming that
   * the buffer space is already allocated.
   */

  // Default to cpu memory
  template <typename T>
  void broadcast(const int root, T* data, const int count, const El::mpi::Comm& c) {
      broadcast(root, data, count, c, El::SyncInfo<El::Device::CPU>{});
  }

  template < typename T, El::Device D, bool S = is_instantiated_El_mpi_type<T>::value >
  void broadcast(const int root, T* data, const int count, const El::mpi::Comm& c,
                 El::SyncInfo<D> const& syncInfo);

  /// World broadcast of a buffer.
  template <typename T>
  void world_broadcast(const int root, T* data, const int count) {
    world_broadcast(root, data, count, El::SyncInfo<El::Device::CPU>{});
  }

  template <typename T, El::Device D>
  void world_broadcast(const int root, T* data, const int count,
                       El::SyncInfo<D> const& syncInfo) {
    broadcast(root, data, count, get_world_comm(), syncInfo);
  }
  /// Inter-trainer broadcast of a buffer.
  template <typename T>
  void intertrainer_broadcast(const int root, T* data, const int count) {
    intertrainer_broadcast(root, data, count, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void intertrainer_broadcast(const int root, T* data, const int count,
                            El::SyncInfo<D> const& syncInfo) {
    broadcast(root, data, count, get_intertrainer_comm(), syncInfo);
  }
  /// Within-trainer broadcast of a buffer.
  template <typename T>
  void trainer_broadcast(const int root, T* data, const int count) {
    trainer_broadcast(root, data, count, El::SyncInfo<El::Device::CPU>{});
  }

  template <typename T, El::Device D>
  void trainer_broadcast(const int root, T* data, const int count,
                       El::SyncInfo<D> const& syncInfo) {
    broadcast(root, data, count, get_trainer_comm(), syncInfo);
  }

  /**
   * Resize vector<> over an arbitrary communicator to match the one on root.
   */
  template <typename T>
  size_t resize(const int root, std::vector<T> &data, const El::mpi::Comm& c) {
    auto const rank_c = El::mpi::Rank(c);
    size_t count = data.size();
    El::mpi::Broadcast(&count, 1, root, c, El::SyncInfo<El::Device::CPU>{});
    count_bytes_broadcast(sizeof(size_t), rank_c, root);
    data.resize(count);
    return count;
  }

  /**
   * Broadcast vector<> over an arbitrary communicator;
   * vector<> for non-root processes will be resized as needed.
   */
  template <typename T>
  void broadcast(const int root, std::vector<T> &data, const El::mpi::Comm& c) {
    const int count = static_cast<int>(resize(root, data, c));
    if (count <= 0) {
      return;
    }
    broadcast<T>(root, data.data(), count, c, El::SyncInfo<El::Device::CPU>{});
  }
  /// Broadcast vector<> to world.
  template <typename T>
  void world_broadcast(int root, std::vector<T> &data) {
    broadcast(root, data, get_world_comm());
  }
  /**
   * Broadcast vector<> within trainer;
   * vector<> for non-root processes will be resized as needed.
   */
  /// Broadcast vector<> across trainers.
  template <typename T>
  void intertrainer_broadcast(int root, std::vector<T> &data) {
    broadcast(root, data, get_intertrainer_comm());
  }
  /// Broadcast vector<> within trainer.
  template <typename T>
  void trainer_broadcast(int root, std::vector<T> &data) {
    broadcast(root, data, get_trainer_comm());
  }

  /**
   * Keep track of the number of broadcast bytes transmitted and received
   */
  void count_bytes_broadcast(const size_t bytes, const int rank, const int root) {
    if (rank == root) {
      bytes_sent += bytes;
    } else {
      bytes_received += bytes;
    }
  }

  /** Allgather over an arbitrary communicator */
  template <typename T>
  void all_gather(const T* src, int src_count, T* rcv, int rcv_count, const El::mpi::Comm& c) {
    all_gather(src, src_count, rcv, rcv_count, c,
                   El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void all_gather(const T* src, int src_count, T* rcv, int rcv_count, const El::mpi::Comm& c,
                  El::SyncInfo<D> const& syncInfo) {
    El::mpi::AllGather<T>(src, src_count, rcv, rcv_count, c, syncInfo);
  }

  /**
   * Allgatherv over an arbitrary communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(std::vector<T> &src, std::vector<T> &rcs, std::vector<int> &rcv_counts, std::vector<int> &rcv_disp, const El::mpi::Comm& c) {
    if (src.size() == 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
              << "all_gather for vector<>: vector.size() == 0;\n"
              << "this doesn't work!";
      lbann_comm_abort(err.str());
    }
    El::mpi::AllGather<T>(src.data(), src.size(), rcs.data(), rcv_counts.data(), rcv_disp.data(), c, El::SyncInfo<El::Device::CPU>{});
  }
  /**
   * Allgatherv over a trainer communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void trainer_all_gather(std::vector<T> &src, std::vector<T> &rcs, std::vector<int> &rcv_counts, std::vector<int> &rcv_disp) {
    all_gather(src, rcs, rcv_counts, rcv_disp, get_trainer_comm());
  }
  /**
   * Allgather for a single element over an arbitrary communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(T &src, std::vector<T> &data, const El::mpi::Comm& c) {
    El::mpi::AllGather(&src, 1, data.data(), 1, c,
                       El::SyncInfo<El::Device::CPU>{});
  }
  /**
   * Allgather for a single element over the trainer communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void trainer_all_gather(T &src, std::vector<T> &data) {
    all_gather(src, data, get_trainer_comm());
  }

  /** Within-trainer scalar gather (for non-root processes). */
  template <typename T>
  void trainer_gather(T snd, int root) {
    gather(snd, root, trainer_comm);
  }
  /** Within-trainer scalar gather (for root processes). */
  template <typename T>
  void trainer_gather(T snd, T* rcv) {
    gather(snd, rcv, trainer_comm);
  }
  /** Within-trainer scalar-array gather (for non-root processes). */
  template <typename T>
  void trainer_gather(T* snd, int count, int root) {
    gather(snd, count, root, trainer_comm);
  }
  /** Within-trainer scalar-array gather (for root processes). */
  template <typename T>
  void trainer_gather(T* snd, int count, T* rcv) {
    gather(snd, count, rcv, trainer_comm);
  }
  /** Within-trainer variable-length-array gather (for non-root processes). */
  template <typename T>
  void trainer_gatherv(T* snd, int count, int root) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Gather(snd, count, nullptr, nullptr, nullptr, root,
                    trainer_comm);
  }
  template <typename T>
  void trainer_gatherv(T* snd, int count, T* rcv, int* rcv_counts,
                       int* rcv_displacements) {
    El::mpi::Gather(snd, count, rcv, rcv_counts, rcv_displacements,
                    get_rank_in_trainer(), trainer_comm);
    bytes_received += sizeof(T) *
      (std::accumulate(rcv_counts, &rcv_counts[get_procs_per_trainer()], 0) -
       rcv_counts[get_rank_in_trainer()]);
  }
  /** Inter-trainer gather (for non-root processes). */
  template <typename T>
  void intertrainer_gather(T snd, int root) {
    gather(snd, root, intertrainer_comm);
  }
  /** Inter-trainer gather (for root processes). */
  template <typename T>
  void intertrainer_gather(T snd, std::vector<T>& rcv) {
    gather(snd, rcv, intertrainer_comm);
  }
  /** Inter-trainer scalar-array gather (for non-root processes). */
  template <typename T>
  void intertrainer_gather(T *snd, int count, int root) {
    gather(snd, count, root, intertrainer_comm);
  }
  /** Inter-trainer scalar-array gather (for root processes). */
  template <typename T>
  void intertrainer_gather(T *snd, int count, T *rcv) {
    gather(snd, count, rcv, intertrainer_comm);
  }
  /** Scalar gather (for non-root processes). */
  template <typename T>
  void gather(T snd, int root, const El::mpi::Comm& c) {
    bytes_sent += sizeof(T);
    El::mpi::Gather(&snd, 1, (T*) nullptr, 0, root, c,
                    El::SyncInfo<El::Device::CPU>{});
  }
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, T *rcv, const El::mpi::Comm& c) {
    auto const size_c = El::mpi::Size(c);
    auto const rank_c = El::mpi::Rank(c);
    El::mpi::Gather(&snd, 1, rcv, 1, rank_c, c,
                    El::SyncInfo<El::Device::CPU>{});
    bytes_received += sizeof(T) * (size_c - 1);
  }
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, std::vector<T>& rcv, const El::mpi::Comm& c) {
    gather(snd, rcv.data(), c);
  }
  /** Scalar-array gather (for non-root processes). */
  template <typename T>
  void gather(T *snd, int count, int root, const El::mpi::Comm& c)
  {
    gather(snd, count, root, c,
           El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void gather(T *snd, int count, int root, const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Gather(snd, count, (T*) nullptr, 0, root, c,
                    syncInfo);
  }
  /** Scalar-array gather (for root processes). */
  template <typename T>
  void gather(T *snd, int count, T *rcv, const El::mpi::Comm& c) {
      gather(snd, count, rcv, c, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void gather(T *snd, int count, T *rcv, const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo) {
    auto const size_c = El::mpi::Size(c);
    auto const rank_c = El::mpi::Rank(c);
    El::mpi::Gather(snd, count, rcv, count, rank_c, c, syncInfo);
    bytes_received += sizeof(T) * count * (size_c - 1);
  }
  /** Scalar scatter (for non-root processes). */
  template <typename T>
  T scatter(int root, const El::mpi::Comm& c) {
    T val = {};
    El::mpi::Scatter((T*) nullptr, 1, &val, 1, root, c,
                     El::SyncInfo<El::Device::CPU>{});
    bytes_received += sizeof(T);
    return val;
  }
  /** Scalar scatter (for root processes). */
  template <typename T>
  T scatter(T *snd, const El::mpi::Comm& c) {
    bytes_sent += sizeof(T) * (El::mpi::Size(c) - 1);
    T val = {};
    auto root = El::mpi::Rank(c);
    El::mpi::Scatter(snd, 1, &val, 1, root, c,
                     El::SyncInfo<El::Device::CPU>{});
    return val;
  }
  /** Inter-trainer reduce (for non-root processes). */
  template <typename T>
  void intertrainer_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, root, intertrainer_comm, op);
  }
  /** Inter-trainer reduce (for root processes). */
  template <typename T>
  T intertrainer_reduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return reduce(snd, intertrainer_comm, op);
  }
  /** Within-trainer reduce (for non-root processes). */
  template <typename T>
  void trainer_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, root, trainer_comm, op);
  }
  /** Within-trainer reduce (for root processes). */
  template <typename T>
  T trainer_reduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return reduce(snd, trainer_comm, op);
  }
  /** Within-trainer scalar array reduce (for non-root processes). */
  template <typename T>
  void trainer_reduce(T *snd, int count, int root, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, count, root, trainer_comm, op);
  }
  /** Within-trainer scalar array reduce (for root processes). */
  template <typename T>
  void trainer_reduce(T *snd, int count, T *rcv, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, count, rcv, trainer_comm, op);
  }
  /** Scalar reduce (for non-root processes). */
  template <typename T>
  void reduce(T snd, int root, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM) {
    bytes_sent += sizeof(T);
    El::mpi::Reduce(&snd, (T*) NULL, 1, op, root, c,
                    El::SyncInfo<El::Device::CPU>{});
  }
  /** Scalar reduce (for root processes). */
  template <typename T>
  T reduce(T snd, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM) {
    T val = {};
    auto const size_c = El::mpi::Size(c);
    auto const rank_c = El::mpi::Rank(c);
    El::mpi::Reduce(&snd, &val, 1, op, rank_c, c,
                    El::SyncInfo<El::Device::CPU>{});
    bytes_received += sizeof(T) * (size_c - 1);
    return val;
  }

  /** Scalar-array reduce (for non-root processes). */
  // Op is "SUM"
  template <typename T>
  void reduce(T *snd, int count, int root, const El::mpi::Comm& c) {
    reduce(snd, count, root, c, El::mpi::SUM,
           El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void reduce(T *snd, int count, int root, const El::mpi::Comm& c, El::SyncInfo<D> const& syncInfo) {
    reduce(snd, count, root, c, El::mpi::SUM, syncInfo);
  }

  template <typename T>
  void reduce(T *snd, int count, int root, const El::mpi::Comm& c, El::mpi::Op op) {
    reduce(snd, count, root, c, op, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void reduce(T *snd, int count, int root, const El::mpi::Comm& c, El::mpi::Op op, El::SyncInfo<D> const& syncInfo) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Reduce(snd, (T*) NULL, count, op, root, c, syncInfo);
  }
  /** Scalar-array reduce (for root processes). */
  template <typename T, El::Device D>
  void reduce(T *snd, int count, T *rcv, const El::mpi::Comm& c, El::SyncInfo<D> const& syncInfo) {
    reduce(snd, count, rcv, c, El::mpi::SUM, syncInfo);
  }
  template <typename T>
  void reduce(T *snd, int count, T *rcv, const El::mpi::Comm& c) {
    reduce(snd, count, rcv, c, El::mpi::SUM, El::SyncInfo<El::Device::CPU>{});
  }

  template <typename T>
  void reduce(T *snd, int count, T *rcv, const El::mpi::Comm& c, El::mpi::Op op) {
      reduce(snd, count, rcv, c, op, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void reduce(T *snd, int count, T *rcv, const El::mpi::Comm& c, El::mpi::Op op, El::SyncInfo<D> const& syncInfo) {
      if (snd == rcv) { snd = (T*)MPI_IN_PLACE; }
    auto const rank_c = El::mpi::Rank(c);
    auto const size_c = El::mpi::Size(c);
    El::mpi::Reduce(snd, rcv, count, op, rank_c, c, syncInfo);
    bytes_received += sizeof(T) * count * (size_c - 1);
  }
  /** Inter-trainer all-reduce. */
  template <typename T>
  T intertrainer_allreduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return allreduce(snd, intertrainer_comm, op);
  }
  /** Within-trainer all-reduce. */
  template <typename T>
  T trainer_allreduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return allreduce(snd, trainer_comm, op);
  }
  /** Scalar array within-trainer all-reduce. */
  template <typename T>
  void trainer_allreduce(T *snd, int count, T *rcv, El::mpi::Op op = El::mpi::SUM) {
    allreduce(snd, count, rcv, trainer_comm, op);
  }
  /** Scalar allreduce. */
  template <typename T>
  T allreduce(T snd, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM) {
    auto const size_c = El::mpi::Size(c);
    bytes_sent += sizeof(T);
    allreduce(&snd, 1, c, op);
    bytes_received += sizeof(T) * (size_c - 1);
    return snd;
  }

  // FIXME (trb): Based on the backend choice of "MPIBackend", I'm
  // assuming this is intended as a CPU-only call.
  /** Scalar-array allreduce. */
  template <typename T>
  void allreduce(T *snd, int count, T *rcv, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM) {
    auto const size_c = El::mpi::Size(c);
    bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
#ifdef LBANN_ALUMINUM_MPI_PASSTHROUGH
    ::Al::MPIAllreduceAlgorithm algo = ::Al::MPIAllreduceAlgorithm::mpi_passthrough;
#else
    ::Al::MPIAllreduceAlgorithm algo = ::Al::MPIAllreduceAlgorithm::automatic;
#endif
    ::Al::Allreduce<::Al::MPIBackend>(
        snd, rcv, count, mpi_op_to_al_op(op), c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}), algo);
#else
    El::mpi::AllReduce(snd, rcv, count, op, c,
                       El::SyncInfo<El::Device::CPU>{});
#endif
    bytes_received += count * sizeof(T) * (size_c - 1);
  }
  /** In-place scalar-array allreduce. */
  template <typename T>
  void allreduce(T *data, int count, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM) {
    auto const size_c = El::mpi::Size(c);
    bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
#ifdef LBANN_ALUMINUM_MPI_PASSTHROUGH
    ::Al::MPIAllreduceAlgorithm algo = ::Al::MPIAllreduceAlgorithm::mpi_passthrough;
#else
    ::Al::MPIAllreduceAlgorithm algo = ::Al::MPIAllreduceAlgorithm::automatic;
#endif
    ::Al::Allreduce<::Al::MPIBackend>(
      data, count, mpi_op_to_al_op(op), c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}), algo);
#else
    El::mpi::AllReduce(data, count, op, c,
                       El::SyncInfo<El::Device::CPU>{});
#endif
    bytes_received += count * sizeof(T) * (size_c - 1);
  }
  /** Matrix allreduce. */
  void allreduce(AbsMat& m,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM);
  /** Matrix allreduce. */
  void allreduce(AbsDistMat& m,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  void nb_allreduce(AbsMat& m,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  void nb_allreduce(AbsDistMat& m,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking in-place scalar-array allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a blocking
   *  allreduce.
   *  This currently only supports host pointers (i.e. the MPI backend).
   */
  template <typename T>
  void nb_allreduce(T *data, int count, const El::mpi::Comm& c, Al::request& req,
                    El::mpi::Op op = El::mpi::SUM) {
#ifdef LBANN_HAS_ALUMINUM
    bytes_sent += count * sizeof(T);
    req.mpi_req = Al::mpi_null_req;
    ::Al::NonblockingAllreduce<::Al::MPIBackend>(
      data, count, mpi_op_to_al_op(op), c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}), req.mpi_req);
    bytes_received += count * sizeof(T) * (El::mpi::Size(c) - 1);
#else
    allreduce(data, count, c, op);
#endif  // LBANN_HAS_ALUMINUM
  }

  /** Wait for a all non-blocking requests to complete. */
  template <typename T>
  void wait_all(std::vector<El::mpi::Request<T>>& req) {
    El::mpi::WaitAll(req.size(), req.data());
  }

  /** Wait for a non-blocking request to complete. */
  template <typename T>
  void wait(El::mpi::Request<T>& req) {
    El::mpi::Wait(req);
  }

  /** Wait for a non-blocking request to complete. */
  void wait(Al::request& req);
  /** Test whether a non-blocking request has completed; true if it has. */
  bool test(Al::request& req);

  /** Barrier among the inter-trainer processes. */
  void intertrainer_barrier();
  /** Barrier among processes in this trainer. */
  void trainer_barrier();
  /** Barrier among all processes. */
  void global_barrier();
  /** Barrier on an arbitrary communicator. */
  void barrier(const El::mpi::Comm& c);

  /** Send a buffer to rank in trainer. */
  template <typename T>
  void send(const T *data, int count, int trainer, int rank) {
    send(data, count, trainer, rank, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void send(const T *data, int count, int trainer, int rank, El::SyncInfo<D> const& syncInfo) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Send(data, count, get_world_rank(trainer, rank), get_world_comm(), syncInfo);
  }
  template <typename T, El::Device D>
  void send(const T *data, int count, int trainer, El::SyncInfo<D> const& syncInfo) {
    send(data, count, trainer, rank_in_trainer, syncInfo);
  }
  void send(const AbsMat& mat, int trainer, int rank);
  void send(const DistMat& mat, int trainer, int rank);
  void send(const AbsMat& mat, int trainer) {
    send(mat, trainer, rank_in_trainer);
  }
  void send(const DistMat& mat, int trainer) {
    send(mat, trainer, rank_in_trainer);
  }

  /** Corresponding non-blocking sends. */
  template <typename T>
  void nb_send(const T *data, int count, int trainer, int rank,
               El::mpi::Request<T>& req) {
    bytes_sent += sizeof(T) * count;
    El::mpi::ISend(data, count, get_world_rank(trainer, rank), get_world_comm(), req);
  }
  template <typename T>
  void nb_tagged_send(const T *data, int count, int rank, int tag,
               El::mpi::Request<T>& req, const El::mpi::Comm& c) {
    bytes_sent += sizeof(T) * count;
    El::mpi::TaggedISend(data, count, rank, tag, c, req);
  }
  template <typename T> void nb_send(const T *data, int count, int trainer,
                                     El::mpi::Request<T>& req) {
    nb_send(data, count, trainer, rank_in_trainer, req);
  }
  void nb_send(const AbsMat& mat, int trainer, int rank,
               El::mpi::Request<DataType>& req);
  void nb_send(const DistMat& mat, int trainer, int rank,
               El::mpi::Request<DataType>& req);
  void nb_send(const AbsMat& mat, int trainer, El::mpi::Request<DataType>& req) {
    nb_send(mat, trainer, rank_in_trainer, req);
  }
  void nb_send(const DistMat& mat, int trainer, El::mpi::Request<DataType>& req) {
    nb_send(mat, trainer, rank_in_trainer, req);
  }

  /** Corresponding receive to send. */
  template <typename T> void recv(T *data, int count, int trainer, int rank) {
    recv(data, count, trainer, rank, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T> void recv(T *data, int count, int trainer) {
    recv(data, count, trainer, rank_in_trainer);
  }
  template <typename T> void recv(T *data, int count) {
    recv(data, count, El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void recv(T *data, int count, int trainer, int rank, El::SyncInfo<D> const& syncInfo) {
    El::mpi::Recv(data, count, get_world_rank(trainer, rank), get_world_comm(), syncInfo);
    bytes_received += sizeof(T) * count;
  }
  template <typename T, El::Device D>
  void recv(T *data, int count, int trainer, El::SyncInfo<D> const& syncInfo) {
    recv(data, count, trainer, rank_in_trainer, syncInfo);
  }
  void recv(AbsMat& mat, int trainer, int rank);
  void recv(DistMat& mat, int trainer, int rank);
  void recv(AbsMat& mat, int trainer) {
    recv(mat, trainer, rank_in_trainer);
  }
  void recv(DistMat& mat, int trainer) {
    recv(mat, trainer, rank_in_trainer);
  }
  /** As above, but receive from anyone. */
  template <typename T, El::Device D>
  void recv(T *data, int count, El::SyncInfo<D> const& syncInfo) {
    El::mpi::Recv(data, count, El::mpi::ANY_SOURCE, get_world_comm(), syncInfo);
    bytes_received += sizeof(T) * count;
  }
  void recv(AbsMat& mat);
  void recv(DistMat& mat);

  /** Corresponding non-blocking receives. */
  template <typename T> void nb_recv(T *data, int count, int trainer, int rank,
                                     El::mpi::Request<T>& req) {
    El::mpi::IRecv(data, count, get_world_rank(trainer, rank), get_world_comm(),
               req);
    bytes_received += sizeof(T) * count;
  }
  template <typename T> void nb_tagged_recv(
               T *data, int count, int rank, int tag,
               El::mpi::Request<T>& req, const El::mpi::Comm& c) {
    El::mpi::TaggedIRecv(data, count, rank, tag, c, req);
    bytes_received += sizeof(T) * count;
  }

  template <typename T> void nb_recv(T *data, int count, int trainer,
                                     El::mpi::Request<T>& req) {
    nb_recv(data, count, trainer, rank_in_trainer, req);
  }
  void nb_recv(AbsMat& mat, int trainer, int rank, El::mpi::Request<DataType>& req);
  void nb_recv(DistMat& mat, int trainer, int rank, El::mpi::Request<DataType>& req);
  void nb_recv(AbsMat& mat, int trainer, El::mpi::Request<DataType>& req) {
    nb_recv(mat, trainer, rank_in_trainer, req);
  }
  void nb_recv(DistMat& mat, int trainer, El::mpi::Request<DataType>& req) {
    nb_recv(mat, trainer, rank_in_trainer, req);
  }
  template <typename T> void nb_recv(T *data, int count, El::mpi::Request<T>& req) {
    El::mpi::IRecv(data, count, El::mpi::ANY_SOURCE, get_world_comm(), req);
    bytes_received += sizeof(T) * count;
  }
  void nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req);
  void nb_recv(DistMat& mat, El::mpi::Request<DataType>& req);

  /** Send/recv to/from ranks. */
  template <typename T, El::Device D>
  void sendrecv(const T *snd, int send_count, int send_trainer, int send_rank,
                T *rcv, int recv_count, int recv_trainer, int recv_rank) {
    sendrecv(snd, send_count, send_trainer, send_rank,
             rcv, recv_count, recv_trainer, recv_rank,
             El::SyncInfo<El::Device::CPU>{});
  }
  template <typename T, El::Device D>
  void sendrecv(const T *snd, int send_count, int send_trainer,
                T *rcv, int recv_count, int recv_trainer) {
    sendrecv(snd, send_count, send_trainer, rank_in_trainer,
             rcv, recv_count, recv_trainer, rank_in_trainer,
             El::SyncInfo<El::Device::CPU>{});
  }

  template <typename T, El::Device D>
  void sendrecv(const T *snd, int send_count, int send_trainer, int send_rank,
                T *rcv, int recv_count, int recv_trainer, int recv_rank,
                El::SyncInfo<D> const& syncInfo) {
    bytes_sent += sizeof(T) * send_count;
    bytes_received += sizeof(T) * recv_count;
    El::mpi::SendRecv(snd, send_count, get_world_rank(send_trainer, send_rank),
                      rcv, recv_count, get_world_rank(recv_trainer, recv_rank),
                      get_world_comm(), syncInfo);
  }
  template <typename T, El::Device D>
  void sendrecv(const T *snd, int send_count, int send_trainer,
                T *rcv, int recv_count, int recv_trainer,
                El::SyncInfo<D> const& syncInfo) {
    sendrecv(snd, send_count, send_trainer, rank_in_trainer,
             rcv, recv_count, recv_trainer, rank_in_trainer, syncInfo);
  }

  /** Determine the size (count) of an incoming message. */
  template <typename T> int get_count(int trainer, int rank) {
    MPI_Status status;
    MPI_Probe(get_world_rank(trainer, rank), MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    return El::mpi::GetCount<T>(status);
  }
  template <typename T>
  int get_count(int trainer) {
    return get_count<T>(trainer, rank_in_trainer);
  }

  // Statistics methods.
  /** Return the number of trainer barriers performed. */
  inline size_t get_num_trainer_barriers() const {
    return num_trainer_barriers;
  }
  /** Return the number of inter-trainer barriers performed. */
  inline size_t get_num_intertrainer_barriers() const {
    return num_intertrainer_barriers;
  }
  /** Return the number of global barriers performed. */
  inline size_t get_num_global_barriers() const {
    return num_global_barriers;
  }
  /** Return the number of bytes sent. */
  inline size_t get_bytes_sent() const {
    return bytes_sent;
  }
  /** Return the number of bytes received. */
  inline size_t get_bytes_received() const {
    return bytes_received;
  }

  inline void reset_stats_counters() {
    num_trainer_barriers = 0;
    num_intertrainer_barriers = 0;
    num_global_barriers = 0;
    bytes_sent = 0;
    bytes_received = 0;
  }

  /** Return true if mat can be transmitted. */
  static inline bool is_sendable(const AbsMat& mat) {
    // This assumes we do not transmit mat with a datatype smaller than
    // DataType.
    // MPI uses "int" as its count type; do calculations with larger ints.
    size_t count = (size_t) mat.Height() * (size_t) mat.Width();
    return count <= (size_t) std::numeric_limits<int>::max();
  }
  /** Return true if the local portion of dist_mat can be transmitted. */
  static inline bool is_sendable(const AbsDistMat& dist_mat) {
    return is_sendable(dist_mat.LockedMatrix());
  }

  /** Return the intertrainer communicator. */
  const El::mpi::Comm& get_intertrainer_comm() const {
    return intertrainer_comm;
  }

  /** Return the trainer communicator. */
  const El::mpi::Comm& get_trainer_comm() const {
    return trainer_comm;
  }

  /** Return the world communicator. */
  const El::mpi::Comm& get_world_comm() const {
    return world_comm;
  }

  /** Return the communicator for this node. */
  const El::mpi::Comm& get_node_comm() const {
    return node_comm;
  }

  /**
   * Return a communicator containing num_per_group processors.
   *
   * This will attempt to pack processes so that the processes in each group
   * are physically close together on the system.
   *
   * num_per_group must evenly divide the number of processors in the world.
   */
  const El::mpi::Comm& get_packed_group_comm(int num_per_group) const;

  /** Return true if rank (in comm) is on the local node. */
  bool is_rank_node_local(int rank, const El::mpi::Comm& comm) const {
    // Translating to COMM_WORLD is typically constant time.
    int world_rank = El::mpi::Translate(comm, rank, get_world_comm());
    return is_world_rank_on_node(world_rank);
  }

  /** throws an lbann_exception **/
  void lbann_comm_abort(std::string msg);

 private:
  /** World communicator. */
  const El::mpi::Comm world_comm;
  /** Communicator for every process in this trainer. */
  El::mpi::Comm trainer_comm;
  /** Communicator for every process with the same trainer rank. */
  El::mpi::Comm intertrainer_comm;
  /** Communicator for every process in the same compute node. */
  El::mpi::Comm node_comm;
  /** Packed group communicators. */
  mutable std::unordered_map<int, El::mpi::Comm> group_communicators;
  /** Grid for this trainer. */
  Grid *grid;
  /** Number of trainers. */
  int num_trainers;
  /** Number of processors per trainer. */
  int procs_per_trainer;
  /** Rank of the trainer this process is in. */
  int trainer_rank;
  /** Rank of this process within its trainer. */
  int rank_in_trainer;
  /** Number of processers per compute node. */
  int procs_per_node;
  /** Rank of this process within its compute node. */
  int rank_in_node;
  /** The list of world ranks that are on this compute node. */
  std::vector<int> world_ranks_on_node;
  /** Default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided.
   */
  int threads_per_proc;

#ifdef LBANN_HAS_ALUMINUM
  /** Convert an MPI_Op to an Aluminum reduction operator. */
  ::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op);
#endif

  // Various statistics counters.
  size_t num_trainer_barriers;
  size_t num_intertrainer_barriers;
  size_t num_global_barriers;
  size_t bytes_sent;
  size_t bytes_received;

  /** Setup communicator for processes in the same compute node. */
  void setup_node_comm();

  /** Initialize the default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided. If the environment
   *  variable OMP_NUM_THREADS is defined, it's value is used for the
   *  default. Otherwise, then the default is the number of hardware
   *  cores per node divided by the number of processes per node.
   */
  void setup_threads();

};

template <typename T, bool S>
void lbann_comm::broadcast(int root, T& val, const El::mpi::Comm& c) {
  auto const rank_c = El::mpi::Rank(c);
  if (S) {
    // Avoid linking error from uninstantiated El::mpi routine if !S by converting T to El::byte
    using TT = typename interpret_as_byte_if_needed<S, T>::type;
    broadcast_native<TT>(root, reinterpret_cast<TT&>(val), c);
  } else {
    broadcast_custom(root, val, c);
  }
  count_bytes_broadcast(sizeof(T), rank_c, root);
}

template <typename T>
void lbann_comm::broadcast_native(int root, T& val, const El::mpi::Comm& c) const {
  El::mpi::Broadcast(val, root, c, El::SyncInfo<El::Device::CPU>{});
}

template <typename T>
void lbann_comm::broadcast_custom(int root, T& val, const El::mpi::Comm& c) const {
 const int bytes =  static_cast<int>(sizeof(T));
 El::mpi::Broadcast<El::byte>(reinterpret_cast<El::byte*>(&val), bytes, root, c,
                              El::SyncInfo<El::Device::CPU>{});
}

template <typename T, El::Device D, bool S>
void lbann_comm::broadcast(const int root, T* data, const int count, const El::mpi::Comm& c, El::SyncInfo<D> const& syncInfo) {
  auto const rank_c = El::mpi::Rank(c);
  const int size = static_cast<int>(S? count : sizeof(T)*count);
  // Avoid linking error from uninstantiated El::mpi routine if !S by converting T to El::byte
  using TT = typename interpret_as_byte_if_needed<S, T>::type;
  El::mpi::Broadcast<TT>(reinterpret_cast<TT*>(data), size, root, c, syncInfo);
  count_bytes_broadcast(sizeof(T)*count, rank_c, root);
}

/// Broadcast std::string over an arbitrary communicator.
template<>
void lbann_comm::broadcast<std::string>(const int root, std::string& str, const El::mpi::Comm& c);

/** Get the current rank within MPI_COMM_WORLD.
 *  This function is safe to call even if MPI has not initialized or
 *  has been finalized. In either case it returns a negative value.
 */
int get_rank_in_world();

} // namespace lbann

#endif  // LBANN_COMM_HPP_INCLUDED
