////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
using nccl_req_type = cudaEvent_t;
static const nccl_req_type nccl_null_req = (nccl_req_type) (-1);
#else
using nccl_backend = lbann::Al::dummy_backend;
using nccl_req_type = nccl_backend::req_type;
static const nccl_req_type nccl_null_req = nccl_backend::null_req;
#endif // defined(LBANN_HAS_ALUMINUM) && defined(AL_HAS_NCCL)

/** Wrapper for Aluminum non-blocking routine requests. */
struct request {
  mpi_req_type mpi_req = mpi_null_req;
  /// @todo MPI-CUDA backend
  nccl_req_type nccl_req = nccl_null_req;
};

} // namespace Al

/**
 * Manage communication.
 * This supports separate models, each of which are split over potentially
 * several processes. Every model is split over the same number of processes.
 * The corresponding processes between models are on the "inter-model
 * communicator".
 * You can also do point-to-point or broadcast communication to arbitrary sets
 * of processes.
 */
class lbann_comm {
 public:
  /**
   * Init communicators for models each with procs_per_model processes,
   * defaulting to every process in one model.
   */
  lbann_comm(int procs_per_model = 0,
             const El::mpi::Comm world = El::mpi::COMM_WORLD);
  /** Don't allow copying; it doesn't make sense for the communicator. */
  lbann_comm(const lbann_comm&) = delete;
  /** Don't allow assignment; it doesn't make sense for the communicator. */
  lbann_comm& operator=(const lbann_comm&) = delete;
  ~lbann_comm();

  /**
   * Split communicators so each model has procs_per_model processes.
   * If you call this multiple times, it will invalidate existing grids
   * and communicators.
   */
  void split_models(int procs_per_model);

  /** Get which model this process is in. */
  inline int get_model_rank() const {
    return model_rank;
  }
  /** Get the rank of this process in its model. */
  inline int get_rank_in_model() const {
    return rank_in_model;
  }
  /** Get my rank in COMM_WORLD. */
  inline int get_rank_in_world() const {
    return El::mpi::Rank(get_world_comm());
  }
  /** Return the COMM_WORLD rank of the rank'th processor in model. */
  inline int get_world_rank(int model, int rank) const {
    return procs_per_model * model + rank;
  }
  /** Return the rank of the master process in this model. */
  inline int get_model_master() const {
    return 0;
  }
  /** Return the rank of the inter-model master process. */
  inline int get_intermodel_master() const {
    return 0;
  }
  /** Return the rank of the world master process. */
  inline int get_world_master() const {
    return 0;
  }
  /** Return true if this process is the master process in its model. */
  inline bool am_model_master() const {
    return get_rank_in_model() == get_model_master();
  }
  /** Return true if this process is the world master process. */
  inline bool am_world_master() const {
    return get_rank_in_world() == get_world_master();
  }
  /** Return a grid to use for this model. */
  inline Grid& get_model_grid() {
    return *grid;
  }
  /** Return the total number of models. */
  inline int get_num_models() const {
    return num_models;
  }
  /* Return the number of processes in a model. */
  inline int get_procs_per_model() const {
    return procs_per_model;
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

  /** Perform a sum reduction of mat over the inter-model communicator. */
  void intermodel_sum_matrix(AbsMat& mat);
  void intermodel_sum_matrix(AbsDistMat& mat);
  /** Broadcast mat over the inter-model communicator starting from root. */
  void intermodel_broadcast_matrix(AbsMat& mat, int root);
  void intermodel_broadcast_matrix(AbsDistMat& mat, int root);

  /// Broadcast a scalar value over an arbitrary communicator
  template < typename T, bool S = is_instantiated_El_mpi_type<T>::value >
  void broadcast(int root, T& val, const El::mpi::Comm c);

  template <typename T>
  void broadcast_custom(int root, T& val, const El::mpi::Comm c) const;
  template <typename T>
  void broadcast_native(int root, T& val, const El::mpi::Comm c) const;

  /// World broadcast of a scalar.
  template <typename T>
  void world_broadcast(int root, T& val) {
    broadcast(root, val, get_world_comm());
  }
  /// Inter-model broadcast of a scalar.
  template <typename T>
  void intermodel_broadcast(int root, T& val) {
    broadcast(root, val, get_intermodel_comm());
  }
  /// Within-model broadcast of a scalar.
  template <typename T>
  void model_broadcast(int root, T& val) {
    broadcast(root, val, get_model_comm());
  }

  /**
   * Broadcast a buffer over an arbitrary communicator assuming that
   * the buffer space is already allocated.
   */
  template < typename T, bool S = is_instantiated_El_mpi_type<T>::value >
  void broadcast(const int root, T* data, const int count, const El::mpi::Comm c);

  /// World broadcast of a buffer.
  template <typename T>
  void world_broadcast(const int root, T* data, const int count) {
    broadcast(root, data, count, get_world_comm());
  }
  /// Inter-model broadcast of a buffer.
  template <typename T>
  void intermodel_broadcast(const int root, T* data, const int count) {
    broadcast(root, data, count, get_intermodel_comm());
  }
  /// Within-model broadcast of a buffer.
  template <typename T>
  void model_broadcast(const int root, T* data, const int count) {
    broadcast(root, data, count, get_model_comm());
  }

  /**
   * Resize vector<> over an arbitrary communicator to match the one on root.
   */
  template <typename T>
  size_t resize(const int root, std::vector<T> &data, const El::mpi::Comm c) {
    size_t count = data.size();
    El::mpi::Broadcast(&count, 1, root, c);
    count_bytes_broadcast(sizeof(size_t), El::mpi::Rank(c), root);
    data.resize(count);
    return count;
  }

  /**
   * Broadcast vector<> over an arbitrary communicator;
   * vector<> for non-root processes will be resized as needed.
   */
  template <typename T>
  void broadcast(const int root, std::vector<T> &data, const El::mpi::Comm c) {
    const int count = static_cast<int>(resize(root, data, c));
    if (count <= 0) {
      return;
    }
    broadcast<T>(root, data.data(), count, c);
  }
  /// Broadcast vector<> to world.
  template <typename T>
  void world_broadcast(int root, std::vector<T> &data) {
    broadcast(root, data, get_world_comm());
  }
  /**
   * Broadcast vector<> within model;
   * vector<> for non-root processes will be resized as needed.
   */
  /// Broadcast vector<> across models.
  template <typename T>
  void intermodel_broadcast(int root, std::vector<T> &data) {
    broadcast(root, data, get_intermodel_comm());
  }
  /// Broadcast vector<> within model.
  template <typename T>
  void model_broadcast(int root, std::vector<T> &data) {
    broadcast(root, data, get_model_comm());
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
  void all_gather(const T* src, int src_count, T* rcv, int rcv_count, El::mpi::Comm c) {
    El::mpi::AllGather<T>(src, src_count, rcv, rcv_count, c);
  }

  /** 
   * Allgatherv over an arbitrary communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(std::vector<T> &src, std::vector<T> &rcs, std::vector<int> &rcv_counts, std::vector<int> &rcv_disp, El::mpi::Comm c) {
    if (src.size() == 0) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
              << "all_gather for vector<>: vector.size() == 0;\n"
              << "this doesn't work!";
      lbann_comm_abort(err.str());
    }
    El::mpi::AllGather<T>(src.data(), src.size(), rcs.data(), rcv_counts.data(), rcv_disp.data(), c);
  }
  /** 
   * Allgatherv over a model communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void model_all_gather(std::vector<T> &src, std::vector<T> &rcs, std::vector<int> &rcv_counts, std::vector<int> &rcv_disp, const El::mpi::Comm c) {
    all_gather(src, rcs, rcv_counts, rcv_disp, get_model_comm());
  }
  /** 
   * Allgather for a single element over an arbitrary communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(T &src, std::vector<T> &data, const El::mpi::Comm c) {
    El::mpi::AllGather(&src, 1, data.data(), 1, c);
  }
  /** 
   * Allgather for a single element over the model communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void model_all_gather(T &src, std::vector<T> &data) {
    all_gather(src, data, get_model_comm());
  }

  /** Within-model scalar gather (for non-root processes). */
  template <typename T>
  void model_gather(T snd, int root) {
    gather(snd, root, model_comm);
  }
  /** Within-model scalar gather (for root processes). */
  template <typename T>
  void model_gather(T snd, T* rcv) {
    gather(snd, get_model_master(), model_comm);
  }
  /** Within-model scalar-array gather (for non-root processes). */
  template <typename T>
  void model_gather(T* snd, int count, int root) {
    gather(snd, count, root, model_comm);
  }
  /** Within-model scalar-array gather (for root processes). */
  template <typename T>
  void model_gather(T* snd, int count, T* rcv) {
    gather(snd, count, rcv, model_comm);
  }
  /** Within-model variable-length-array gather (for non-root processes). */
  template <typename T>
  void model_gatherv(T* snd, int count, int root) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Gather(snd, count, (T *) NULL, (int *) nullptr, (int *) nullptr, root,
                    model_comm);
  }
  template <typename T>
  void model_gatherv(T* snd, int count, T* rcv, int* rcv_counts,
                     int* rcv_displacements) {
    El::mpi::Gather(snd, count, rcv, rcv_counts, rcv_displacements,
                    get_rank_in_model(), model_comm);
    bytes_received += sizeof(T) *
      (std::accumulate(rcv_counts, &rcv_counts[get_procs_per_model()], 0) -
       rcv_counts[get_rank_in_model()]);
  }
  /** Inter-model gather (for non-root processes). */
  template <typename T>
  void intermodel_gather(T snd, int root) {
    gather(snd, root, intermodel_comm);
  }
  /** Inter-model gather (for root processes). */
  template <typename T>
  void intermodel_gather(T snd, std::vector<T>& rcv) {
    gather(snd, rcv, intermodel_comm);
  }
  /** Inter-model scalar-array gather (for non-root processes). */
  template <typename T>
  void intermodel_gather(T *snd, int count, int root) {
    gather(snd, count, root, intermodel_comm);
  }
  /** Inter-model scalar-array gather (for root processes). */
  template <typename T>
  void intermodel_gather(T *snd, int count, T *rcv) {
    gather(snd, count, rcv, intermodel_comm);
  }
  /** Scalar gather (for non-root processes). */
  template <typename T>
  void gather(T snd, int root, const El::mpi::Comm c) {
    bytes_sent += sizeof(T);
    El::mpi::Gather(&snd, 1, (T*) nullptr, 0, root, c);
  }
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, T *rcv, const El::mpi::Comm c) {
    El::mpi::Gather(&snd, 1, rcv, 1, El::mpi::Rank(c), c);
    bytes_received += sizeof(T) * (El::mpi::Size(c) - 1);
  }
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, std::vector<T>& rcv, const El::mpi::Comm c) {
    gather(snd, rcv.data(), c);
  }
  /** Scalar-array gather (for non-root processes). */
  template <typename T>
  void gather(T *snd, int count, int root, const El::mpi::Comm c) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Gather(snd, count, (T*) nullptr, 0, root, c);
  }
  /** Scalar-array gather (for root processes). */
  template <typename T>
  void gather(T *snd, int count, T *rcv, const El::mpi::Comm c) {
    El::mpi::Gather(snd, count, rcv, count, El::mpi::Rank(c), c);
    bytes_received += sizeof(T) * count * (El::mpi::Size(c) - 1);
  }
  /** Scalar scatter (for non-root processes). */
  template <typename T>
  T scatter(int root, const El::mpi::Comm c) {
    T val = {};
    El::mpi::Scatter((T*) nullptr, 0, &val, 1, root, c);
    bytes_received += sizeof(T);
    return val;
  }
  /** Scalar scatter (for root processes). */
  template <typename T>
  T scatter(T *snd, const El::mpi::Comm c) {
    bytes_sent += sizeof(T) * (El::mpi::Size(c) - 1);
    T val = {};
    El::mpi::Scatter(snd, 1, &val, 1, El::mpi::Rank(c), c);
    return val;
  }
  /** Inter-model reduce (for non-root processes). */
  template <typename T>
  void intermodel_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, root, intermodel_comm, op);
  }
  /** Inter-model reduce (for root processes). */
  template <typename T>
  T intermodel_reduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return reduce(snd, intermodel_comm, op);
  }
  /** Within-model reduce (for non-root processes). */
  template <typename T>
  void model_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, root, model_comm, op);
  }
  /** Within-model reduce (for root processes). */
  template <typename T>
  T model_reduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return reduce(snd, model_comm, op);
  }
  /** Within-model scalar array reduce (for non-root processes). */
  template <typename T>
  void model_reduce(T *snd, int count, int root, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, count, root, model_comm, op);
  }
  /** Within-model scalar array reduce (for root processes). */
  template <typename T>
  void model_reduce(T *snd, int count, T *rcv, El::mpi::Op op = El::mpi::SUM) {
    reduce(snd, count, rcv, model_comm, op);
  }
  /** Scalar reduce (for non-root processes). */
  template <typename T>
  void reduce(T snd, int root, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    bytes_sent += sizeof(T);
    El::mpi::Reduce(&snd, (T*) NULL, 1, op, root, c);
  }
  /** Scalar reduce (for root processes). */
  template <typename T>
  T reduce(T snd, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    T val = {};
    El::mpi::Reduce(&snd, &val, 1, op, El::mpi::Rank(c), c);
    bytes_received += sizeof(T) * (El::mpi::Size(c) - 1);
    return val;
  }
  /** Scalar-array reduce (for non-root processes). */
  template <typename T>
  void reduce(T *snd, int count, int root, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Reduce(snd, (T*) NULL, count, op, root, c);
  }
  /** Scalar-array reduce (for root processes). */
  template <typename T>
  void reduce(T *snd, int count, T *rcv, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    if (snd == rcv) { snd = MPI_IN_PLACE; }
    El::mpi::Reduce(snd, rcv, count, op, El::mpi::Rank(c), c);
    bytes_received += sizeof(T) * count * (El::mpi::Size(c) - 1);
  }
  /** Inter-model all-reduce. */
  template <typename T>
  T intermodel_allreduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return allreduce(snd, intermodel_comm, op);
  }
  /** Within-model all-reduce. */
  template <typename T>
  T model_allreduce(T snd, El::mpi::Op op = El::mpi::SUM) {
    return allreduce(snd, model_comm, op);
  }
  /** Scalar array within-model all-reduce. */
  template <typename T>
  void model_allreduce(T *snd, int count, T *rcv, El::mpi::Op op = El::mpi::SUM) {
    allreduce(snd, count, rcv, model_comm, op);
  }
  /** Scalar allreduce. */
  template <typename T>
  T allreduce(T snd, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    bytes_sent += sizeof(T);
    allreduce(&snd, 1, c, op);
    bytes_received += sizeof(T) * (El::mpi::Size(c) - 1);
    return snd;
  }
  /** Scalar-array allreduce. */
  template <typename T>
  void allreduce(T *snd, int count, T *rcv, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
#ifdef LBANN_ALUMINUM_MPI_PASSTHROUGH
    ::Al::AllreduceAlgorithm algo = ::Al::AllreduceAlgorithm::mpi_passthrough;
#else
    ::Al::AllreduceAlgorithm algo = ::Al::AllreduceAlgorithm::automatic;
#endif
    ::Al::Allreduce<::Al::MPIBackend>(
      snd, rcv, count, mpi_op_to_al_op(op), *get_al_comm(c), algo);
#else
    El::mpi::AllReduce(snd, rcv, count, op, c);
#endif
    bytes_received += count * sizeof(T) * (El::mpi::Size(c) - 1);
  }
  /** In-place scalar-array allreduce. */
  template <typename T>
  void allreduce(T *data, int count, const El::mpi::Comm c, El::mpi::Op op = El::mpi::SUM) {
    bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
#ifdef LBANN_ALUMINUM_MPI_PASSTHROUGH
    ::Al::AllreduceAlgorithm algo = ::Al::AllreduceAlgorithm::mpi_passthrough;
#else
    ::Al::AllreduceAlgorithm algo = ::Al::AllreduceAlgorithm::automatic;
#endif
    ::Al::Allreduce<::Al::MPIBackend>(
      data, count, mpi_op_to_al_op(op), *get_al_comm(c), algo);
#else
    El::mpi::AllReduce(data, count, op, c);
#endif
    bytes_received += count * sizeof(T) * (El::mpi::Size(c) - 1);
  }
  /** Matrix allreduce. */
  void allreduce(AbsDistMat& m,
                 const El::mpi::Comm c,
                 El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  void nb_allreduce(AbsMat& m,
                    const El::mpi::Comm c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  void nb_allreduce(AbsDistMat& m,
                    const El::mpi::Comm c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM);
  /** Non-blocking in-place scalar-array allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a blocking
   *  allreduce.
   *  This currently only supports host pointers (i.e. the MPI backend).
   */
  template <typename T>
  void nb_allreduce(T *data, int count, const El::mpi::Comm c, Al::request& req,
                    El::mpi::Op op = El::mpi::SUM) {
#ifdef LBANN_HAS_ALUMINUM
    bytes_sent += count * sizeof(T);
    req.mpi_req = Al::mpi_null_req;
    ::Al::NonblockingAllreduce<::Al::MPIBackend>(
      data, count, mpi_op_to_al_op(op), *get_al_comm(c), req.mpi_req);
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

  /** Barrier among the inter-model processes. */
  void intermodel_barrier();
  /** Barrier among processes in this model. */
  void model_barrier();
  /** Barrier among all processes. */
  void global_barrier();
  /** Barrier on an arbitrary communicator. */
  void barrier(const El::mpi::Comm c);

  /** Send a buffer to rank in model. */
  template <typename T>
  void send(const T *data, int count, int model, int rank) {
    bytes_sent += sizeof(T) * count;
    El::mpi::Send(data, count, get_world_rank(model, rank), get_world_comm());
  }
  template <typename T> void send(const T *data, int count, int model) {
    send(data, count, model, rank_in_model);
  }
  void send(const AbsMat& mat, int model, int rank);
  void send(const DistMat& mat, int model, int rank);
  void send(const AbsMat& mat, int model) {
    send(mat, model, rank_in_model);
  }
  void send(const DistMat& mat, int model) {
    send(mat, model, rank_in_model);
  }

  /** Corresponding non-blocking sends. */
  template <typename T>
  void nb_send(const T *data, int count, int model, int rank,
               El::mpi::Request<T>& req) {
    bytes_sent += sizeof(T) * count;
    El::mpi::ISend(data, count, get_world_rank(model, rank), get_world_comm(), req);
  }
  template <typename T>
  void nb_tagged_send(const T *data, int count, int rank, int tag,
               El::mpi::Request<T>& req, const El::mpi::Comm c) {
    bytes_sent += sizeof(T) * count;
    El::mpi::TaggedISend(data, count, rank, tag, c, req);
  }
  template <typename T> void nb_send(const T *data, int count, int model,
                                     El::mpi::Request<T>& req) {
    nb_send(data, count, model, rank_in_model, req);
  }
  void nb_send(const AbsMat& mat, int model, int rank,
               El::mpi::Request<DataType>& req);
  void nb_send(const DistMat& mat, int model, int rank,
               El::mpi::Request<DataType>& req);
  void nb_send(const AbsMat& mat, int model, El::mpi::Request<DataType>& req) {
    nb_send(mat, model, rank_in_model, req);
  }
  void nb_send(const DistMat& mat, int model, El::mpi::Request<DataType>& req) {
    nb_send(mat, model, rank_in_model, req);
  }

  /** Corresponding receive to send. */
  template <typename T> void recv(T *data, int count, int model, int rank) {
    El::mpi::Recv(data, count, get_world_rank(model, rank), get_world_comm());
    bytes_received += sizeof(T) * count;
  }
  template <typename T> void recv(T *data, int count, int model) {
    recv(data, count, model, rank_in_model);
  }
  void recv(AbsMat& mat, int model, int rank);
  void recv(DistMat& mat, int model, int rank);
  void recv(AbsMat& mat, int model) {
    recv(mat, model, rank_in_model);
  }
  void recv(DistMat& mat, int model) {
    recv(mat, model, rank_in_model);
  }
  /** As above, but receive from anyone. */
  template <typename T> void recv(T *data, int count) {
    El::mpi::Recv(data, count, El::mpi::ANY_SOURCE, get_world_comm());
    bytes_received += sizeof(T) * count;
  }
  void recv(AbsMat& mat);
  void recv(DistMat& mat);

  /** Corresponding non-blocking receives. */
  template <typename T> void nb_recv(T *data, int count, int model, int rank,
                                     El::mpi::Request<T>& req) {
    El::mpi::IRecv(data, count, get_world_rank(model, rank), get_world_comm(),
               req);
    bytes_received += sizeof(T) * count;
  }
  template <typename T> void nb_tagged_recv(
               T *data, int count, int rank, int tag,
               El::mpi::Request<T>& req, const El::mpi::Comm c) {
    El::mpi::TaggedIRecv(data, count, rank, tag, c, req);
    bytes_received += sizeof(T) * count;
  }

  template <typename T> void nb_recv(T *data, int count, int model,
                                     El::mpi::Request<T>& req) {
    nb_recv(data, count, model, rank_in_model, req);
  }
  void nb_recv(AbsMat& mat, int model, int rank, El::mpi::Request<DataType>& req);
  void nb_recv(DistMat& mat, int model, int rank, El::mpi::Request<DataType>& req);
  void nb_recv(AbsMat& mat, int model, El::mpi::Request<DataType>& req) {
    nb_recv(mat, model, rank_in_model, req);
  }
  void nb_recv(DistMat& mat, int model, El::mpi::Request<DataType>& req) {
    nb_recv(mat, model, rank_in_model, req);
  }
  template <typename T> void nb_recv(T *data, int count, El::mpi::Request<T>& req) {
    El::mpi::IRecv(data, count, El::mpi::ANY_SOURCE, get_world_comm(), req);
    bytes_received += sizeof(T) * count;
  }
  void nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req);
  void nb_recv(DistMat& mat, El::mpi::Request<DataType>& req);

  /** Send/recv to/from ranks. */
  template <typename T>
  void sendrecv(const T *snd, int send_count, int send_model, int send_rank,
                T *rcv, int recv_count, int recv_model, int recv_rank) {
    bytes_sent += sizeof(T) * send_count;
    bytes_received += sizeof(T) * recv_count;
    El::mpi::SendRecv(snd, send_count, get_world_rank(send_model, send_rank),
                      rcv, recv_count, get_world_rank(recv_model, recv_rank),
                      get_world_comm());
  }
  template <typename T>
  void sendrecv(const T *snd, int send_count, int send_model,
                T *rcv, int recv_count, int recv_model) {
    sendrecv(snd, send_count, send_model, rank_in_model,
             rcv, recv_count, recv_model, rank_in_model);
  }

  /** Determine the size (count) of an incoming message. */
  template <typename T> int get_count(int model, int rank) {
    MPI_Status status;
    MPI_Probe(get_world_rank(model, rank), MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    return El::mpi::GetCount<T>(status);
  }
  template <typename T>
  int get_count(int model) {
    return get_count<T>(model, rank_in_model);
  }

  // Statistics methods.
  /** Return the number of model barriers performed. */
  inline size_t get_num_model_barriers() const {
    return num_model_barriers;
  }
  /** Return the number of inter-model barriers performed. */
  inline size_t get_num_intermodel_barriers() const {
    return num_intermodel_barriers;
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
  /** Return the number of bytes sent in allreduces. */
  inline size_t get_ar_bytes_sent() const {
    return ar_bytes_sent;
  }
  /** Return the number of bytes received in allreduces. */
  inline size_t get_ar_bytes_received() const {
    return ar_bytes_sent;
  }
  /** Return the number of bytes sent in allreduce reduce-scatters. */
  inline size_t get_ar_rs_bytes_sent() const {
    return ar_rs_bytes_sent;
  }
  /** Return the number of bytes received in allreduce reduce-scatters. */
  inline size_t get_ar_rs_bytes_received() const {
    return ar_rs_bytes_received;
  }
  /** Return the number of bytes sent in allreduce allgathers. */
  inline size_t get_ar_ag_bytes_sent() const {
    return ar_ag_bytes_sent;
  }
  /** Return the number of bytes received in allreduce allgathers. */
  inline size_t get_ar_ag_bytes_received() const {
    return ar_ag_bytes_received;
  }
  /** Return the time spent in allreduces. */
  inline double get_ar_time() const {
    return ar_time;
  }
  /** Return the time spent in allreduce reduce-scatters. */
  inline double get_ar_rs_time() const {
    return ar_rs_time;
  }
  /** Return the time spent in allreduce allgathers. */
  inline double get_ar_ag_time() const {
    return ar_ag_time;
  }
  /** Return the time spent in allreduce send transforms. */
  inline double get_ar_send_transform_time() const {
    return ar_send_transform_time;
  }
  /** Return the time spent in allreduce receive transforms. */
  inline double get_ar_recv_transform_time() const {
    return ar_recv_transform_time;
  }
  /** Return the time spent in allreduce receive/apply transforms. */
  inline double get_ar_recv_apply_transform_time() const {
    return ar_recv_apply_transform_time;
  }
  /** Return the time spent sending in allreduces. */
  inline double get_ar_send_time() const {
    return ar_send_time;
  }
  /** Return the time spent receiving in allreduces. */
  inline double get_ar_recv_time() const {
    return ar_recv_time;
  }
  /** Return the time spent sending in allreduce reduce-scatters. */
  inline double get_ar_rs_send_time() const {
    return ar_rs_send_time;
  }
  /** Return the time spent receiving in allreduce reduce-scatters. */
  inline double get_ar_rs_recv_time() const {
    return ar_rs_recv_time;
  }
  /** Return the time spent sending in allreduce allgathers. */
  inline double get_ar_ag_send_time() const {
    return ar_ag_send_time;
  }
  /** Return the time spent receiving in allreduce allgathers. */
  inline double get_ar_ag_recv_time() const {
    return ar_ag_recv_time;
  }
  inline void reset_stats_counters() {
    num_model_barriers = 0;
    num_intermodel_barriers = 0;
    num_global_barriers = 0;
    bytes_sent = 0;
    bytes_received = 0;
    ar_bytes_sent = 0;
    ar_bytes_received = 0;
    ar_rs_bytes_sent = 0;
    ar_rs_bytes_received = 0;
    ar_ag_bytes_sent = 0;
    ar_ag_bytes_received = 0;
    ar_time = 0.0;
    ar_rs_time = 0.0;
    ar_ag_time = 0.0;
    ar_send_transform_time = 0.0;
    ar_recv_transform_time = 0.0;
    ar_recv_apply_transform_time = 0.0;
    ar_send_time = 0.0;
    ar_recv_time = 0.0;
    ar_rs_send_time = 0.0;
    ar_rs_recv_time = 0.0;
    ar_ag_send_time = 0.0;
    ar_ag_recv_time = 0.0;
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

  // Custom allreduce implementations.
  /** Specify different allreduce algorithms. */
  enum class allreduce_algorithm {
    DEFAULT,
    DYNAMIC,  /** Choose algorithm based on data size. */
    RECURSIVE_DOUBLING,
    PAIRWISE_EXCHANGE_RING,
    RING,
    RABENSEIFNER,
    INVALID
  };

  /** Allreduce options. */
  struct allreduce_options {
    /** Allreduce algorithm to use. */
    allreduce_algorithm algo = allreduce_algorithm::DEFAULT;
    /** Optimization: the recv_transform is the identity. */
    bool id_recv = false;
    /**
     * Optimization: When communication is node-local, do not apply the
     * send_transform. Implies id_recv when possible and sets the local flag
     * in recv_apply_transform to true when taken advantage of.
     */
    bool no_local_trans = false;
    /** Max number of concurrent reduce steps, must be >= 1. */
    int max_reduces = 1;
  };

  /** Get the default allreduce algorithm to use (may be DYNAMIC). */
  allreduce_algorithm get_default_allreduce_algorithm() const {
    return default_allreduce_algo;
  }
  /**
   * Set the default allreduce algorithm to algo.
   * Do *not* set it to DEFAULT.
   */
  void set_default_allreduce_algorithm(allreduce_algorithm algo) {
    default_allreduce_algo = algo;
  }

  /**
   * Do a custom allreduce on mat on the intermodel communicator.
   * This selects the allreduce algorithm to use based on the size of mat.
   * All counts/sizes are in bytes.
   * @param mat The matrix to allreduce.
   * @param max_recv_count An upper bound on the size of data that will be
   * received in any step; this will be the size of receive buffers used in
   * the allreduce.
   * @param send_transform A function that takes a range of a matrix and
   * applies a transformation to it. The return value is a pointer to a buffer
   * containing the transformed data, which will be sent. The int& param
   * should be set to the count of how many elements are in the buffer. The
   * boolean parameter indicates whether the matrix is constant between
   * different calls to send_transform; if true, the function may be able to
   * take advantage of this. The int parameter gives a count of how many times
   * send_transform has been called concurrently, starting from 0.
   * @param recv_transform A function that takes a pointer to a buffer and a
   * matrix and applies a transform to the buffer, storing the result in the
   * matrix. The buffer will be data transformed with send_transform and
   * received from another rank. The return value is the actual count of the
   * received data (i.e. the count that the data was sent using).
   * @param recv_apply_transform A function like recv_transform except that
   * the transformed data should be combined (applied, reduced) with the
   * current data in the matrix argument. A boolean parameter indicates that
   * no_local_trans was true and the data was not transformed.
   * @param options Various allreduce options.
   */
  void intermodel_allreduce(
    AbsMat& mat, int max_recv_count,
    std::function<uint8_t *(AbsMat&, El::IR, El::IR, int&, bool, int)> send_transform,
    std::function<int(uint8_t *, AbsMat&)> recv_transform,
    std::function<int(uint8_t *, AbsMat&, bool)> recv_apply_transform,
    const allreduce_options opts);

  /**
   * A recursive-doubling allreduce.
   * This implementation only works for a power-of-2 number of processes.
   */
  void recursive_doubling_allreduce_pow2(
    const El::mpi::Comm comm, AbsMat& mat, int max_recv_count,
    std::function<uint8_t *(AbsMat&, El::IR, El::IR, int&, bool, int)> send_transform,
    std::function<int(uint8_t *, AbsMat&, bool)> recv_apply_transform,
    const allreduce_options opts);

  /**
   * An allreduce based on a pairwise-exchange reduce-scatter followed by a
   * ring-based allgather.
   * @param num_reduces If >1, performs up to num_reduces reduces concurrently
   * in the reduce-scatter phase.
   */
  template <El::Device D>
  void pe_ring_allreduce(
                         const El::mpi::Comm comm, DMat<D>& mat, int max_recv_count,
    std::function<uint8_t *(AbsMat&, El::IR, El::IR, int&, bool, int)> send_transform,
    std::function<int(uint8_t *, AbsMat&)> recv_transform,
    std::function<int(uint8_t *, AbsMat&, bool)> recv_apply_transform,
    const allreduce_options opts);

  /**
   * An allreduce using ring-based reduce-scatter and allgather.
   */
  template <El::Device D>
  void ring_allreduce(
    const El::mpi::Comm comm, DMat<D>& mat, int max_recv_count,
    std::function<uint8_t *(AbsMat&, El::IR, El::IR, int&, bool, int)> send_transform,
    std::function<int(uint8_t *, AbsMat&)> recv_transform,
    std::function<int(uint8_t *, AbsMat&, bool)> recv_apply_transform,
    const allreduce_options opts);

  /**
   * An allreduce using a recursive-halving reduce-scatter followed by a
   * recursive-doubling allgather.
   */
  template <El::Device D>
  void rabenseifner_allreduce(
    const El::mpi::Comm comm, DMat<D>& mat, int max_recv_count,
    std::function<uint8_t *(AbsMat&, El::IR, El::IR, int&, bool, int)> send_transform,
    std::function<int(uint8_t *, AbsMat&)> recv_transform,
    std::function<int(uint8_t *, AbsMat&, bool)> recv_apply_transform,
    const allreduce_options opts);

  /** Return the intermodel communicator. */
  El::mpi::Comm get_intermodel_comm() const {
    return intermodel_comm;
  }

  /** Return the model communicator. */
  El::mpi::Comm get_model_comm() const {
    return model_comm;
  }

  /** Return the world communicator. */
  const El::mpi::Comm get_world_comm() const {
    return world_comm;
  }

  /** Return the communicator for this node. */
  const El::mpi::Comm get_node_comm() const {
    return node_comm;
  }

  /** Return true if rank (in comm) is on the local node. */
  bool is_rank_node_local(int rank, const El::mpi::Comm comm) const {
    // Translating to COMM_WORLD is typically constant time.
    int world_rank = El::mpi::Translate(comm, rank, get_world_comm());
    return is_world_rank_on_node(world_rank);
  }

  /** throws an lbann_exception **/
  void lbann_comm_abort(std::string msg);

 private:
  /** World communicator. */
  const El::mpi::Comm world_comm;
  /** Communicator for every process in this model. */
  El::mpi::Comm model_comm;
  /** Communicator for every process with the same model rank. */
  El::mpi::Comm intermodel_comm;
  /** Communicator for every process in the same compute node. */
  El::mpi::Comm node_comm;
  /** Grid for this model. */
  Grid *grid;
  /** Number of models. */
  int num_models;
  /** Number of processors per model. */
  int procs_per_model;
  /** Rank of the model this process is in. */
  int model_rank;
  /** Rank of this process within its model. */
  int rank_in_model;
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
  /** Pre-allocated buffers for collectives. */
  std::map<size_t, std::vector<uint8_t *>> collective_bufs;
  /** Current default allreduce algorithm. */
  allreduce_algorithm default_allreduce_algo =
    allreduce_algorithm::DYNAMIC;

#ifdef LBANN_HAS_ALUMINUM
  using al_comms_key_type = std::pair<MPI_Comm, std::type_index>;
  using al_comms_val_type = std::unique_ptr<::Al::MPICommunicator>;
  std::map<al_comms_key_type, al_comms_val_type> m_al_comms;
#ifdef AL_HAS_NCCL
  /** Number of streams to round-robin between for NCCL allreduces. */
  static constexpr int m_num_al_nccl_streams = 5;
  /** Streams for non-blocking NCCL allreduces. */
  cudaStream_t m_al_nccl_streams[m_num_al_nccl_streams];
  /** Counter for round-robin'ing across streams. */
  size_t m_al_cur_nccl_req = 0;
  /** Event for synchronizing between LBANN and NCCL streams. */
  cudaEvent_t m_al_nccl_sync_event;
  /** Events used to check for completion of NCCL allreduces. */
  std::vector<cudaEvent_t> m_al_nccl_req_events;
#endif

  /** Get an Aluminum communicator.
   *  The communicator will have the same process configuration as the
   *  Elemental communicator c and use the backend corresponding to
   *  type index t. An Aluminum communicator will be created if
   *  needed.
   */
  ::Al::MPICommunicator* get_al_comm(
    El::mpi::Comm c, std::type_index t = std::type_index(typeid(Al::mpi_backend)));

  /** Convert an MPI_Op to an Aluminum reduction operator. */
  ::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op);
#endif

  // Various statistics counters.
  size_t num_model_barriers;
  size_t num_intermodel_barriers;
  size_t num_global_barriers;
  size_t bytes_sent;
  size_t bytes_received;
  // Allreduce statistics.
  size_t ar_bytes_sent;
  size_t ar_bytes_received;
  size_t ar_rs_bytes_sent;
  size_t ar_rs_bytes_received;
  size_t ar_ag_bytes_sent;
  size_t ar_ag_bytes_received;
  double ar_time;
  double ar_rs_time;
  double ar_ag_time;
  double ar_send_transform_time;
  double ar_recv_transform_time;
  double ar_recv_apply_transform_time;
  double ar_send_time;
  double ar_recv_time;
  double ar_rs_send_time;
  double ar_rs_recv_time;
  double ar_ag_send_time;
  double ar_ag_recv_time;

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

  /**
   * Return a buffer from collective_bufs, allocating it if needed.
   * @param size The size of the buffer (in bytes).
   * @param idx The index of the buffer (default 0).
   */
  uint8_t *get_collective_buffer(size_t size, size_t idx = 0);

};

template <typename T, bool S>
void lbann_comm::broadcast(int root, T& val, const El::mpi::Comm c) {
  if (S) {
    // Avoid linking error from uninstantiated El::mpi routine if !S by converting T to El::byte
    using TT = typename interpret_as_byte_if_needed<S, T>::type;
    broadcast_native<TT>(root, reinterpret_cast<TT&>(val), c);
  } else {
    broadcast_custom(root, val, c);
  }
  count_bytes_broadcast(sizeof(T), El::mpi::Rank(c), root);
}

template <typename T>
void lbann_comm::broadcast_native(int root, T& val, const El::mpi::Comm c) const {
  El::mpi::Broadcast(val, root, c);
}

template <typename T>
void lbann_comm::broadcast_custom(int root, T& val, const El::mpi::Comm c) const {
 const int bytes =  static_cast<int>(sizeof(T));
 El::mpi::Broadcast<El::byte>(reinterpret_cast<El::byte*>(&val), bytes, root, c);
}

template <typename T, bool S>
void lbann_comm::broadcast(const int root, T* data, const int count, const El::mpi::Comm c) {
  const int size = static_cast<int>(S? count : sizeof(T)*count);
  // Avoid linking error from uninstantiated El::mpi routine if !S by converting T to El::byte
  using TT = typename interpret_as_byte_if_needed<S, T>::type;
  El::mpi::Broadcast<TT>(reinterpret_cast<TT*>(data), size, root, c);
  count_bytes_broadcast(sizeof(T)*count, El::mpi::Rank(c), root);
}

/// Broadcast std::string over an arbitrary communicator.
template<>
void lbann_comm::broadcast<std::string>(const int root, std::string& str, const El::mpi::Comm c);

} // namespace lbann

#endif  // LBANN_COMM_HPP_INCLUDED
