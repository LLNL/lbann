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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_COMM_HPP_INCLUDED
#define LBANN_COMM_HPP_INCLUDED

#include "base.hpp"

#ifdef LBANN_HAS_CUDA
#include <cuda_runtime.h>
#endif // LBANN_HAS_CUDA
#ifdef LBANN_HAS_ALUMINUM
#include <Al.hpp>
#endif // LBANN_HAS_ALUMINUM

#include "lbann/comm_nb_request.hpp"

#include "detect_El_mpi.hpp"

#include <map>
#include <typeindex>
#include <vector>

namespace lbann {

#ifdef LBANN_HAS_ALUMINUM
/** Convert an MPI_Op to an Aluminum reduction operator. */
::Al::ReductionOperator mpi_op_to_al_op(El::mpi::Op op);
#endif

/** Grid types in sub-grid parallelism (2nd order) */
enum class GridType
{
  NO_GRID = 0,
  PRIMARY_GRID = 1,
  SECONDARY_GRID = 2
};

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
class lbann_comm
{
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

  /** @brief Construct communicators for trainers
   *
   *  Invalidates any existing trainer communicators.
   *
   *  @param procs_per_trainer Number of MPI ranks in a trainer.
   *  Default is size of world communicator.
   *  @param trainer_grid_height Height of 2D process grid for each
   *  trainer. Must divide @c procs_per_trainer. Default grid is
   *  approximately square.
   */
  void split_trainers(int procs_per_trainer = -1, int trainer_grid_height = -1);

  /** Split the commicator for the given trainer into primary and secondary
   *
   *  @param num_process_primar_grid Absolute number of MPI ranks
   *  assigned to the primary grid
   *  @param create_two_models Create a secondary copy of the model on
   *  the secondary grid to perform redundant computation and minimize
   *  communication.
   *  @param enable_async_comm Use non-blocking sends and receivces
   *  @param enable_topo_aware Assign primary and secondary grid
   *  resources so that they are interleaved and thus should be
   *  allocated to the same compute node assuming that there are
   *  always an even number of accelerators per node.
   */
  void split_trainer_grid(int num_process_primary_grid = 0,
                          bool create_two_models = false,
                          bool enable_async_comm = false,
                          bool enable_topo_aware = false);

  /** Get trainer grid number (0: no primary/secondary grid, 1: part of primary
   * grid, 2: part of secondary grid). */
  inline GridType get_grid_type() const noexcept { return m_grid_type; }

  /** Get which trainer this process is in. */
  inline int get_trainer_rank() const noexcept { return m_trainer_rank; }
  /** Get the rank of this process in its trainer. */
  inline int get_rank_in_trainer() const noexcept { return m_rank_in_trainer; }
  /** Get my rank in COMM_WORLD. */
  inline int get_rank_in_world() const
  {
    return El::mpi::Rank(get_world_comm());
  }
  /** Return the COMM_WORLD rank of the rank'th processor in trainer. */
  inline int get_world_rank(int trainer, int rank) const noexcept
  {
    if (m_secondary_grid_ranks.size() == 0) {
      return m_procs_per_trainer * trainer + rank;
    }
    else {
      return (m_secondary_grid_ranks.size() + m_primary_grid_ranks.size()) *
               trainer +
             rank;
    }
  }
  /** Return the "rank" of the trainer that this rank is in */
  inline int map_world_rank_to_trainer_rank(int world_rank) const noexcept
  {
    return (world_rank / m_procs_per_trainer);
  }
  /** Return the "rank" within the trainer that this rank is in */
  inline int map_world_rank_to_rank_in_trainer(int world_rank) const noexcept
  {
    return (world_rank % m_procs_per_trainer);
  }
  /** Return the rank of the master process in this trainer. */
  inline int get_trainer_master() const noexcept { return 0; }
  /** Return the rank of the inter-trainer master process. */
  inline int get_intertrainer_master() const noexcept { return 0; }
  /** Return the rank of the world master process. */
  inline int get_world_master() const noexcept { return 0; }
  /** Return true if this process is the master process in its trainer. */
  inline bool am_trainer_master() const noexcept
  {
    return get_rank_in_trainer() == get_trainer_master();
  }
  /** Return true if this process is the world master process. */
  inline bool am_world_master() const noexcept
  {
    return get_rank_in_world() == get_world_master();
  }
  /** Return a grid to use for this trainer. */
  inline El::Grid& get_trainer_grid() { return *m_grid; }
  /** Return a read-only grid to use for this trainer. */
  inline const El::Grid& get_trainer_grid() const { return *m_grid; }
  /** Return secondary grid to use for this trainer when sub-grid parallelism is
   * enabled. */
  inline El::Grid& get_secondary_grid() { return *m_secondary_grid; }
  /** Return read-only secondary grid to use for this trainer. */
  inline const El::Grid& get_secondary_grid() const
  {
    return *m_secondary_grid;
  }
  /** Return subset grid to use for this trainer when sub-grid parallelism is
   * enabled. */
  inline El::Grid& get_subset_grid() { return *m_subset_grid; }
  /** Return read-only subset grid to use for this trainer when sub-grid
   * parallelism is enabled. */
  inline const El::Grid& get_subset_grid() const { return *m_subset_grid; }
  /** Return the total number of trainers. */
  inline int get_num_trainers() const noexcept { return m_num_trainers; }
  /* Return the number of processes in a trainer. */
  inline int get_procs_per_trainer() const noexcept
  {
    return m_procs_per_trainer;
  }
  /** Return the number of processes in a compute node. */
  inline int get_procs_per_node() const noexcept { return m_procs_per_node; }
  /** Return the total number of ranks. */
  inline int get_procs_in_world() const
  {
    return El::mpi::Size(get_world_comm());
  }
  /** Return the rank of this process within its compute node. */
  inline int get_rank_in_node() const noexcept { return m_rank_in_node; }
  /** Return true if rank (in COMM_WORLD) is on this compute node. */
  inline bool is_world_rank_on_node(int rank) const
  {
    return std::find(m_world_ranks_on_node.begin(),
                     m_world_ranks_on_node.end(),
                     rank) != m_world_ranks_on_node.end();
  }

  /** Get default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided.
   */
  inline int get_default_threads_per_proc() const noexcept
  {
    return m_threads_per_proc;
  }

  /** Reset the number of threads per process to the default. */
  void reset_threads() const noexcept;

  /** Perform a sum reduction of mat over the inter-trainer communicator. */
  void intertrainer_sum_matrix(AbsMat& mat) const;
  void intertrainer_sum_matrix(AbsDistMat& mat) const;
  /** Broadcast mat over the inter-trainer communicator starting from root. */
  void intertrainer_broadcast_matrix(AbsMat& mat, int root) const;
  void intertrainer_broadcast_matrix(AbsDistMat& mat, int root) const;

  /// Broadcast a scalar value over an arbitrary communicator
  template <typename T, bool S = is_instantiated_El_mpi_type<T>::value>
  void broadcast(int root, T& val, const El::mpi::Comm& c) const;

  template <typename T>
  void broadcast_custom(int root, T& val, const El::mpi::Comm& c) const;
  template <typename T>
  void broadcast_native(int root, T& val, const El::mpi::Comm& c) const;

  /// World broadcast of a scalar.
  template <typename T>
  void world_broadcast(int root, T& val) const;
  /// Inter-trainer broadcast of a scalar.
  template <typename T>
  void intertrainer_broadcast(int root, T& val) const;
  /// Within-trainer broadcast of a scalar.
  template <typename T>
  void trainer_broadcast(int root, T& val) const;

  /**
   * Broadcast a buffer over an arbitrary communicator assuming that
   * the buffer space is already allocated.
   */

  // Default to cpu memory
  template <typename T>
  void broadcast(const int root,
                 T* data,
                 const int count,
                 const El::mpi::Comm& c) const;

  template <typename T,
            El::Device D,
            bool S = is_instantiated_El_mpi_type<T>::value>
  void broadcast(const int root,
                 T* data,
                 const int count,
                 const El::mpi::Comm& c,
                 El::SyncInfo<D> const& syncInfo) const;

  /// World broadcast of a buffer.
  template <typename T>
  void world_broadcast(const int root, T* data, const int count) const;

  template <typename T, El::Device D>
  void world_broadcast(const int root,
                       T* data,
                       const int count,
                       El::SyncInfo<D> const& syncInfo) const;
  /// Inter-trainer broadcast of a buffer.
  template <typename T>
  void intertrainer_broadcast(const int root, T* data, const int count) const;
  template <typename T, El::Device D>
  void intertrainer_broadcast(const int root,
                              T* data,
                              const int count,
                              El::SyncInfo<D> const& syncInfo) const;
  /// Within-trainer broadcast of a buffer.
  template <typename T>
  void trainer_broadcast(const int root, T* data, const int count) const;

  template <typename T, El::Device D>
  void trainer_broadcast(const int root,
                         T* data,
                         const int count,
                         El::SyncInfo<D> const& syncInfo) const;

  /**
   * Resize vector<> over an arbitrary communicator to match the one on root.
   */
  template <typename T>
  size_t
  resize(const int root, std::vector<T>& data, const El::mpi::Comm& c) const;

  /**
   * Broadcast vector<> over an arbitrary communicator;
   * vector<> for non-root processes will be resized as needed.
   */
  template <typename T>
  void
  broadcast(const int root, std::vector<T>& data, const El::mpi::Comm& c) const;
  /// Broadcast vector<> to world.
  template <typename T>
  void world_broadcast(int root, std::vector<T>& data) const;
  /**
   * Broadcast vector<> within trainer;
   * vector<> for non-root processes will be resized as needed.
   */
  /// Broadcast vector<> across trainers.
  template <typename T>
  void intertrainer_broadcast(int root, std::vector<T>& data) const;
  /// Broadcast vector<> within trainer.
  template <typename T>
  void trainer_broadcast(int root, std::vector<T>& data) const;

  /** Allgather over an arbitrary communicator */
  template <typename T>
  void all_gather(const T* src,
                  int src_count,
                  T* rcv,
                  int rcv_count,
                  const El::mpi::Comm& c) const;
  template <typename T, El::Device D>
  void all_gather(const T* src,
                  int src_count,
                  T* rcv,
                  int rcv_count,
                  const El::mpi::Comm& c,
                  El::SyncInfo<D> const& syncInfo) const;

  /**
   * Allgatherv over an arbitrary communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void all_gather(std::vector<T> const& src,
                  std::vector<T>& rcs,
                  std::vector<int> const& rcv_counts,
                  std::vector<int> const& rcv_disp,
                  const El::mpi::Comm& c) const;
  /**
   * Allgatherv over a trainer communicator;
   * all vectors must be correctly sized prior to entry.
   */
  template <typename T>
  void trainer_all_gather(std::vector<T> const& src,
                          std::vector<T>& rcs,
                          std::vector<int> const& rcv_counts,
                          std::vector<int> const& rcv_disp) const;
  /**
   * Allgather for a single element over an arbitrary communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void
  all_gather(T const& src, std::vector<T>& data, const El::mpi::Comm& c) const;
  /**
   * Allgather for a single element over the world communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void world_all_gather(T const& src, std::vector<T>& data) const;
  /**
   * Allgather for a single element over the trainer communicator;
   * std::vector<T> &data must be correctly sized prior to entry.
   */
  template <typename T>
  void trainer_all_gather(T const& src, std::vector<T>& data) const;

  /** Within-trainer scalar gather (for non-root processes). */
  template <typename T>
  void trainer_gather(T snd, int root) const;
  /** Within-trainer scalar gather (for root processes). */
  template <typename T>
  void trainer_gather(T snd, T* rcv) const;
  /** Within-trainer scalar-array gather (for non-root processes). */
  template <typename T>
  void trainer_gather(T const* snd, int count, int root) const;
  /** Within-trainer scalar-array gather (for root processes). */
  template <typename T>
  void trainer_gather(T const* snd, int count, T* rcv) const;
  /** Within-trainer variable-length-array gather (for non-root processes). */
  template <typename T>
  void trainer_gatherv(T const* snd, int count, int root) const;
  template <typename T>
  void trainer_gatherv(T const* snd,
                       int count,
                       T* rcv,
                       int const* rcv_counts,
                       int const* rcv_displacements) const;
  /** Inter-trainer gather (for non-root processes). */
  template <typename T>
  void intertrainer_gather(T snd, int root) const;
  /** Inter-trainer gather (for root processes). */
  template <typename T>
  void intertrainer_gather(T snd, std::vector<T>& rcv) const;
  /** Inter-trainer scalar-array gather (for non-root processes). */
  template <typename T>
  void intertrainer_gather(T const* snd, int count, int root) const;
  /** Inter-trainer scalar-array gather (for root processes). */
  template <typename T>
  void intertrainer_gather(T const* snd, int count, T* rcv) const;
  /** Scalar gather (for non-root processes). */
  template <typename T>
  void gather(T snd, int root, const El::mpi::Comm& c) const;
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, T* rcv, const El::mpi::Comm& c) const;
  /** Scalar gather (for root processes). */
  template <typename T>
  void gather(T snd, std::vector<T>& rcv, const El::mpi::Comm& c) const;
  /** Scalar-array gather (for non-root processes). */
  template <typename T>
  void gather(T const* snd, int count, int root, const El::mpi::Comm& c) const;
  template <typename T, El::Device D>
  void gather(T const* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo) const;
  /** Scalar-array gather (for root processes). */
  template <typename T>
  void gather(T const* snd, int count, T* rcv, const El::mpi::Comm& c) const;
  template <typename T, El::Device D>
  void gather(T const* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo) const;
  /** Scalar scatter (for non-root processes). */
  template <typename T>
  T scatter(int root, const El::mpi::Comm& c) const;
  /** Scalar scatter (for root processes). */
  template <typename T>
  T scatter(T const* snd, const El::mpi::Comm& c) const;
  /** Inter-trainer reduce (for non-root processes). */
  template <typename T>
  void
  intertrainer_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM) const;
  /** Inter-trainer reduce (for root processes). */
  template <typename T>
  T intertrainer_reduce(T snd, El::mpi::Op op = El::mpi::SUM) const;
  /** Within-trainer reduce (for non-root processes). */
  template <typename T>
  void trainer_reduce(T snd, int root, El::mpi::Op op = El::mpi::SUM) const;
  /** Within-trainer reduce (for root processes). */
  template <typename T>
  T trainer_reduce(T snd, El::mpi::Op op = El::mpi::SUM) const;
  /** Within-trainer scalar array reduce (for non-root processes). */
  template <typename T>
  void trainer_reduce(T const* snd,
                      int count,
                      int root,
                      El::mpi::Op op = El::mpi::SUM) const;
  /** Within-trainer scalar array reduce (for root processes). */
  template <typename T>
  void trainer_reduce(T const* snd,
                      int count,
                      T* rcv,
                      El::mpi::Op op = El::mpi::SUM) const;
  /** Scalar reduce (for non-root processes). */
  template <typename T>
  void reduce(T snd,
              int root,
              const El::mpi::Comm& c,
              El::mpi::Op op = El::mpi::SUM) const;
  /** Scalar reduce (for root processes). */
  template <typename T>
  T reduce(T snd, const El::mpi::Comm& c, El::mpi::Op op = El::mpi::SUM) const;

  /** Scalar-array reduce (for non-root processes). */
  // Op is "SUM"
  template <typename T>
  void reduce(T const* snd, int count, int root, const El::mpi::Comm& c) const;
  template <typename T, El::Device D>
  void reduce(T const* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo) const;

  template <typename T>
  void reduce(T const* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::mpi::Op op) const;
  template <typename T, El::Device D>
  void reduce(T const* snd,
              int count,
              int root,
              const El::mpi::Comm& c,
              El::mpi::Op op,
              El::SyncInfo<D> const& syncInfo) const;
  /** Scalar-array reduce (for root processes). */
  template <typename T, El::Device D>
  void reduce(T const* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::SyncInfo<D> const& syncInfo) const;
  template <typename T>
  void reduce(T const* snd, int count, T* rcv, const El::mpi::Comm& c) const;

  template <typename T>
  void reduce(T const* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::mpi::Op op) const;
  template <typename T, El::Device D>
  void reduce(T const* snd,
              int count,
              T* rcv,
              const El::mpi::Comm& c,
              El::mpi::Op op,
              El::SyncInfo<D> const& syncInfo) const;
  /** Inter-trainer all-reduce. */
  template <typename T>
  T intertrainer_allreduce(T snd, El::mpi::Op op = El::mpi::SUM) const;
  /** Within-trainer all-reduce. */
  template <typename T>
  T trainer_allreduce(T snd, El::mpi::Op op = El::mpi::SUM) const;
  /** Scalar array within-trainer all-reduce. */
  template <typename T>
  void trainer_allreduce(T const* snd,
                         int count,
                         T* rcv,
                         El::mpi::Op op = El::mpi::SUM) const;
  /** Scalar allreduce. */
  template <typename T>
  T allreduce(T snd,
              const El::mpi::Comm& c,
              El::mpi::Op op = El::mpi::SUM) const;

  // FIXME (trb): Based on the backend choice of "MPIBackend", I'm
  // assuming this is intended as a CPU-only call.
  /** Scalar-array allreduce. */
  template <typename T>
  void allreduce(T const* snd,
                 int count,
                 T* rcv,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM) const;
  /** In-place scalar-array allreduce. */
  template <typename T>
  void allreduce(T* data,
                 int count,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM) const;
  /** Matrix allreduce. */
  template <typename TensorDataType>
  void allreduce(El::AbstractMatrix<TensorDataType>& m,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM) const;
  /** Matrix allreduce. */
  template <typename TensorDataType>
  void allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                 const El::mpi::Comm& c,
                 El::mpi::Op op = El::mpi::SUM) const;
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  template <typename TensorDataType>
  void nb_allreduce(El::AbstractMatrix<TensorDataType>& m,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM) const;
  /** Non-blocking matrix allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix allreduce.
   */
  template <typename TensorDataType>
  void nb_allreduce(El::AbstractDistMatrix<TensorDataType>& m,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM) const;
  /** Non-blocking in-place scalar-array allreduce.
   *  If LBANN has not been built with Aluminum, then this calls a blocking
   *  allreduce.
   *  This currently only supports host pointers (i.e. the MPI backend).
   */
  template <typename T>
  void nb_allreduce(T* data,
                    int count,
                    const El::mpi::Comm& c,
                    Al::request& req,
                    El::mpi::Op op = El::mpi::SUM) const;
  /** Non-blocking matrix reduce-scatter.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix operation.
   */
  template <typename TensorDataType>
  void nb_reduce_scatter(const El::AbstractMatrix<TensorDataType>& src,
                         El::AbstractMatrix<TensorDataType>& dst,
                         const El::mpi::Comm& c,
                         Al::request& req,
                         El::mpi::Op op = El::mpi::SUM) const;
  /** Non-blocking matrix reduce-scatter.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix operation.
   */
  template <typename TensorDataType>
  void nb_reduce_scatter(const El::AbstractDistMatrix<TensorDataType>& src,
                         El::AbstractDistMatrix<TensorDataType>& dst,
                         const El::mpi::Comm& c,
                         Al::request& req,
                         El::mpi::Op op = El::mpi::SUM) const;
  /** Non-blocking matrix reduce-scatter.
   *  If LBANN has not been built with Aluminum, then this calls a
   *  blocking matrix operation.
   */
  template <typename T>
  void nb_reduce_scatter(const T* src,
                         T* dst,
                         int count,
                         const El::mpi::Comm& c,
                         Al::request& req,
                         El::mpi::Op op = El::mpi::SUM) const;
  /** Wait for a all non-blocking requests to complete. */
  template <typename T>
  void wait_all(std::vector<El::mpi::Request<T>>& req) const;

  /** Wait for a non-blocking request to complete. */
  template <typename T>
  void wait(El::mpi::Request<T>& req) const;

  /** Wait for a non-blocking request to complete. */
  void wait(Al::request& req) const;
  /** Test whether a non-blocking request has completed; true if it has. */
  bool test(Al::request& req) const;

  /** Barrier among the inter-trainer processes. */
  void intertrainer_barrier() const;
  /** Barrier among processes in this trainer. */
  void trainer_barrier() const;
  /** Barrier among all processes. */
  void global_barrier() const;
  /** Barrier on an arbitrary communicator. */
  void barrier(const El::mpi::Comm& c) const;

  /** Send a buffer to rank in trainer. */
  template <typename T>
  void send(const T* data, int count, int trainer, int rank) const;
  template <typename T, El::Device D>
  void send(const T* data,
            int count,
            int trainer,
            int rank,
            El::SyncInfo<D> const& syncInfo) const;
  template <typename T, El::Device D>
  void send(const T* data,
            int count,
            int trainer,
            El::SyncInfo<D> const& syncInfo) const;
  void send(const AbsMat& mat, int trainer, int rank) const;
  void send(const DistMat& mat, int trainer, int rank) const;
  void send(const AbsMat& mat, int trainer) const
  {
    send(mat, trainer, m_rank_in_trainer);
  }
  void send(const DistMat& mat, int trainer) const
  {
    send(mat, trainer, m_rank_in_trainer);
  }

  /** Corresponding non-blocking sends. */
  template <typename T>
  void nb_send(const T* data,
               int count,
               int trainer,
               int rank,
               El::mpi::Request<T>& req) const;
  template <typename T>
  void nb_tagged_send(const T* data,
                      int count,
                      int rank,
                      int tag,
                      El::mpi::Request<T>& req,
                      const El::mpi::Comm& c) const;
  template <typename T>
  void nb_send(const T* data,
               int count,
               int trainer,
               El::mpi::Request<T>& req) const;
  void nb_send(const AbsMat& mat,
               int trainer,
               int rank,
               El::mpi::Request<DataType>& req) const;
  void nb_send(const DistMat& mat,
               int trainer,
               int rank,
               El::mpi::Request<DataType>& req) const;
  void
  nb_send(const AbsMat& mat, int trainer, El::mpi::Request<DataType>& req) const
  {
    nb_send(mat, trainer, m_rank_in_trainer, req);
  }
  void nb_send(const DistMat& mat,
               int trainer,
               El::mpi::Request<DataType>& req) const
  {
    nb_send(mat, trainer, m_rank_in_trainer, req);
  }

  /** Corresponding receive to send. */
  template <typename T>
  void recv(T* data, int count, int trainer, int rank) const;
  template <typename T>
  void recv(T* data, int count, int trainer) const;
  template <typename T>
  void recv(T* data, int count) const;
  template <typename T, El::Device D>
  void recv(T* data,
            int count,
            int trainer,
            int rank,
            El::SyncInfo<D> const& syncInfo) const;
  template <typename T, El::Device D>
  void
  recv(T* data, int count, int trainer, El::SyncInfo<D> const& syncInfo) const;
  void recv(AbsMat& mat, int trainer, int rank) const;
  void recv(DistMat& mat, int trainer, int rank) const;
  void recv(AbsMat& mat, int trainer) const
  {
    recv(mat, trainer, m_rank_in_trainer);
  }
  void recv(DistMat& mat, int trainer) const
  {
    recv(mat, trainer, m_rank_in_trainer);
  }
  /** As above, but receive from anyone. */
  template <typename T, El::Device D>
  void recv(T* data, int count, El::SyncInfo<D> const& syncInfo) const;
  void recv(AbsMat& mat) const;
  void recv(DistMat& mat) const;

  /** Corresponding non-blocking receives. */
  template <typename T>
  void nb_recv(T* data,
               int count,
               int trainer,
               int rank,
               El::mpi::Request<T>& req) const;
  template <typename T>
  void nb_tagged_recv(T* data,
                      int count,
                      int rank,
                      int tag,
                      El::mpi::Request<T>& req,
                      const El::mpi::Comm& c) const;

  template <typename T>
  void nb_recv(T* data, int count, int trainer, El::mpi::Request<T>& req) const;
  void nb_recv(AbsMat& mat,
               int trainer,
               int rank,
               El::mpi::Request<DataType>& req) const;
  void nb_recv(DistMat& mat,
               int trainer,
               int rank,
               El::mpi::Request<DataType>& req) const;
  void nb_recv(AbsMat& mat, int trainer, El::mpi::Request<DataType>& req) const
  {
    nb_recv(mat, trainer, m_rank_in_trainer, req);
  }
  void nb_recv(DistMat& mat, int trainer, El::mpi::Request<DataType>& req) const
  {
    nb_recv(mat, trainer, m_rank_in_trainer, req);
  }
  template <typename T>
  void nb_recv(T* data, int count, El::mpi::Request<T>& req) const;
  void nb_recv(AbsMat& mat, El::mpi::Request<DataType>& req) const;
  void nb_recv(DistMat& mat, El::mpi::Request<DataType>& req) const;

  /** Send/recv to/from ranks. */
  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                int send_rank,
                T* rcv,
                int recv_count,
                int recv_trainer,
                int recv_rank) const;
  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                T* rcv,
                int recv_count,
                int recv_trainer) const;

  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                int send_rank,
                T* rcv,
                int recv_count,
                int recv_trainer,
                int recv_rank,
                El::SyncInfo<D> const& syncInfo) const;
  template <typename T, El::Device D>
  void sendrecv(const T* snd,
                int send_count,
                int send_trainer,
                T* rcv,
                int recv_count,
                int recv_trainer,
                El::SyncInfo<D> const& syncInfo) const;

  /** Determine the size (count) of an incoming message. */
  template <typename T>
  int get_count(int trainer, int rank) const;
  template <typename T>
  int get_count(int trainer) const;

  // Statistics methods.
  /** Return the number of trainer barriers performed. */
  inline size_t get_num_trainer_barriers() const noexcept
  {
    return m_num_trainer_barriers;
  }
  /** Return the number of inter-trainer barriers performed. */
  inline size_t get_num_intertrainer_barriers() const noexcept
  {
    return m_num_intertrainer_barriers;
  }
  /** Return the number of global barriers performed. */
  inline size_t get_num_global_barriers() const noexcept
  {
    return m_num_global_barriers;
  }
  /** Return the number of bytes sent. */
  inline size_t get_bytes_sent() const noexcept { return m_bytes_sent; }
  /** Return the number of bytes received. */
  inline size_t get_bytes_received() const noexcept { return m_bytes_received; }

  inline void reset_stats_counters() noexcept
  {
    m_num_trainer_barriers = 0;
    m_num_intertrainer_barriers = 0;
    m_num_global_barriers = 0;
    m_bytes_sent = 0;
    m_bytes_received = 0;
  }

  /** Return true if mat can be transmitted. */
  static inline bool is_sendable(const AbsMat& mat) noexcept
  {
    // This assumes we do not transmit mat with a datatype smaller than
    // DataType.
    // MPI uses "int" as its count type; do calculations with larger ints.
    size_t count = (size_t)mat.Height() * (size_t)mat.Width();
    return count <= (size_t)std::numeric_limits<int>::max();
  }
  /** Return true if the local portion of dist_mat can be transmitted. */
  static inline bool is_sendable(const AbsDistMat& dist_mat) noexcept
  {
    return is_sendable(dist_mat.LockedMatrix());
  }

  /** Developer's note: to get the raw MPI communicator, which may be needed
   *  when working with external libraries, by example:
   *     comm->get_intertrainer_comm().GetMPIComm()
   */

  /** Return the intertrainer communicator. */
  const El::mpi::Comm& get_intertrainer_comm() const noexcept
  {
    return m_intertrainer_comm;
  }

  /** Return the trainer communicator. */
  const El::mpi::Comm& get_trainer_comm() const noexcept
  {
    return m_trainer_comm;
  }

  /** Return the combined grid communicator for a trainer. */
  const El::mpi::Comm& get_combined_grid_comm() const noexcept
  {
    return m_combined_grid_comm;
  }

  /** Return the world communicator. */
  const El::mpi::Comm& get_world_comm() const noexcept { return m_world_comm; }

  /** Return the communicator for this node. */
  const El::mpi::Comm& get_node_comm() const noexcept { return m_node_comm; }

  /** Return the communicator for this grid in sub-grid parallelism. */
  const El::mpi::Comm& get_KFAC_comm() const noexcept { return m_trainer_comm; }

  /** Return the ranks of primary grid in the trainer */
  std::vector<int> get_primary_grid_ranks() { return m_primary_grid_ranks; }

  /** Return the ranks of secondary grid in the trainer */
  std::vector<int> get_secondary_grid_ranks() { return m_secondary_grid_ranks; }

  bool get_KFAC_subgrid_create_two_models() { return m_create_two_models; }

  /** Return asynchronous flag for sub-grid parallelism */
  bool enable_subgrid_async_communication() { return m_subgrid_async_progress; }

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
  bool is_rank_node_local(int rank, const El::mpi::Comm& comm) const
  {
    // Translating to COMM_WORLD is typically constant time.
    int world_rank = El::mpi::Translate(comm, rank, get_world_comm());
    return is_world_rank_on_node(world_rank);
  }

  /** throws an lbann_exception **/
  void lbann_comm_abort(std::string msg) const;

private:
  /** World communicator. */
  const El::mpi::Comm m_world_comm;
  /** Communicator for every process in this trainer. */
  El::mpi::Comm m_trainer_comm;
  /** Communicator for every process with the same trainer rank. */
  El::mpi::Comm m_intertrainer_comm;
  /** Communicator for every process in the same compute node. */
  El::mpi::Comm m_node_comm;
  /** Communicator for primary grid in each trainer */
  El::mpi::Comm m_primary_grid_comm;
  /** Communicator for secondary grid in each trainer */
  El::mpi::Comm m_secondary_grid_comm;
  /** Combined communicator for primary and secondary grid in each trainer */
  El::mpi::Comm m_combined_grid_comm;
  /** Packed group communicators. */
  mutable std::unordered_map<int, El::mpi::Comm> m_group_communicators;
  /** Grid for this trainer. */
  std::unique_ptr<El::Grid> m_grid;
  /** Number of trainers. */
  int m_num_trainers;
  /** Number of processors per trainer. */
  int m_procs_per_trainer;
  /** Rank of the trainer this process is in. */
  int m_trainer_rank;
  /** Rank of this process within its trainer. */
  int m_rank_in_trainer;
  /** Number of processers per compute node. */
  int m_procs_per_node;
  /** Rank of this process within its compute node. */
  int m_rank_in_node;
  /** The list of world ranks that are on this compute node. */
  std::vector<int> m_world_ranks_on_node;
  /** Default number of threads per process.
   *  This is the number of OpenMP threads to use for parallel
   *  regions, provided omp_set_num_threads has not been called or the
   *  num_threads directive has not been provided.
   */
  int m_threads_per_proc;

  /** Grid type for current process when sub-grid parallelism is enabled */
  GridType m_grid_type = GridType::NO_GRID;

  bool m_create_two_models = false, m_subgrid_async_progress = false;

  std::unique_ptr<El::Grid> m_secondary_grid, m_subset_grid;

  /**
  Ranks in primary and secondary grids
  */
  std::vector<int> m_primary_grid_ranks;
  std::vector<int> m_secondary_grid_ranks;

  // Various statistics counters.
  mutable size_t m_num_trainer_barriers;
  mutable size_t m_num_intertrainer_barriers;
  mutable size_t m_num_global_barriers;
  mutable size_t m_bytes_sent;
  mutable size_t m_bytes_received;

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
   * Keep track of the number of broadcast bytes transmitted and received
   */
  void count_bytes_broadcast(const size_t bytes,
                             const int rank,
                             const int root) const noexcept
  {
    if (rank == root) {
      m_bytes_sent += bytes;
    }
    else {
      m_bytes_received += bytes;
    }
  }
}; // class lbann_comm

/** Get the current rank within MPI_COMM_WORLD.
 *  This function is safe to call even if MPI has not initialized or
 *  has been finalized. In either case it returns a negative value.
 */
int get_rank_in_world();

} // namespace lbann

#endif // LBANN_COMM_HPP_INCLUDED
