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
//
// lbann_comm .hpp .cpp - LBANN communication utilities
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_COMM_HPP_INCLUDED
#define LBANN_COMM_HPP_INCLUDED

#include <vector>
#include <unordered_map>
#include "lbann_base.hpp"
using namespace El;

namespace lbann
{

  /**
   * Manage communication.
   * This supports separate models, each of which are split over potentially
   * several processes. Every model is split over the same number of processes.
   * The corresponding processes between models are on the "inter-model
   * communicator".
   * You can also do point-to-point or broadcast communication to arbitrary sets
   * of processes.
   */
  class lbann_comm
  {
  public:
    /**
     * Init communicators for models each with procs_per_model processes,
     * defaulting to every process in one model.
     */
    lbann_comm(int procs_per_model = 0);
    ~lbann_comm();

    /** Get which model this process is in. */
    inline int get_model_rank() const { return model_rank; }
    /** Get the rank of this process in its model. */
    inline int get_rank_in_model() const { return rank_in_model; }
    /** Get my rank in COMM_WORLD. */
    inline int get_rank_in_world() const { return mpi::Rank(mpi::COMM_WORLD); }
    /** Return the COMM_WORLD rank of the rank'th processor in model. */
    inline int get_world_rank(int model, int rank) const {
      return procs_per_model * model + rank;
    }
    /** Return the rank of the master process in this model. */
    inline int get_model_master() const { return 0; }
    /** Return the rank of the inter-model master process. */
    inline int get_intermodel_master() const { return 0; }
    /** Return the rank of the world master process. */
    inline int get_world_master() const { return 0; }
    /** Return true if this process is the master process in its model. */
    inline bool am_model_master() const {
      return get_rank_in_model() == get_model_master();
    }
    /** Return true if this process is the world master process. */
    inline bool am_world_master() const {
      return get_rank_in_world() == get_world_master();
    }
    /** Return a grid to use for this model. */
    inline Grid& get_model_grid() { return *grid; }
    /** Return the total number of models. */
    inline int get_num_models() const { return num_models; }
    /* Return the number of processes in a model. */
    inline int get_procs_per_model() const { return procs_per_model; }
    /** Return the number of processes in a compute node. */
    inline int get_procs_per_node() const { return procs_per_node; }
    /** Return the rank of this process within its compute node. */
    inline int get_rank_in_node() const { return rank_in_node; }
    /** Return true if rank (in the model comm) is on this compute node. */
    inline bool is_model_rank_on_node(int rank) const {
      return std::find(model_ranks_on_node.begin(),
                       model_ranks_on_node.end(),
                       rank) != model_ranks_on_node.end();
    }

    /** Perform a sum reduction of mat over the inter-model communicator. */
    void intermodel_sum_matrix(Mat& mat);
    void intermodel_sum_matrix(DistMat& mat);
    /** Non-blocking intermodel_sum_matrix. */
    //void nb_intermodel_sum_matrix(Mat& mat, mpi::Request& req);
    //void nb_intermodel_sum_matrix(DistMat& mat, mpi::Request& req);
    /** Broadcast mat over the inter-model communicator starting from root. */
    void intermodel_broadcast_matrix(Mat& mat, int root);
    void intermodel_broadcast_matrix(DistMat& mat, int root);
    /** Non-blocking intermodel_broadcast_matrix. */
    //void nb_intermodel_broadcast_matrix(Mat& mat, int root, mpi::Request& req);
    //void nb_intermodel_broadcast_matrix(DistMat& mat, int root,
    //                                    mpi::Request& req);
    /**
     * Inter-model broadcast, returns the broadcast value.
     * Root process specifies root and val, other processes just root.
     */
    template <typename T>
    T intermodel_broadcast(int root, T val = {}) {
      mpi::Broadcast(&val, 1, root, intermodel_comm);
      if (get_rank_in_model() == root) {
        bytes_sent += sizeof(T);
      } else {
        bytes_received += sizeof(T);
      }
      return val;
    }
    /**
     * Within-model broadcast, returns the broadcast value.
     * Root process specifies root and val, other processes just root.
     */
    template <typename T>
    T model_broadcast(int root, T val = {}) {
      mpi::Broadcast(&val, 1, root, model_comm);
      if (get_rank_in_model() == root) {
        bytes_sent += sizeof(T);
      } else {
        bytes_received += sizeof(T);
      }
      return val;
    }
    /** Inter-model gather (for non-root processes). */
    template <typename T>
    void intermodel_gather(T send, int root) {
      bytes_sent += sizeof(T);
      mpi::Gather(&send, 1, (T*) NULL, 0, root, intermodel_comm);
    }
    /** Inter-model gather (for root processes). */
    template <typename T>
    void intermodel_gather(T send, std::vector<T>& recv) {
      mpi::Gather(&send, 1, recv.data(), 1, get_model_rank(),
                  intermodel_comm);
      bytes_received += sizeof(T) * (get_num_models() - 1);
    }
    /** Inter-model scalar-array gather (for non-root processes). */
    template <typename T>
    void intermodel_gather(T* send, int count, int root) {
      bytes_sent += sizeof(T) * count;
      mpi::Gather(send, count, (T*) NULL, 0, root, intermodel_comm);
    }
    /** Inter-model scalar-array gather (for root processes). */
    template <typename T>
    void intermodel_gather(T* send, int count, T* recv) {
      mpi::Gather(send, count, recv, count, get_model_rank(), intermodel_comm);
      bytes_received += sizeof(T) * count * (get_num_models() - 1);
    }
    /** Inter-model reduce (for non-root processes). */
    template <typename T>
    void intermodel_reduce(T send, int root, mpi::Op op = mpi::SUM) {
      bytes_sent += sizeof(T);
      mpi::Reduce(&send, (T*) NULL, 0, op, root, intermodel_comm);
    }
    /** Inter-model reduce (for root processes). */
    template <typename T>
    T intermodel_reduce(T send, mpi::Op op = mpi::SUM) {
      T val;
      mpi::Reduce(&send, &val, 1, op, get_model_rank(),
                  intermodel_comm);
      bytes_received += sizeof(T) * (get_num_models() - 1);
      return val;
    }
    /** Inter-model all-reduce. */
    template <typename T>
    T intermodel_allreduce(T send, mpi::Op op = mpi::SUM) {
      T val;
      bytes_sent += sizeof(T);
      mpi::AllReduce(&send, &val, 1, op, intermodel_comm);
      bytes_received += sizeof(T) * (get_num_models() - 1);
      return val;
    }
    /** Within-model reduce (for non-root processes). */
    template <typename T>
    void model_reduce(T send, int root, mpi::Op op = mpi::SUM) {
      bytes_sent += sizeof(T);
      mpi::Reduce(&send, (T*) NULL, 1, op, root, model_comm);
    }
    /** Within-model reduce (for root processes). */
    template <typename T>
    T model_reduce(T send, mpi::Op op = mpi::SUM) {
      T val;
      mpi::Reduce(&send, &val, 1, op, get_rank_in_model(), model_comm);
      bytes_received += sizeof(T) * (get_procs_per_model() - 1);
      return val;
    }
    /** Within-model scalar array reduce (for non-root processes). */
    template <typename T>
    void model_reduce(T* send, int count, int root, mpi::Op op = mpi::SUM) {
      bytes_sent += sizeof(T) * count;
      mpi::Reduce(send, (T*) NULL, count, op, root, model_comm);
    }
    /** Within-model scalar array reduce (for root processes). */
    template <typename T>
    void model_reduce(T* send, int count, T* recv, mpi::Op op = mpi::SUM) {
      mpi::Reduce(send, recv, count, op, get_rank_in_model(), model_comm);
      bytes_received += sizeof(T) * count * (get_procs_per_model() - 1);
    }
    /** Within-model all-reduce. */
    template <typename T>
    T model_allreduce(T send, mpi::Op op = mpi::SUM) {
      T val;
      bytes_sent += sizeof(T);
      mpi::AllReduce(&send, &val, 1, op, model_comm);
      bytes_received += sizeof(T) * (get_procs_per_model() - 1);
      return val;
    }
    /** Scalar array within-model all-reduce. */
    template <typename T>
    void model_allreduce(T* send, int count, T* recv, mpi::Op op = mpi::SUM) {
      bytes_sent += count * sizeof(T);
      mpi::AllReduce(send, recv, count, op, model_comm);
      bytes_received += count * sizeof(T) * (get_procs_per_model() - 1);
    }

    /** Wait for a non-blocking request to complete. */
    template <typename T>
    void wait(mpi::Request<T>& req) {
      mpi::Wait(req);
    }

    /** Barrier among the inter-model processes. */
    void intermodel_barrier();
    /** Barrier among processes in this model. */
    void model_barrier();
    /** Barrier among all processes. */
    void global_barrier();

    /** Send a buffer to rank in model. */
    template <typename T>
    void send(const T* data, int count, int model, int rank) {
      bytes_sent += sizeof(T) * count;
      mpi::Send(data, count, get_world_rank(model, rank), mpi::COMM_WORLD);
    }
    template <typename T> void send(const T* data, int count, int model) {
      send(data, count, model, rank_in_model);
    }
    void send(Mat& mat, int model, int rank);
    void send(DistMat& mat, int model, int rank);
    void send(Mat& mat, int model) { send(mat, model, rank_in_model); }
    void send(DistMat& mat, int model) { send(mat, model, rank_in_model); }

    /** Corresponding non-blocking sends. */
    template <typename T>
    void nb_send(const T* data, int count, int model, int rank,
                 mpi::Request<T>& req) {
      bytes_sent += sizeof(T) * count;
      mpi::ISend(data, count, get_world_rank(model, rank), mpi::COMM_WORLD, req);
    }
    template <typename T> void nb_send(const T* data, int count, int model,
                                       mpi::Request<T>& req) {
      nb_send(data, count, model, rank_in_model, req);
    }
    void nb_send(Mat& mat, int model, int rank, mpi::Request<DataType>& req);
    void nb_send(DistMat& mat, int model, int rank, mpi::Request<DataType>& req);
    void nb_send(Mat& mat, int model, mpi::Request<DataType>& req) {
      nb_send(mat, model, rank_in_model, req);
    }
    void nb_send(DistMat& mat, int model, mpi::Request<DataType>& req) {
      nb_send(mat, model, rank_in_model, req);
    }

    /** Corresponding receive to send. */
    template <typename T> void recv(T* data, int count, int model, int rank) {
      mpi::Recv(data, count, get_world_rank(model, rank), mpi::COMM_WORLD);
      bytes_received += sizeof(T) * count;
    }
    template <typename T> void recv(T* data, int count, int model) {
      recv(data, count, model, rank_in_model);
    }
    void recv(Mat& mat, int model, int rank);
    void recv(DistMat& mat, int model, int rank);
    void recv(Mat& mat, int model) { recv(mat, model, rank_in_model); }
    void recv(DistMat& mat, int model) { recv(mat, model, rank_in_model); }
    /** As above, but receive from anyone. */
    template <typename T> void recv(T* data, int count) {
      mpi::Recv(data, count, mpi::ANY_SOURCE, mpi::COMM_WORLD);
      bytes_received += sizeof(T) * count;
    }
    void recv(Mat& mat);
    void recv(DistMat& mat);

    /** Corresponding non-blocking receives. */
    template <typename T> void nb_recv(T* data, int count, int model, int rank,
                                       mpi::Request<T>& req) {
      mpi::IRecv(data, count, get_world_rank(model, rank), mpi::COMM_WORLD,
                 req);
      bytes_received += sizeof(T) * count;
    }
    template <typename T> void nb_recv(T* data, int count, int model,
                                       mpi::Request<T>& req) {
      nb_recv(data, count, model, rank_in_model, req);
    }
    void nb_recv(Mat& mat, int model, int rank, mpi::Request<DataType>& req);
    void nb_recv(DistMat& mat, int model, int rank, mpi::Request<DataType>& req);
    void nb_recv(Mat& mat, int model, mpi::Request<DataType>& req) {
      nb_recv(mat, model, rank_in_model, req);
    }
    void nb_recv(DistMat& mat, int model, mpi::Request<DataType>& req) {
      nb_recv(mat, model, rank_in_model, req);
    }
    template <typename T> void nb_recv(T* data, int count, mpi::Request<T>& req) {
      mpi::IRecv(data, count, mpi::ANY_SOURCE, mpi::COMM_WORLD, req);
      bytes_received += sizeof(T) * count;
    }
    void nb_recv(Mat& mat, mpi::Request<DataType>& req);
    void nb_recv(DistMat& mat, mpi::Request<DataType>& req);

    /** Send/recv to/from ranks. */
    template <typename T>
    void sendrecv(const T* send, int send_count, int send_model, int send_rank,
                  T* recv, int recv_count, int recv_model, int recv_rank) {
      bytes_sent += sizeof(T) * send_count;
      bytes_received += sizeof(T) * recv_count;
      mpi::SendRecv(send, send_count, get_world_rank(send_model, send_rank),
                    recv, recv_count, get_world_rank(recv_model, recv_rank),
                    mpi::COMM_WORLD);
    }
    template <typename T>
    void sendrecv(const T* send, int send_count, int send_model,
                  T* recv, int recv_count, int recv_model) {
      bytes_sent += sizeof(T) * send_count;
      bytes_received += sizeof(T) * recv_count;
      sendrecv(send, send_count, send_model, rank_in_model,
               recv, recv_count, recv_model, rank_in_model);
    }

    /** Determine the size (count) of an incoming message. */
    template <typename T> int get_count(int model, int rank) {
      MPI_Status status;
      MPI_Probe(get_world_rank(model, rank), MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      return mpi::GetCount<T>(status);
    }
    template <typename T>
    int get_count(int model) { return get_count<T>(model, rank_in_model); }

    /**
     * Broadcast data to the ranks in dests, beginning from root.
     * @todo Can probably optimize this.
     */
    template <typename T>
    void broadcast(T* data, int count, std::vector<int>& dests, int root) {
      mpi::Group bcast_group;
      mpi::Comm bcast_comm;
      std::vector<int> ranks;
      ranks.push_back(root);
      ranks.insert(ranks.end(), dests.begin(), dests.end());
      create_group(ranks, bcast_group);
      // Elemental doesn't expose this, so we have to reach into its internals.
      // This lets us create a communicator without involving all of COMM_WORLD.
      // Use a tag of 0; should not matter unless we're multi-threaded.
      MPI_Comm_create_group(mpi::COMM_WORLD.comm, bcast_group.group, 0,
                            &(bcast_comm.comm));
      int translated_root = mpi::Translate(mpi::COMM_WORLD, root, bcast_comm);
      mpi::Broadcast(data, count, translated_root, bcast_comm);
      mpi::Free(bcast_comm);
      mpi::Free(bcast_group);
    }
    void broadcast(Mat& mat, std::vector<int>& dests, int root);
    void broadcast(DistMat& mat, std::vector<int>& dests, int root);

    // Statistics methods.
    /** Return the number of model barriers performed. */
    inline size_t get_num_model_barriers() const { return num_model_barriers; }
    /** Return the number of inter-model barriers performed. */
    inline size_t get_num_intermodel_barriers() const { return num_intermodel_barriers; }
    /** Return the number of global barriers performed. */
    inline size_t get_num_global_barriers() const { return num_global_barriers; }
    /** Return the number of bytes sent. */
    inline size_t get_bytes_sent() const { return bytes_sent; }
    /** Return the number of bytes received. */
    inline size_t get_bytes_received() const { return bytes_received; }
    inline void reset_stats_counters() {
      num_model_barriers = 0;
      num_intermodel_barriers = 0;
      num_global_barriers = 0;
      bytes_sent = 0;
      bytes_received = 0;
    }

    /** Return true if mat can be transmitted. */
    static inline bool is_sendable(const Mat& mat) {
      // This assumes we do not transmit mat with a datatype smaller than
      // DataType.
      // MPI uses "int" as its count type; do calculations with larger ints.
      size_t count = (size_t) mat.Height() * (size_t) mat.Width();
      return count <= (size_t) std::numeric_limits<int>::max();
    }
    /** Return true if the local portion of dist_mat can be transmitted. */
    static inline bool is_sendable(const ElMat& dist_mat) {
      return is_sendable(dist_mat.LockedMatrix());
    }

    // Custom allreduce implementations.
    /**
     * Do a custom allreduce on mat on the intermodel communicator.
     * This selects the allreduce algorithm to use based on the size of mat.
     * @param mat The matrix to allreduce.
     * @param max_recv_count An upper bound on the size of data that will be
     * received in any step; this will be the size of receive buffers used in
     * the allreduce.
     * @param send_transform A function that takes a range of a matrix and
     * applies atransformation to it. The return value is a pointer to a buffer
     * containing the transformed data, which will be sent. The int param
     * should be filled in with the count of how many elements are in the
     * buffer. A boolean parameter indicates whether the matrix is constant
     * between different calls to send_transform; if true, the function may be
     * able to take advantage of this.
     * @param recv_transform A function that takes a pointer to a buffer and a
     * matrix and applies a transform to the buffer, storing the result in the
     * matrix. The buffer will be data transformed with send_transform and
     * received from another rank. The return value is the actual count of the
     * received data (i.e. the count that the data was sent using).
     * @param recv_apply_transform A function like recv_transform except that
     * the transformed data should be combined (applied, reduced) with the
     * current data in the matrix argument.
     */
    template <typename T>
    void intermodel_allreduce(
      Mat& mat, int max_recv_count,
      std::function<T*(Mat&, IR, IR, int&, bool)> send_transform,
      std::function<int(T*, Mat&)> recv_transform,
      std::function<int(T*, Mat&)> recv_apply_transform) {
      // If not a power-of-2, we can't use the recursive doubling.
      const int nprocs = get_num_models();
      if (nprocs & (nprocs - 1)) {
        pe_ring_allreduce(intermodel_comm, mat, max_recv_count,
                          send_transform, recv_transform,
                          recv_apply_transform);
      } else {
        // TODO: Don't hardcode this.
        if (mat.Height() <= 64 && mat.Width() <= 64) {
          recursive_doubling_allreduce_pow2(
            intermodel_comm, mat, max_recv_count,
            send_transform, recv_apply_transform);
        } else {
          pe_ring_allreduce(intermodel_comm, mat, max_recv_count,
                            send_transform, recv_transform,
                            recv_apply_transform);
        }
      }
    }

    /**
     * A recursive-doubling allreduce.
     * This implementation only works for a power-of-2 number of processes.
     */
    template <typename T>
    void recursive_doubling_allreduce_pow2(
      mpi::Comm comm, Mat& mat, int max_recv_count,
      std::function<T*(Mat&, IR, IR, int&, bool)> send_transform,
      std::function<int(T*, Mat&)> recv_apply_transform) {
      const int rank = mpi::Rank(comm);
      const int nprocs = mpi::Size(comm);
      // This implementation requires a power-of-2 number of processes.
      if (nprocs & (nprocs - 1)) {
        return;
      }
      T* recv_buf = (T*) get_collective_buffer(sizeof(T) * max_recv_count);
      unsigned int mask = 1;
      while (mask < nprocs) {
        int partner = rank ^ mask;  // The rank we exchange with this step.
        // Transform the data we want to send.
        int send_size;
        T* send_buf = send_transform(mat, ALL, ALL, send_size, false);
        bytes_sent += sizeof(T) * send_size;
        mpi::SendRecv(send_buf, send_size, partner,
                      recv_buf, max_recv_count, partner, comm);
        // Transform and reduce the received data.
        int recv_size = recv_apply_transform(recv_buf, mat);
        bytes_received += sizeof(T) * recv_size;
        mask <<= 1;
      }
    }

    /**
     * An allreduce based on a pairwise-exchange reduce-scatter followed by a
     * ring-based allgather.
     */
    template <typename T>
    void pe_ring_allreduce(
      mpi::Comm comm, Mat& mat, int max_recv_count,
      std::function<T*(Mat&, IR, IR, int&, bool)> send_transform,
      std::function<int(T*, Mat&)> recv_transform,
      std::function<int(T*, Mat&)> recv_apply_transform) {
      const int rank = mpi::Rank(comm);
      const int nprocs = mpi::Size(comm);
      // Compute the number of columns each processor sends.
      // If it doesn't divide evenly, give one extra to the earlier ranks.
      const Int cols_per_proc = mat.Width() / nprocs;
      const Int cols_remainder = mat.Width() % nprocs;
      // Compute the lengths/ends of the slices.
      std::vector<Int> slice_lengths(nprocs, cols_per_proc);
      for (int i = 0; i < cols_remainder; ++i) {
        slice_lengths[i] += 1;
      }
      std::vector<Int> slice_ends(nprocs);
      std::partial_sum(slice_lengths.begin(), slice_lengths.end(),
                       slice_ends.begin());
      T* recv_buf = (T*) get_collective_buffer(sizeof(T) * max_recv_count);
      // Local slice of our accumulated data.
      auto accum_view = mat(ALL, IR(slice_ends[rank] - slice_lengths[rank],
                                    slice_ends[rank]));
      // Do a pairwise-exchange reduce-scatter.
      for (int step = 1; step < nprocs; ++step) {
        // Compute where we send to/receive from.
        const int dst = (rank + step) % nprocs;
        const int src = (rank - step + nprocs) % nprocs;
        // Transform the data we send. We do not look at the same chunk of data
        // twice.
        int send_size;
        T* send_buf = send_transform(
          mat, ALL, IR(slice_ends[dst] - slice_lengths[dst], slice_ends[dst]),
          send_size, true);
        bytes_sent += sizeof(T) * send_size;
        mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, max_recv_count, src, comm);
        int recv_size = recv_apply_transform(recv_buf, accum_view);
        bytes_received += sizeof(T) * recv_size;
      }
      // Do a ring allgather.
      const int src = (rank - 1 + nprocs) % nprocs;
      const int dst = (rank + 1) % nprocs;
      // Apply the transform to our locally-accumulated slice of the data.
      int send_size;
      T* send_buf = send_transform(
        mat, ALL, IR(slice_ends[rank] - slice_lengths[rank], slice_ends[rank]),
        send_size, false);
      // Do the first step where we forward our local data.
      {
        const int data_src = (rank - 1 + nprocs) % nprocs;
        bytes_sent += sizeof(T) * send_size;
        mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, max_recv_count, src, comm);
        auto recv_view = mat(ALL,
                             IR(slice_ends[data_src] - slice_lengths[data_src],
                                slice_ends[data_src]));
        int recv_size = recv_transform(recv_buf, recv_view);
        bytes_received += sizeof(T) * recv_size;
        send_size = recv_size;
      }
      // Now do the remaining nprocs - 2 steps.
      // We always send from recv_buf and receive to recv_buf2, swapping
      // pointers to avoid copying.
      T* recv_buf2 = (T*) get_collective_buffer(sizeof(T) * max_recv_count, 1);
      for (int step = 1; step < nprocs - 1; ++step) {
        // Compute where the data we get is coming from.
        const int data_src = (rank - step - 1 + nprocs) % nprocs;
        auto recv_view = mat(ALL,
                             IR(slice_ends[data_src] - slice_lengths[data_src],
                                slice_ends[data_src]));
        bytes_sent += sizeof(T) * send_size;
        mpi::SendRecv(recv_buf, send_size, dst,
                      recv_buf2, max_recv_count, src, comm);
        int recv_size = recv_transform(recv_buf2, recv_view);
        bytes_received += sizeof(T) * recv_size;
        // Swap the send and receive buffers.
        std::swap(recv_buf, recv_buf2);
        send_size = recv_size;
      }
    }

    /**
     * An allreduce using ring-based reduce-scatter and allgather.
     */
    template <typename T>
    void ring_allreduce(
      mpi::Comm comm, Mat& mat, int max_recv_count,
      std::function<T*(Mat&, IR, IR, int&, bool)> send_transform,
      std::function<int(T*, Mat&)> recv_transform,
      std::function<int(T*, Mat&)> recv_apply_transform) {
      const int rank = mpi::Rank(comm);
      const int nprocs = mpi::Size(comm);
      // Compute the number of columns each processor sends.
      const Int cols_per_proc = mat.Width() / nprocs;
      const Int cols_remainder = mat.Width() % nprocs;
      // Compute the lengths/ends of the slices.
      std::vector<Int> slice_lengths(nprocs, cols_per_proc);
      for (int i = 0; i < cols_remainder; ++i) {
        slice_lengths[i] += 1;
      }
      std::vector<Int> slice_ends(nprocs);
      std::partial_sum(slice_lengths.begin(), slice_lengths.end(),
                       slice_ends.begin());
      T* recv_buf = (T*) get_collective_buffer(sizeof(T) * max_recv_count);
      // Compute source/destination in the ring.
      const int src = (rank - 1 + nprocs) % nprocs;
      const int dst = (rank + 1) % nprocs;
      // Do a ring-based reduce-scatter.
      // This is like the pairwise-exchange reduce-scatter except instead of
      // rank i accumulating only slice i, the slices are cycled around and
      // each node accumulates its portion into the slice when it passes
      // through. After the nprocs-1 steps slice k will be on rank
      // (k + nprocs - 1) % nprocs.
      for (int step = 0; step < nprocs - 1; ++step) {
        // Compute the slices to send/recv.
        const int send_slice = (rank - step + nprocs) % nprocs;
        const int recv_slice = (rank - step - 1 + nprocs) % nprocs;
        // Transform the data to send.
        int send_size;
        T* send_buf = send_transform(
          mat, ALL, IR(slice_ends[send_slice] - slice_lengths[send_slice],
                       slice_ends[send_slice]), send_size, false);
        mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, max_recv_count, src, comm);
        auto recv_view = mat(
          ALL, IR(slice_ends[recv_slice] - slice_lengths[recv_slice],
                  slice_ends[recv_slice]));
        int recv_size = recv_apply_transform(recv_buf, recv_view);
      }
      // Do a ring allgather, first applying the transform to local data.
      int send_size;
      {
        const int send_slice = (rank + 1) % nprocs;
        const int recv_slice = rank;
        T* send_buf = send_transform(
          mat, ALL, IR(slice_ends[send_slice] - slice_lengths[send_slice],
                       slice_ends[send_slice]), send_size, false);
        mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, max_recv_count, src, comm);
        auto recv_view = mat(ALL,
                             IR(slice_ends[recv_slice] - slice_lengths[recv_slice],
                                slice_ends[recv_slice]));
        int recv_size = recv_transform(recv_buf, recv_view);
        send_size = recv_size;
      }
      T* recv_buf2 = (T*) get_collective_buffer(sizeof(T) * max_recv_count, 1);
      for (int step = 1; step < nprocs - 1; ++step) {
        const int send_slice = (rank - step + 1 + nprocs) % nprocs;
        const int recv_slice = (rank - step + nprocs) % nprocs;
        auto recv_view = mat(ALL,
                             IR(slice_ends[recv_slice] - slice_lengths[recv_slice],
                                slice_ends[recv_slice]));
        mpi::SendRecv(recv_buf, send_size, dst,
                      recv_buf2, max_recv_count, src, comm);
        int recv_size = recv_transform(recv_buf2, recv_view);
        // Swap the send and receive buffers.
        std::swap(recv_buf, recv_buf2);
        send_size = recv_size;
      }
    }

    /** Return the intermodel communicator. */
    mpi::Comm get_intermodel_comm() const { return intermodel_comm; }

  private:
    /** Communicator for every process in this model. */
    mpi::Comm model_comm;
    /** Communicator for every process with the same model rank. */
    mpi::Comm intermodel_comm;
    /** Communicator for every process in the same compute node. */
    mpi::Comm node_comm;
    /** Grid for this model. */
    Grid* grid;
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
    /** The list of ranks in the model_comm that are on this compute node. */
    std::vector<int> model_ranks_on_node;
    /** Pre-allocated buffers for collectives. */
    std::unordered_map<size_t, std::vector<uint8_t*>> collective_bufs;
    
    // Various statistics counters.
    size_t num_model_barriers;
    size_t num_intermodel_barriers;
    size_t num_global_barriers;
    size_t bytes_sent;
    size_t bytes_received;

    /** MPI tag for point-to-point communication. (Unused) */
    static const int PT2PT_TAG = 42;
    /** Create a new group from a list of ranks. (Needs to be freed.) */
    inline void create_group(std::vector<int>& ranks, mpi::Group& g) {
      mpi::Group world_group;
      mpi::CommGroup(mpi::COMM_WORLD, world_group);
      mpi::Incl(world_group, (int) ranks.size(), ranks.data(), g);
    }

    /** Setup communicator for processes in the same compute node.
     *  We obtain a string specifying the compute node. The string is
     *  hashed (with salt) and used to split the communicators. To
     *  avoid hash collisions, the splitting procedure is repeated
     *  with a different salt. */
    void setup_node_comm();

    /**
     * Return a buffer from collective_bufs, allocating it if needed.
     * @param size The size of the buffer (in bytes).
     * @param idx The index of the buffer (default 0).
     */
    uint8_t* get_collective_buffer(size_t size, size_t idx = 0);
    
  };
}

#endif  // LBANN_COMM_HPP_INCLUDED
