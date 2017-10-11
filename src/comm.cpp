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

#include "lbann/comm.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/exception.hpp"
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

lbann_comm::lbann_comm(int ppm) :
  grid(nullptr), procs_per_model(ppm), num_model_barriers(0),
  num_intermodel_barriers(0), num_global_barriers(0), bytes_sent(0),
  bytes_received(0) {

  // Set up the initial model split
  split_models(procs_per_model);

  // Initialize node communicators
  setup_node_comm();
  procs_per_node = El::mpi::Size(node_comm);
  rank_in_node = El::mpi::Rank(node_comm);

  // Setup threads
  setup_threads();
  
}

lbann_comm::~lbann_comm() {
  delete grid;
  El::mpi::Free(model_comm);
  El::mpi::Free(intermodel_comm);
  El::mpi::Free(node_comm);
  for (auto&& buf_vec : collective_bufs) {
    for (auto&& buf : buf_vec.second) {
      delete[] buf;
    }
  }
}

void lbann_comm::split_models(int ppm) {
  int world_size = El::mpi::Size(El::mpi::COMM_WORLD);
  procs_per_model = ppm;
  if (ppm == 0) {
    procs_per_model = world_size;
  }
  // Check if parameters are valid
  if (procs_per_model > world_size) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: Not enough processes to create one model; procs_per_model: " +
      std::to_string(procs_per_model) + " is larger than world_size: " +
      std::to_string(world_size));
  }
  if (world_size % procs_per_model != 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: Procs per model does not divide total number of procs; procs_per_model: " +
      std::to_string(procs_per_model) + " total number of procs (world size): " +
      std::to_string(world_size));
  }

  num_models = world_size / procs_per_model;
  model_rank = El::mpi::Rank(El::mpi::COMM_WORLD) / procs_per_model;
  rank_in_model = El::mpi::Rank(El::mpi::COMM_WORLD) % procs_per_model;

  // Initialize model and intermodel communicators
  El::mpi::Split(El::mpi::COMM_WORLD, model_rank, rank_in_model, model_comm);
  El::mpi::Split(El::mpi::COMM_WORLD, rank_in_model, model_rank,
                 intermodel_comm);

  // Initialize Elemental grid
  if (grid != nullptr) {
    delete grid;
  }
  grid = new Grid(model_comm);
}

void lbann_comm::intermodel_sum_matrix(Mat& mat) {
  bytes_sent += sizeof(DataType) * mat.Height() * mat.Width();
  AllReduce(mat, intermodel_comm, El::mpi::SUM);
  bytes_received += sizeof(DataType) * mat.Height() * mat.Width();
}

void lbann_comm::intermodel_sum_matrix(DistMat& mat) {
  bytes_sent += sizeof(DataType) * mat.LocalHeight() * mat.LocalWidth();
  AllReduce(mat, intermodel_comm, El::mpi::SUM);
  bytes_received += sizeof(DataType) * mat.LocalHeight() * mat.LocalWidth();
}

void lbann_comm::intermodel_broadcast_matrix(Mat& mat, int root) {
  Broadcast(mat, intermodel_comm, root);
}

void lbann_comm::intermodel_broadcast_matrix(DistMat& mat, int root) {
  Broadcast(mat, intermodel_comm, root);
}

void lbann_comm::intermodel_barrier() {
  ++num_intermodel_barriers;
  El::mpi::Barrier(intermodel_comm);
}

void lbann_comm::model_barrier() {
  ++num_model_barriers;
  El::mpi::Barrier(model_comm);
}

void lbann_comm::global_barrier() {
  ++num_global_barriers;
  El::mpi::Barrier(El::mpi::COMM_WORLD);
}

void lbann_comm::send(const Mat& mat, int model, int rank) {
  send(mat.LockedBuffer(), mat.Height() * mat.Width(), model, rank);
}

void lbann_comm::send(const DistMat& mat, int model, int rank) {
  send(mat.LockedBuffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank);
}

void lbann_comm::nb_send(const Mat& mat, int model, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_send(mat.LockedBuffer(), mat.Height() * mat.Width(), model, rank, req);
}

void lbann_comm::nb_send(const DistMat& mat, int model, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_send(mat.LockedBuffer(), mat.LocalHeight() * mat.LocalWidth(), model,
          rank, req);
}

void lbann_comm::recv(Mat& mat, int model, int rank) {
  recv(mat.Buffer(), mat.Height() * mat.Width(), model, rank);
}

void lbann_comm::recv(DistMat& mat, int model, int rank) {
  recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank);
}

void lbann_comm::recv(Mat& mat) {
  recv(mat.Buffer(), mat.Height() * mat.Width());
}

void lbann_comm::recv(DistMat& mat) {
  recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth());
}

void lbann_comm::nb_recv(Mat& mat, int model, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), model, rank, req);
}

void lbann_comm::nb_recv(DistMat& mat, int model, int rank,
                         El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank, req);
}

void lbann_comm::nb_recv(Mat& mat, El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), req);
}

void lbann_comm::nb_recv(DistMat& mat, El::mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), req);
}

void lbann_comm::intermodel_allreduce(
  Mat& mat, int max_recv_count,
  std::function<uint8_t *(Mat&, El::IR, El::IR, int&, bool, int)> send_transform,
  std::function<int(uint8_t *, Mat&)> recv_transform,
  std::function<int(uint8_t *, Mat&, bool)> recv_apply_transform,
  const lbann_comm::allreduce_options opts) {
  // Determine which algorithm to actually use.
  lbann_comm::allreduce_algorithm algo = opts.algo;
  if (algo == allreduce_algorithm::DEFAULT) {
    algo = get_default_allreduce_algorithm();
  }
  const int nprocs = get_num_models();
  const Int small_message_threshold = 64*64;
  if (algo == allreduce_algorithm::DYNAMIC) {
    // For small messages and power-of-2 number of processes, use RD.
    if (!(nprocs & (nprocs - 1)) &&
        mat.Height() * mat.Width() <= small_message_threshold) {
      algo = allreduce_algorithm::RECURSIVE_DOUBLING;
    } else {
      algo = allreduce_algorithm::PAIRWISE_EXCHANGE_RING;
    }
  }
  switch (algo) {
  case allreduce_algorithm::RECURSIVE_DOUBLING:
    recursive_doubling_allreduce_pow2(
      intermodel_comm, mat, max_recv_count,
      send_transform, recv_apply_transform, opts);
    break;
  case allreduce_algorithm::PAIRWISE_EXCHANGE_RING:
    pe_ring_allreduce(
      intermodel_comm, mat, max_recv_count, send_transform,
      recv_transform, recv_apply_transform, opts);
    break;
  case allreduce_algorithm::RING:
    ring_allreduce(
      intermodel_comm, mat, max_recv_count, send_transform,
      recv_transform, recv_apply_transform, opts);
    break;
  case allreduce_algorithm::RABENSEIFNER:
    rabenseifner_allreduce(
      intermodel_comm, mat, max_recv_count, send_transform,
      recv_transform, recv_apply_transform, opts);
    break;
  case allreduce_algorithm::DEFAULT:
  case allreduce_algorithm::DYNAMIC:
  default:
    throw lbann_exception("intermodel_allreduce: bad algorithm");
    break;
  }
}

void lbann_comm::recursive_doubling_allreduce_pow2(
  El::mpi::Comm comm, Mat& mat, int max_recv_count,
  std::function<uint8_t *(Mat&, El::IR, El::IR, int&, bool, int)> send_transform,
  std::function<int(uint8_t *, Mat&, bool)> recv_apply_transform,
  const lbann_comm::allreduce_options opts) {
  double ar_start = get_time();
  const int rank = El::mpi::Rank(comm);
  const unsigned int nprocs = El::mpi::Size(comm);
  if (nprocs == 1) {
    return;  // Nothing to do.
  }
  // This implementation requires a power-of-2 number of processes.
  if (nprocs & (nprocs - 1)) {
    throw lbann_exception("lbann_comm: recursive doubling allreduce requires"
                          " a power-of-2 number of participating processes");
  }
  uint8_t *max_recv_buf = get_collective_buffer(max_recv_count);
  uint8_t *recv_buf = max_recv_buf;
  unsigned int mask = 1;
  while (mask < nprocs) {
    int partner = rank ^ mask;  // The rank we exchange with this step.
    const bool is_local = opts.no_local_trans &&
                          is_rank_node_local(partner, comm);
    // Transform the data we want to send.
    double send_trans_start = get_time();
    int send_size;
    int recv_size = max_recv_count;
    uint8_t *send_buf = nullptr;
    if (is_local) {
      send_buf = (uint8_t *) mat.Buffer();
      send_size = sizeof(DataType) * mat.Height() * mat.Width();
      recv_size = send_size;
      recv_buf = get_collective_buffer(recv_size);
    } else {
      send_buf = send_transform(mat, El::ALL, El::ALL, send_size, false, 0);
      recv_buf = max_recv_buf;
    }
    ar_send_transform_time += get_time() - send_trans_start;
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    double sendrecv_start = get_time();
    El::mpi::SendRecv(send_buf, send_size, partner,
                      recv_buf, recv_size, partner, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    // Transform and reduce the received data.
    double recv_apply_trans_start = get_time();
    recv_size = recv_apply_transform(recv_buf, mat, is_local);
    ar_recv_apply_transform_time += get_time() - recv_apply_trans_start;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    mask <<= 1;
  }
  ar_time += get_time() - ar_start;
}

void lbann_comm::pe_ring_allreduce(
  El::mpi::Comm comm, Mat& mat, int max_recv_count,
  std::function<uint8_t *(Mat&, El::IR, El::IR, int&, bool, int)> send_transform,
  std::function<int(uint8_t *, Mat&)> recv_transform,
  std::function<int(uint8_t *, Mat&, bool)> recv_apply_transform,
  const lbann_comm::allreduce_options opts) {
  double ar_start = get_time();
  const int rank = El::mpi::Rank(comm);
  const int nprocs = El::mpi::Size(comm);
  if (nprocs == 1) {
    return;  // Nothing to do.
  }
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
  std::vector<uint8_t *> max_recv_buffers(opts.max_reduces, nullptr);
  for (size_t i = 0; i < max_recv_buffers.size(); ++i) {
    max_recv_buffers[i] = get_collective_buffer(max_recv_count, i);
  }
  // Local slice of our accumulated data.
  auto accum_view = mat(El::ALL, El::IR(slice_ends[rank] - slice_lengths[rank],
                                    slice_ends[rank]));
  // Do a pairwise-exchange reduce-scatter.
  double rs_start = get_time();
  for (int outer_step = 1;
       outer_step < nprocs;
       outer_step += opts.max_reduces) {
    const int reduces_this_step = std::min(opts.max_reduces,
                                           nprocs - outer_step);
    std::vector<El::mpi::Request<uint8_t>> send_reqs(reduces_this_step);
    std::vector<El::mpi::Request<uint8_t>> recv_reqs(reduces_this_step);
    std::vector<uint8_t *> recv_buffers(max_recv_buffers);
    int num_local_recvs = 0;
    std::vector<bool> local_recvs(reduces_this_step, false);
    for (int step = outer_step; step < outer_step + reduces_this_step; ++step) {
      const int reduce_idx = step - outer_step;
      // Compute where we send to/receive from.
      const int dst = (rank + step) % nprocs;
      const int src = (rank - step + nprocs) % nprocs;
      const bool is_send_local = opts.no_local_trans &&
                                 is_rank_node_local(dst, comm);
      const bool is_recv_local = opts.no_local_trans &&
                                 is_rank_node_local(src, comm);
      // Post the receive.
      double recv_start = get_time();
      int recv_size = max_recv_count;
      if (is_recv_local) {
        recv_size = sizeof(DataType) * accum_view.Height() * accum_view.Width();
        recv_buffers[reduce_idx] = get_collective_buffer(recv_size,
                                   num_local_recvs);
        ++num_local_recvs;
        local_recvs[reduce_idx] = is_recv_local;
      }
      El::mpi::IRecv(recv_buffers[reduce_idx], recv_size, src, comm,
                     recv_reqs[reduce_idx]);
      double recv_tot = get_time() - recv_start;
      ar_recv_time += recv_tot;
      ar_rs_recv_time += recv_tot;
      // Transform the data we send. We do not look at the same chunk of data
      // twice.
      double send_trans_start = get_time();
      int send_size;
      uint8_t *send_buf = nullptr;
      if (is_send_local) {
        auto send_view = mat(El::ALL,
                             El::IR(slice_ends[dst] - slice_lengths[dst], slice_ends[dst]));
        send_buf = (uint8_t *) send_view.Buffer();
        send_size = sizeof(DataType) * send_view.Height() * send_view.Width();
      } else {
        send_buf = send_transform(
                     mat, El::ALL, El::IR(slice_ends[dst] - slice_lengths[dst], slice_ends[dst]),
                     send_size, true, reduce_idx);
      }
      ar_send_transform_time += get_time() - send_trans_start;
      bytes_sent += send_size;
      ar_bytes_sent += send_size;
      ar_rs_bytes_sent += send_size;
      // Post the send.
      double send_start = get_time();
      El::mpi::ISend(send_buf, send_size, dst, comm, send_reqs[reduce_idx]);
      double send_tot = get_time() - send_start;
      ar_send_time += send_tot;
      ar_rs_send_time += send_tot;
    }
    // Complete the receives (in any order).
    // We need to extract the raw MPI_Request because Elemental does not support
    // MPI_Waitany.
    std::vector<MPI_Request> raw_reqs(reduces_this_step);
    for (int i = 0; i < reduces_this_step; ++i) {
      raw_reqs[i] = recv_reqs[i].backend;
    }
    for (int i = 0; i < reduces_this_step; ++i) {
      int completed_idx;
      double recv_start = get_time();
      MPI_Waitany(reduces_this_step, raw_reqs.data(), &completed_idx,
                  MPI_STATUS_IGNORE);
      double recv_tot = get_time() - recv_start;
      ar_recv_time += recv_tot;
      ar_rs_recv_time += recv_tot;
      double recv_apply_trans_start = get_time();
      int recv_size = recv_apply_transform(
                        recv_buffers[completed_idx], accum_view, local_recvs[completed_idx]);
      ar_recv_apply_transform_time += get_time() - recv_apply_trans_start;
      bytes_received += recv_size;
      ar_bytes_received += recv_size;
      ar_rs_bytes_received += recv_size;
    }
    // Complete all the sends.
    double send_start = get_time();
    El::mpi::WaitAll(reduces_this_step, send_reqs.data(), MPI_STATUSES_IGNORE);
    double send_tot = get_time() - send_start;
    ar_send_time += send_tot;
    ar_rs_send_time += send_tot;
  }
  uint8_t *recv_buf = max_recv_buffers[0];  // Get a regular recv buffer.
  ar_rs_time += get_time() - rs_start;
  // Do a ring allgather.
  double ag_start = get_time();
  const int src = (rank - 1 + nprocs) % nprocs;
  const int dst = (rank + 1) % nprocs;
  // Apply the transform to our locally-accumulated slice of the data.
  // Since the same data is cycled to every process, we do not do the
  // no_local_trans here.
  int send_size;
  // Do the first step where we forward our local data.
  {
    double send_trans_start = get_time();
    uint8_t *send_buf = send_transform(
                          mat, El::ALL, El::IR(slice_ends[rank] - slice_lengths[rank], slice_ends[rank]),
                          send_size, false, 0);
    ar_send_transform_time += get_time() - send_trans_start;
    const int data_src = (rank - 1 + nprocs) % nprocs;
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_ag_bytes_sent += send_size;
    auto recv_view = mat(El::ALL,
                         El::IR(slice_ends[data_src] - slice_lengths[data_src],
                                slice_ends[data_src]));
    // If we can, receive directly into the destination matrix.
    if (opts.id_recv) {
      recv_buf = (uint8_t *) recv_view.Buffer();
      max_recv_count = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    }
    double sendrecv_start = get_time();
    El::mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, max_recv_count, src, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_ag_send_time += sendrecv_tot;
    ar_ag_recv_time += sendrecv_tot;
    double recv_trans_start = get_time();
    int recv_size = 0;
    if (opts.id_recv) {
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      recv_size = recv_transform(recv_buf, recv_view);
    }
    ar_recv_transform_time += get_time() - recv_trans_start;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    ar_ag_bytes_received += send_size;
    send_size = recv_size;
  }
  // Now do the remaining nprocs - 2 steps.
  // We always send from recv_buf and receive to recv_buf2, swapping
  // pointers to avoid copying.
  uint8_t *recv_buf2 = nullptr;
  if (!opts.id_recv) {
    recv_buf2 = get_collective_buffer(max_recv_count, 1);
  }
  for (int step = 1; step < nprocs - 1; ++step) {
    // Compute where the data we get is coming from.
    const int data_src = (rank - step - 1 + nprocs) % nprocs;
    auto recv_view = mat(El::ALL,
                         El::IR(slice_ends[data_src] - slice_lengths[data_src],
                                slice_ends[data_src]));
    if (opts.id_recv) {
      recv_buf2 = (uint8_t *) recv_view.Buffer();
      max_recv_count = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    }
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_ag_bytes_sent += send_size;
    double sendrecv_start = get_time();
    El::mpi::SendRecv(recv_buf, send_size, dst,
                      recv_buf2, max_recv_count, src, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_ag_send_time += sendrecv_tot;
    ar_ag_recv_time += sendrecv_tot;
    double recv_trans_start = get_time();
    int recv_size = 0;
    if (opts.id_recv) {
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      recv_size = recv_transform(recv_buf2, recv_view);
    }
    ar_recv_transform_time += get_time() - recv_trans_start;
    bytes_received += recv_size;
    // Swap the send and receive buffers.
    std::swap(recv_buf, recv_buf2);
    send_size = recv_size;
    ar_bytes_received += recv_size;
    ar_ag_bytes_received += send_size;
  }
  ar_ag_time += get_time() - ag_start;
  ar_time += get_time() - ar_start;
}

void lbann_comm::ring_allreduce(
  El::mpi::Comm comm, Mat& mat, int max_recv_count,
  std::function<uint8_t *(Mat&, El::IR, El::IR, int&, bool, int)> send_transform,
  std::function<int(uint8_t *, Mat&)> recv_transform,
  std::function<int(uint8_t *, Mat&, bool)> recv_apply_transform,
  const lbann_comm::allreduce_options opts) {
  double ar_start = get_time();
  const int rank = El::mpi::Rank(comm);
  const int nprocs = El::mpi::Size(comm);
  if (nprocs == 1) {
    return;  // Nothing to do.
  }
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
  uint8_t *max_recv_buf = get_collective_buffer(max_recv_count);
  uint8_t *recv_buf = max_recv_buf;
  // Compute source/destination in the ring.
  const int src = (rank - 1 + nprocs) % nprocs;
  const int dst = (rank + 1) % nprocs;
  const bool is_send_local = opts.no_local_trans &&
                             is_rank_node_local(dst, comm);
  const bool is_recv_local = opts.no_local_trans &&
                             is_rank_node_local(src, comm);
  // Do a ring-based reduce-scatter.
  // This is like the pairwise-exchange reduce-scatter except instead of
  // rank i accumulating only slice i, the slices are cycled around and
  // each node accumulates its portion into the slice when it passes
  // through. After the nprocs-1 steps slice k will be on rank
  // (k + nprocs - 1) % nprocs.
  double rs_start = get_time();
  for (int step = 0; step < nprocs - 1; ++step) {
    // Compute the slices to send/recv.
    const int send_slice = (rank - step + nprocs) % nprocs;
    const int recv_slice = (rank - step - 1 + nprocs) % nprocs;
    // Transform the data to send.
    double send_trans_start = get_time();
    int send_size;
    int recv_size = max_recv_count;
    uint8_t *send_buf = nullptr;
    if (is_send_local) {
      auto send_view = mat(El::ALL,
                           El::IR(slice_ends[dst] - slice_lengths[dst], slice_ends[dst]));
      send_buf = (uint8_t *) send_view.Buffer();
      send_size = sizeof(DataType) * send_view.Height() * send_view.Width();
    } else {
      send_buf = send_transform(mat, El::ALL,
                                     El::IR(slice_ends[send_slice] - slice_lengths[send_slice],
                                            slice_ends[send_slice]), send_size, false, 0);
    }
    auto recv_view = mat(El::ALL,
                         El::IR(slice_ends[recv_slice] - slice_lengths[recv_slice], slice_ends[recv_slice]));
    if (is_recv_local) {
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
      recv_buf = get_collective_buffer(recv_size);
    } else {
      recv_buf = max_recv_buf;
    }
    ar_send_transform_time += get_time() - send_trans_start;
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_rs_bytes_sent += send_size;
    double sendrecv_start = get_time();
    El::mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, recv_size, src, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_rs_send_time += sendrecv_tot;
    ar_rs_recv_time += sendrecv_tot;
    double recv_apply_trans_start = get_time();
    recv_size = recv_apply_transform(recv_buf, recv_view, is_recv_local);
    ar_recv_apply_transform_time += get_time() - recv_apply_trans_start;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    ar_rs_bytes_received += recv_size;
  }
  recv_buf = max_recv_buf;  // Ensure we're back to the original.
  ar_rs_time += get_time() - rs_start;
  // Do a ring allgather, first applying the transform to local data.
  double ag_start = get_time();
  int send_size;
  {
    const int send_slice = (rank + 1) % nprocs;
    const int recv_slice = rank;
    double send_trans_start = get_time();
    uint8_t *send_buf = send_transform(
                          mat, El::ALL, El::IR(slice_ends[send_slice] - slice_lengths[send_slice],
                                           slice_ends[send_slice]), send_size, false, 0);
    ar_send_transform_time += get_time() - send_trans_start;
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_ag_bytes_sent += send_size;
    auto recv_view = mat(El::ALL,
                         El::IR(slice_ends[recv_slice] - slice_lengths[recv_slice],
                                slice_ends[recv_slice]));
    // If we can, receive directly into the destination matrix.
    if (opts.id_recv) {
      recv_buf = (uint8_t *) recv_view.Buffer();
      max_recv_count = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    }
    double sendrecv_start = get_time();
    El::mpi::SendRecv(send_buf, send_size, dst,
                      recv_buf, max_recv_count, src, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_ag_send_time += sendrecv_tot;
    ar_ag_recv_time += sendrecv_tot;
    double recv_trans_start = get_time();
    int recv_size = 0;
    if (opts.id_recv) {
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      recv_size = recv_transform(recv_buf, recv_view);
    }
    ar_recv_transform_time += get_time() - recv_trans_start;
    send_size = recv_size;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    ar_ag_bytes_received += recv_size;
  }
  uint8_t *recv_buf2 = nullptr;
  if (!opts.id_recv) {
    recv_buf2 = get_collective_buffer(max_recv_count, 1);
  }
  for (int step = 1; step < nprocs - 1; ++step) {
    const int recv_slice = (rank - step + nprocs) % nprocs;
    auto recv_view = mat(El::ALL,
                         El::IR(slice_ends[recv_slice] - slice_lengths[recv_slice],
                                slice_ends[recv_slice]));
    if (opts.id_recv) {
      recv_buf2 = (uint8_t *) recv_view.Buffer();
      max_recv_count = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    }
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_ag_bytes_sent += send_size;
    double sendrecv_start = get_time();
    El::mpi::SendRecv(recv_buf, send_size, dst,
                      recv_buf2, max_recv_count, src, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_ag_send_time += sendrecv_tot;
    ar_ag_recv_time += sendrecv_tot;
    double recv_trans_start = get_time();
    int recv_size = 0;
    if (opts.id_recv) {
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      recv_size = recv_transform(recv_buf2, recv_view);
    }
    ar_recv_transform_time += get_time() - recv_trans_start;
    // Swap the send and receive buffers.
    std::swap(recv_buf, recv_buf2);
    send_size = recv_size;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    ar_ag_bytes_received += recv_size;
  }
  ar_ag_time += get_time() - ag_start;
  ar_time += get_time() - ar_start;
}

void lbann_comm::rabenseifner_allreduce(
  El::mpi::Comm comm, Mat& mat, int max_recv_count,
  std::function<uint8_t *(Mat&, El::IR, El::IR, int&, bool, int)> send_transform,
  std::function<int(uint8_t *, Mat&)> recv_transform,
  std::function<int(uint8_t *, Mat&, bool)> recv_apply_transform,
  const lbann_comm::allreduce_options opts) {
  double ar_start = get_time();
  const int rank = El::mpi::Rank(comm);
  const unsigned int nprocs = El::mpi::Size(comm);
  if (nprocs == 1) {
    return;  // Nothing to do.
  }
  // This implementation requires a power-of-2 number of processes.
  if (nprocs & (nprocs - 1)) {
    throw lbann_exception("lbann_comm: Rabenseifner allreduce requires"
                          " a power-of-2 number of participating processes");
  }
  // Compute the slices on each processor.
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
  // Do a recursive-halving reduce-scatter.
  // In each step here a process sends all the data needed for the other
  // "half" of the processes. i.e. each process sends half their data in the
  // first step, a quarter in the second step, etc.
  double rs_start = get_time();
  unsigned int partner_mask = nprocs >> 1;
  unsigned int slice_mask = 1;
  unsigned int send_idx = 0;
  unsigned int recv_idx = 0;
  unsigned int last_idx = nprocs;
  uint8_t *recv_buf = get_collective_buffer(max_recv_count);
  while (partner_mask > 0) {
    int partner = rank ^ partner_mask;  // The rank we exchange with this step.
    const bool is_local = opts.no_local_trans &&
                          is_rank_node_local(partner, comm);
    // Determine the range of data to send/recv.
    El::IR send_range, recv_range;
    if (rank < partner) {
      send_idx = recv_idx + nprocs / (slice_mask*2);
      send_range = El::IR(slice_ends[send_idx] - slice_lengths[send_idx],
                          slice_ends[last_idx-1]);
      recv_range = El::IR(slice_ends[recv_idx] - slice_lengths[recv_idx],
                          slice_ends[send_idx-1]);
    } else {
      recv_idx = send_idx + nprocs / (slice_mask*2);
      send_range = El::IR(slice_ends[send_idx] - slice_lengths[send_idx],
                          slice_ends[recv_idx-1]);
      recv_range = El::IR(slice_ends[recv_idx] - slice_lengths[recv_idx],
                          slice_ends[last_idx-1]);
    }
    auto recv_view = mat(El::ALL, recv_range);
    // Transform the data to send.
    double send_trans_start = get_time();
    int send_size;
    int recv_size = max_recv_count;
    uint8_t *send_buf = nullptr;
    if (is_local) {
      auto send_view = mat(El::ALL, send_range);
      send_buf = (uint8_t *) send_view.Buffer();
      send_size = sizeof(DataType) * send_view.Height() * send_view.Width();
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      send_buf = send_transform(mat, El::ALL, send_range, send_size, false, 0);
    }
    ar_send_transform_time += get_time() - send_trans_start;
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_rs_bytes_sent += send_size;
    double sendrecv_start = get_time();
    El::mpi::SendRecv(send_buf, send_size, partner,
                      recv_buf, recv_size, partner, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_rs_send_time += sendrecv_tot;
    ar_rs_recv_time += sendrecv_tot;
    // Transform the received data.
    double recv_apply_trans_start = get_time();
    recv_size = recv_apply_transform(recv_buf, recv_view, is_local);
    ar_recv_apply_transform_time += get_time() - recv_apply_trans_start;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    ar_rs_bytes_received += send_size;
    // Update info for next iteration.
    // Except last_idx when needed for the allgather.
    send_idx = recv_idx;
    partner_mask >>= 1;
    slice_mask <<= 1;
    if (partner_mask > 0) {
      last_idx = recv_idx + nprocs / (slice_mask);
    }
  }
  ar_rs_time += get_time() - rs_start;
  // Do a recursive-doubling algather.
  double ag_start = get_time();
  slice_mask >>= 1;
  partner_mask = 1;
  // Now do the remaining steps.
  while (partner_mask < nprocs) {
    int partner = rank ^ partner_mask;
    const bool is_local = opts.no_local_trans &&
                          is_rank_node_local(partner, comm);
    // Determine range to send/recv.
    El::IR send_range, recv_range;
    if (rank < partner) {
      if (slice_mask != nprocs / 2) {
        last_idx = last_idx + nprocs / (slice_mask*2);
      }
      recv_idx = send_idx + nprocs / (slice_mask*2);
      send_range = El::IR(slice_ends[send_idx] - slice_lengths[send_idx],
                          slice_ends[recv_idx-1]);
      recv_range = El::IR(slice_ends[recv_idx] - slice_lengths[recv_idx],
                          slice_ends[last_idx-1]);
    } else {
      recv_idx = send_idx - nprocs / (slice_mask*2);
      send_range = El::IR(slice_ends[send_idx] - slice_lengths[send_idx],
                          slice_ends[last_idx-1]);
      recv_range = El::IR(slice_ends[recv_idx] - slice_lengths[recv_idx],
                          slice_ends[send_idx-1]);
    }
    auto recv_view = mat(El::ALL, recv_range);
    // Transform the data to send.
    double send_trans_start = get_time();
    int send_size;
    int recv_size = max_recv_count;
    uint8_t *send_buf = nullptr;
    if (is_local) {
      auto send_view = mat(El::ALL, send_range);
      send_buf = (uint8_t *) send_view.Buffer();
      send_size = sizeof(DataType) * send_view.Height() * send_view.Width();
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      send_buf = send_transform(mat, El::ALL, send_range, send_size, false, 0);
    }
    ar_send_transform_time += get_time() - send_trans_start;
    if (opts.id_recv || is_local) {
      recv_buf = (uint8_t *) recv_view.Buffer();
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    }
    bytes_sent += send_size;
    ar_bytes_sent += send_size;
    ar_ag_bytes_sent += send_size;
    double sendrecv_start = get_time();
    El::mpi::SendRecv(send_buf, send_size, partner,
                      recv_buf, recv_size, partner, comm);
    double sendrecv_tot = get_time() - sendrecv_start;
    ar_send_time += sendrecv_tot;
    ar_recv_time += sendrecv_tot;
    ar_ag_send_time += sendrecv_tot;
    ar_ag_recv_time += sendrecv_tot;
    double recv_trans_start = get_time();
    if (opts.id_recv) {
      recv_size = sizeof(DataType) * recv_view.Height() * recv_view.Width();
    } else {
      recv_size = recv_transform(recv_buf, recv_view);
    }
    ar_recv_transform_time += get_time() - recv_trans_start;
    bytes_received += recv_size;
    ar_bytes_received += recv_size;
    ar_ag_bytes_received += send_size;
    // Update for the next iteration.
    if (rank > partner) {
      send_idx = recv_idx;
    }
    partner_mask <<= 1;
    slice_mask >>= 1;
  }
  ar_ag_time += get_time() - ag_start;
  ar_time += get_time() - ar_start;
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
  El::mpi::Split(El::mpi::COMM_WORLD, hash,
                 El::mpi::Rank(El::mpi::COMM_WORLD), hash_comm);
  const int hash_comm_size = El::mpi::Size(hash_comm);

  // Compare node names and split MPI processes
  char *node_name_list = new char[hash_comm_size*MPI_MAX_PROCESSOR_NAME];
  checkMPI(MPI_Allgather(node_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         node_name_list, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         hash_comm.comm));
  int node_num = El::mpi::Rank(hash_comm);
  for(int i=0; i<hash_comm_size; ++i) {
    const std::string other_node_string(node_name_list + i*MPI_MAX_PROCESSOR_NAME);
    if(node_string == other_node_string) {
      node_num = i;
      break;
    }
  }
  delete[] node_name_list;
  El::mpi::Split(hash_comm, node_num, El::mpi::Rank(El::mpi::COMM_WORLD),
                 node_comm);
  El::mpi::Free(hash_comm);

  // Set up list of ranks that are local.
  int node_comm_size = El::mpi::Size(node_comm);
  for (int i = 0; i < node_comm_size; ++i) {
    world_ranks_on_node.push_back(
      El::mpi::Translate(node_comm, i, El::mpi::COMM_WORLD));
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
  if (threads_per_proc != omp_get_num_threads()) {
    omp_set_num_threads(threads_per_proc);
  }
}

uint8_t *lbann_comm::get_collective_buffer(size_t size, size_t idx) {
  auto buf_iter = collective_bufs.find(size);
  if (buf_iter == collective_bufs.end()) {
    if (idx != 0) {
      throw lbann_exception("get_collective_buffer: non-contiguous index");
    }
    collective_bufs.emplace(std::make_pair(size, std::vector<uint8_t *>()));
    collective_bufs[size].push_back(new uint8_t[size]);
    return collective_bufs[size][0];
  } else {
    if (collective_bufs[size].size() > idx) {
      return collective_bufs[size][idx];
    } else {
      if (collective_bufs[size].size() != idx) {
        throw lbann_exception("get_collective_buffer: non-contiguous index");
      }
      collective_bufs[size].push_back(new uint8_t[size]);
      return collective_bufs[size][idx];
    }
  }
}

}  // namespace lbann
