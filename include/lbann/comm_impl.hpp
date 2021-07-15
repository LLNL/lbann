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

#ifndef LBANN_COMM_HPP_IMPL_INCLUDED
#define LBANN_COMM_HPP_IMPL_INCLUDED

#include "lbann/comm.hpp"

namespace lbann {

/// World broadcast of a scalar.
template <typename T> void lbann_comm::world_broadcast(int root, T& val) const
{
  broadcast(root, val, get_world_comm());
}
/// Inter-trainer broadcast of a scalar.
template <typename T>
void lbann_comm::intertrainer_broadcast(int root, T& val) const
{
  broadcast(root, val, get_intertrainer_comm());
}
/// Within-trainer broadcast of a scalar.
template <typename T> void lbann_comm::trainer_broadcast(int root, T& val) const
{
  broadcast(root, val, get_trainer_comm());
}

/**
 * Broadcast a buffer over an arbitrary communicator assuming that
 * the buffer space is already allocated.
 */

// Default to cpu memory
template <typename T>
void lbann_comm::broadcast(const int root,
                           T* const data,
                           const int count,
                           const El::mpi::Comm& c) const
{
  broadcast(root, data, count, c, El::SyncInfo<El::Device::CPU>{});
}

/// World broadcast of a buffer.
template <typename T>
void lbann_comm::world_broadcast(const int root,
                                 T* const data,
                                 const int count) const
{
  world_broadcast(root, data, count, El::SyncInfo<El::Device::CPU>{});
}

template <typename T, El::Device D>
void lbann_comm::world_broadcast(const int root,
                                 T* const data,
                                 const int count,
                                 El::SyncInfo<D> const& syncInfo) const
{
  broadcast(root, data, count, get_world_comm(), syncInfo);
}
/// Inter-trainer broadcast of a buffer.
template <typename T>
void lbann_comm::intertrainer_broadcast(const int root,
                                        T* data,
                                        const int count) const
{
  intertrainer_broadcast(root, data, count, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::intertrainer_broadcast(const int root,
                                        T* const data,
                                        const int count,
                                        El::SyncInfo<D> const& syncInfo) const
{
  broadcast(root, data, count, get_intertrainer_comm(), syncInfo);
}
/// Within-trainer broadcast of a buffer.
template <typename T>
void lbann_comm::trainer_broadcast(const int root,
                                   T* const data,
                                   const int count) const
{
  trainer_broadcast(root, data, count, El::SyncInfo<El::Device::CPU>{});
}

template <typename T, El::Device D>
void lbann_comm::trainer_broadcast(const int root,
                                   T* const data,
                                   const int count,
                                   El::SyncInfo<D> const& syncInfo) const
{
  broadcast(root, data, count, get_trainer_comm(), syncInfo);
}

/**
 * Resize vector<> over an arbitrary communicator to match the one on root.
 */
template <typename T>
size_t lbann_comm::resize(const int root,
                          std::vector<T>& data,
                          const El::mpi::Comm& c) const
{
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
void lbann_comm::broadcast(const int root,
                           std::vector<T>& data,
                           const El::mpi::Comm& c) const
{
  const int count = static_cast<int>(resize(root, data, c));
  if (count <= 0) {
    return;
  }
  broadcast(root, data.data(), count, c, El::SyncInfo<El::Device::CPU>{});
}
/// Broadcast vector<> to world.
template <typename T>
void lbann_comm::world_broadcast(const int root, std::vector<T>& data) const
{
  broadcast(root, data, get_world_comm());
}
/**
 * Broadcast vector<> within trainer;
 * vector<> for non-root processes will be resized as needed.
 */
/// Broadcast vector<> across trainers.
template <typename T>
void lbann_comm::intertrainer_broadcast(const int root,
                                        std::vector<T>& data) const
{
  broadcast(root, data, get_intertrainer_comm());
}
/// Broadcast vector<> within trainer.
template <typename T>
void lbann_comm::trainer_broadcast(const int root, std::vector<T>& data) const
{
  broadcast(root, data, get_trainer_comm());
}

/** Allgather over an arbitrary communicator */
template <typename T>
void lbann_comm::all_gather(const T* const src,
                            const int src_count,
                            T* const rcv,
                            const int rcv_count,
                            const El::mpi::Comm& c) const
{
  all_gather(src,
             src_count,
             rcv,
             rcv_count,
             c,
             El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::all_gather(const T* const src,
                            const int src_count,
                            T* const rcv,
                            const int rcv_count,
                            const El::mpi::Comm& c,
                            El::SyncInfo<D> const& syncInfo) const
{
  El::mpi::AllGather(src, src_count, rcv, rcv_count, c, syncInfo);
}

/**
 * Allgatherv over an arbitrary communicator;
 * all vectors must be correctly sized prior to entry.
 */
template <typename T>
void lbann_comm::all_gather(std::vector<T> const& src,
                            std::vector<T>& rcs,
                            std::vector<int> const& rcv_counts,
                            std::vector<int> const& rcv_disp,
                            const El::mpi::Comm& c) const
{
  if (src.size() == 0) {
    std::ostringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "all_gather for vector<>: vector.size() == 0;\n"
        << "this doesn't work!";
    lbann_comm_abort(err.str());
  }
  El::mpi::AllGather(src.data(),
                     src.size(),
                     rcs.data(),
                     rcv_counts.data(),
                     rcv_disp.data(),
                     c,
                     El::SyncInfo<El::Device::CPU>{});
}
/**
 * Allgatherv over a trainer communicator;
 * all vectors must be correctly sized prior to entry.
 */
template <typename T>
void lbann_comm::trainer_all_gather(std::vector<T> const& src,
                                    std::vector<T>& rcs,
                                    std::vector<int> const& rcv_counts,
                                    std::vector<int> const& rcv_disp) const
{
  all_gather(src, rcs, rcv_counts, rcv_disp, get_trainer_comm());
}
/**
 * Allgather for a single element over an arbitrary communicator;
 * std::vector<T> &data must be correctly sized prior to entry.
 */
template <typename T>
void lbann_comm::all_gather(T const& src,
                            std::vector<T>& data,
                            const El::mpi::Comm& c) const
{
  El::mpi::AllGather(&src,
                     1,
                     data.data(),
                     1,
                     c,
                     El::SyncInfo<El::Device::CPU>{});
}
/**
 * Allgather for a single element over the world communicator;
 * std::vector<T> &data must be correctly sized prior to entry.
 */
template <typename T>
void lbann_comm::world_all_gather(T const& src, std::vector<T>& data) const
{
  all_gather(src, data, get_world_comm());
}
/**
 * Allgather for a single element over the trainer communicator;
 * std::vector<T> &data must be correctly sized prior to entry.
 */
template <typename T>
void lbann_comm::trainer_all_gather(T const& src, std::vector<T>& data) const
{
  all_gather(src, data, get_trainer_comm());
}

/** Within-trainer scalar gather (for non-root processes). */
template <typename T>
void lbann_comm::trainer_gather(const T snd, const int root) const
{
  gather(snd, root, m_trainer_comm);
}
/** Within-trainer scalar gather (for root processes). */
template <typename T> void lbann_comm::trainer_gather(const T snd, T* rcv) const
{
  gather(snd, rcv, m_trainer_comm);
}
/** Within-trainer scalar-array gather (for non-root processes). */
template <typename T>
void lbann_comm::trainer_gather(T const* snd,
                                const int count,
                                const int root) const
{
  gather(snd, count, root, m_trainer_comm);
}
/** Within-trainer scalar-array gather (for root processes). */
template <typename T>
void lbann_comm::trainer_gather(T const* const snd,
                                const int count,
                                T* const rcv) const
{
  gather(snd, count, rcv, m_trainer_comm);
}
/** Within-trainer variable-length-array gather (for non-root processes). */
template <typename T>
void lbann_comm::trainer_gatherv(T const* snd,
                                 const int count,
                                 const int root) const
{
  m_bytes_sent += sizeof(T) * count;
  El::mpi::Gather(snd, count, nullptr, nullptr, nullptr, root, m_trainer_comm);
}
template <typename T>
void lbann_comm::trainer_gatherv(T const* const snd,
                                 const int count,
                                 T* const rcv,
                                 int const* const rcv_counts,
                                 int const* const rcv_displacements) const
{
  El::mpi::Gather(snd,
                  count,
                  rcv,
                  rcv_counts,
                  rcv_displacements,
                  get_rank_in_trainer(),
                  m_trainer_comm);
  m_bytes_received +=
    sizeof(T) *
    (std::accumulate(rcv_counts, &rcv_counts[get_procs_per_trainer()], 0) -
     rcv_counts[get_rank_in_trainer()]);
}
/** Inter-trainer gather (for non-root processes). */
template <typename T>
void lbann_comm::intertrainer_gather(const T snd, const int root) const
{
  gather(snd, root, m_intertrainer_comm);
}
/** Inter-trainer gather (for root processes). */
template <typename T>
void lbann_comm::intertrainer_gather(const T snd, std::vector<T>& rcv) const
{
  gather(snd, rcv, m_intertrainer_comm);
}
/** Inter-trainer scalar-array gather (for non-root processes). */
template <typename T>
void lbann_comm::intertrainer_gather(T const* const snd,
                                     const int count,
                                     const int root) const
{
  gather(snd, count, root, m_intertrainer_comm);
}
/** Inter-trainer scalar-array gather (for root processes). */
template <typename T>
void lbann_comm::intertrainer_gather(T const* const snd,
                                     const int count,
                                     T* const rcv) const
{
  gather(snd, count, rcv, m_intertrainer_comm);
}
/** Scalar gather (for non-root processes). */
template <typename T>
void lbann_comm::gather(const T snd,
                        const int root,
                        const El::mpi::Comm& c) const
{
  m_bytes_sent += sizeof(T);
  El::mpi::Gather(&snd,
                  1,
                  (T*)nullptr,
                  0,
                  root,
                  c,
                  El::SyncInfo<El::Device::CPU>{});
}
/** Scalar gather (for root processes). */
template <typename T>
void lbann_comm::gather(const T snd, T* const rcv, const El::mpi::Comm& c) const
{
  auto const size_c = El::mpi::Size(c);
  auto const rank_c = El::mpi::Rank(c);
  El::mpi::Gather(&snd, 1, rcv, 1, rank_c, c, El::SyncInfo<El::Device::CPU>{});
  m_bytes_received += sizeof(T) * (size_c - 1);
}
/** Scalar gather (for root processes). */
template <typename T>
void lbann_comm::gather(const T snd,
                        std::vector<T>& rcv,
                        const El::mpi::Comm& c) const
{
  gather(snd, rcv.data(), c);
}
/** Scalar-array gather (for non-root processes). */
template <typename T>
void lbann_comm::gather(T const* const snd,
                        const int count,
                        const int root,
                        const El::mpi::Comm& c) const
{
  gather(snd, count, root, c, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::gather(T const* const snd,
                        const int count,
                        const int root,
                        const El::mpi::Comm& c,
                        El::SyncInfo<D> const& syncInfo) const
{
  m_bytes_sent += sizeof(T) * count;
  El::mpi::Gather(snd, count, (T*)nullptr, 0, root, c, syncInfo);
}
/** Scalar-array gather (for root processes). */
template <typename T>
void lbann_comm::gather(T const* const snd,
                        const int count,
                        T* const rcv,
                        const El::mpi::Comm& c) const
{
  gather(snd, count, rcv, c, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::gather(T const* const snd,
                        const int count,
                        T* const rcv,
                        const El::mpi::Comm& c,
                        El::SyncInfo<D> const& syncInfo) const
{
  auto const size_c = El::mpi::Size(c);
  auto const rank_c = El::mpi::Rank(c);
  El::mpi::Gather(snd, count, rcv, count, rank_c, c, syncInfo);
  m_bytes_received += sizeof(T) * count * (size_c - 1);
}
/** Scalar scatter (for non-root processes). */
template <typename T>
T lbann_comm::scatter(const int root, const El::mpi::Comm& c) const
{
  T val = {};
  El::mpi::Scatter((T*)nullptr,
                   1,
                   &val,
                   1,
                   root,
                   c,
                   El::SyncInfo<El::Device::CPU>{});
  m_bytes_received += sizeof(T);
  return val;
}
/** Scalar scatter (for root processes). */
template <typename T>
T lbann_comm::scatter(T const* const snd, const El::mpi::Comm& c) const
{
  m_bytes_sent += sizeof(T) * (El::mpi::Size(c) - 1);
  T val = {};
  auto root = El::mpi::Rank(c);
  El::mpi::Scatter(snd, 1, &val, 1, root, c, El::SyncInfo<El::Device::CPU>{});
  return val;
}
/** Inter-trainer reduce (for non-root processes). */
template <typename T>
void lbann_comm::intertrainer_reduce(const T snd,
                                     const int root,
                                     const El::mpi::Op op) const
{
  reduce(snd, root, m_intertrainer_comm, op);
}
/** Inter-trainer reduce (for root processes). */
template <typename T>
T lbann_comm::intertrainer_reduce(const T snd, const El::mpi::Op op) const
{
  return reduce(snd, m_intertrainer_comm, op);
}
/** Within-trainer reduce (for non-root processes). */
template <typename T>
void lbann_comm::trainer_reduce(const T snd,
                                const int root,
                                const El::mpi::Op op) const
{
  reduce(snd, root, m_trainer_comm, op);
}
/** Within-trainer reduce (for root processes). */
template <typename T>
T lbann_comm::trainer_reduce(const T snd, const El::mpi::Op op) const
{
  return reduce(snd, m_trainer_comm, op);
}
/** Within-trainer scalar array reduce (for non-root processes). */
template <typename T>
void lbann_comm::trainer_reduce(T const* const snd,
                                const int count,
                                const int root,
                                const El::mpi::Op op) const
{
  reduce(snd, count, root, m_trainer_comm, op);
}
/** Within-trainer scalar array reduce (for root processes). */
template <typename T>
void lbann_comm::trainer_reduce(T const* const snd,
                                const int count,
                                T* const rcv,
                                const El::mpi::Op op) const
{
  reduce(snd, count, rcv, m_trainer_comm, op);
}
/** Scalar reduce (for non-root processes). */
template <typename T>
void lbann_comm::reduce(const T snd,
                        const int root,
                        const El::mpi::Comm& c,
                        const El::mpi::Op op) const
{
  m_bytes_sent += sizeof(T);
  El::mpi::Reduce(&snd,
                  (T*)NULL,
                  1,
                  op,
                  root,
                  c,
                  El::SyncInfo<El::Device::CPU>{});
}
/** Scalar reduce (for root processes). */
template <typename T>
T lbann_comm::reduce(const T snd,
                     const El::mpi::Comm& c,
                     const El::mpi::Op op) const
{
  T val = {};
  auto const size_c = El::mpi::Size(c);
  auto const rank_c = El::mpi::Rank(c);
  El::mpi::Reduce(&snd,
                  &val,
                  1,
                  op,
                  rank_c,
                  c,
                  El::SyncInfo<El::Device::CPU>{});
  m_bytes_received += sizeof(T) * (size_c - 1);
  return val;
}

/** Scalar-array reduce (for non-root processes). */
// Op is "SUM"
template <typename T>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        const int root,
                        const El::mpi::Comm& c) const
{
  reduce(snd, count, root, c, El::mpi::SUM, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        const int root,
                        const El::mpi::Comm& c,
                        El::SyncInfo<D> const& syncInfo) const
{
  reduce(snd, count, root, c, El::mpi::SUM, syncInfo);
}

template <typename T>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        const int root,
                        const El::mpi::Comm& c,
                        const El::mpi::Op op) const
{
  reduce(snd, count, root, c, op, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        const int root,
                        const El::mpi::Comm& c,
                        const El::mpi::Op op,
                        El::SyncInfo<D> const& syncInfo) const
{
  m_bytes_sent += sizeof(T) * count;
  El::mpi::Reduce(snd, (T*)nullptr, count, op, root, c, syncInfo);
}
/** Scalar-array reduce (for root processes). */
template <typename T, El::Device D>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        T* const rcv,
                        const El::mpi::Comm& c,
                        El::SyncInfo<D> const& syncInfo) const
{
  reduce(snd, count, rcv, c, El::mpi::SUM, syncInfo);
}
template <typename T>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        T* const rcv,
                        const El::mpi::Comm& c) const
{
  reduce(snd, count, rcv, c, El::mpi::SUM, El::SyncInfo<El::Device::CPU>{});
}

template <typename T>
void lbann_comm::reduce(T const* const snd,
                        const int count,
                        T* const rcv,
                        const El::mpi::Comm& c,
                        const El::mpi::Op op) const
{
  reduce(snd, count, rcv, c, op, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::reduce(T const* snd,
                        const int count,
                        T* const rcv,
                        const El::mpi::Comm& c,
                        El::mpi::Op op,
                        El::SyncInfo<D> const& syncInfo) const
{
  if (snd == rcv) {
    snd = (T const*)MPI_IN_PLACE;
  }
  auto const rank_c = El::mpi::Rank(c);
  auto const size_c = El::mpi::Size(c);
  El::mpi::Reduce(snd, rcv, count, op, rank_c, c, syncInfo);
  m_bytes_received += sizeof(T) * count * (size_c - 1);
}
/** Inter-trainer all-reduce. */
template <typename T>
T lbann_comm::intertrainer_allreduce(const T snd, const El::mpi::Op op) const
{
  return allreduce(snd, m_intertrainer_comm, op);
}
/** Within-trainer all-reduce. */
template <typename T>
T lbann_comm::trainer_allreduce(const T snd, const El::mpi::Op op) const
{
  return allreduce(snd, m_trainer_comm, op);
}
/** Scalar array within-trainer all-reduce. */
template <typename T>
void lbann_comm::trainer_allreduce(T const* const snd,
                                   const int count,
                                   T* const rcv,
                                   const El::mpi::Op op) const
{
  allreduce(snd, count, rcv, m_trainer_comm, op);
}
/** Scalar allreduce. */
template <typename T>
T lbann_comm::allreduce(T snd,
                        const El::mpi::Comm& c,
                        const El::mpi::Op op) const
{
  auto const size_c = El::mpi::Size(c);
  m_bytes_sent += sizeof(T);
  allreduce(&snd, 1, c, op);
  m_bytes_received += sizeof(T) * (size_c - 1);
  return snd;
}

// FIXME (trb): Based on the backend choice of "MPIBackend", I'm
// assuming this is intended as a CPU-only call.
/** Scalar-array allreduce. */
template <typename T>
void lbann_comm::allreduce(T const* const snd,
                           const int count,
                           T* const rcv,
                           const El::mpi::Comm& c,
                           const El::mpi::Op op) const
{
  auto const size_c = El::mpi::Size(c);
  m_bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
#ifdef LBANN_ALUMINUM_MPI_PASSTHROUGH
  ::Al::MPIAllreduceAlgorithm algo =
    ::Al::MPIAllreduceAlgorithm::mpi_passthrough;
#else
  ::Al::MPIAllreduceAlgorithm algo = ::Al::MPIAllreduceAlgorithm::automatic;
#endif
  ::Al::Allreduce<::Al::MPIBackend>(
    snd,
    rcv,
    count,
    mpi_op_to_al_op(op),
    c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}),
    algo);
#else
  El::mpi::AllReduce(snd, rcv, count, op, c, El::SyncInfo<El::Device::CPU>{});
#endif
  m_bytes_received += count * sizeof(T) * (size_c - 1);
}
/** In-place scalar-array allreduce. */
template <typename T>
void lbann_comm::allreduce(T* const data,
                           const int count,
                           const El::mpi::Comm& c,
                           const El::mpi::Op op) const
{
  auto const size_c = El::mpi::Size(c);
  m_bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
#ifdef LBANN_ALUMINUM_MPI_PASSTHROUGH
  ::Al::MPIAllreduceAlgorithm algo =
    ::Al::MPIAllreduceAlgorithm::mpi_passthrough;
#else
  ::Al::MPIAllreduceAlgorithm algo = ::Al::MPIAllreduceAlgorithm::automatic;
#endif
  ::Al::Allreduce<::Al::MPIBackend>(
    data,
    count,
    mpi_op_to_al_op(op),
    c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}),
    algo);
#else
  El::mpi::AllReduce(data, count, op, c, El::SyncInfo<El::Device::CPU>{});
#endif
  m_bytes_received += count * sizeof(T) * (size_c - 1);
}
/** Non-blocking in-place scalar-array allreduce.
 *  If LBANN has not been built with Aluminum, then this calls a blocking
 *  allreduce.
 *  This currently only supports host pointers (i.e. the MPI backend).
 */
template <typename T>
void lbann_comm::nb_allreduce(T* data,
                              const int count,
                              const El::mpi::Comm& c,
                              Al::request& req,
                              const El::mpi::Op op) const
{
  m_bytes_sent += count * sizeof(T);
#ifdef LBANN_HAS_ALUMINUM
  req.mpi_req = Al::mpi_null_req;
  ::Al::NonblockingAllreduce<::Al::MPIBackend>(
    data,
    count,
    mpi_op_to_al_op(op),
    c.template GetComm<::Al::MPIBackend>(El::SyncInfo<El::Device::CPU>{}),
    req.mpi_req);
#else
  MPI_Iallreduce(MPI_IN_PLACE,
                 data,
                 count,
                 El::mpi::TypeMap<T>(),
                 op.op,
                 c.GetMPIComm(),
                 &(req.raw_mpi_req));
#endif // LBANN_HAS_ALUMINUM
  m_bytes_received += count * sizeof(T) * (El::mpi::Size(c) - 1);
}

/** Wait for a all non-blocking requests to complete. */
template <typename T>
void lbann_comm::wait_all(std::vector<El::mpi::Request<T>>& req) const
{
  El::mpi::WaitAll(req.size(), req.data());
}

/** Wait for a non-blocking request to complete. */
template <typename T> void lbann_comm::wait(El::mpi::Request<T>& req) const
{
  El::mpi::Wait(req);
}

/** Send a buffer to rank in trainer. */
template <typename T>
void lbann_comm::send(const T* const data,
                      const int count,
                      const int trainer,
                      const int rank) const
{
  send(data, count, trainer, rank, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::send(const T* const data,
                      const int count,
                      const int trainer,
                      const int rank,
                      El::SyncInfo<D> const& syncInfo) const
{
  m_bytes_sent += sizeof(T) * count;
  El::mpi::Send(data,
                count,
                get_world_rank(trainer, rank),
                get_world_comm(),
                syncInfo);
}
template <typename T, El::Device D>
void lbann_comm::send(const T* const data,
                      const int count,
                      const int trainer,
                      El::SyncInfo<D> const& syncInfo) const
{
  send(data, count, trainer, m_rank_in_trainer, syncInfo);
}

/** Corresponding non-blocking sends. */
template <typename T>
void lbann_comm::nb_send(const T* const data,
                         const int count,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<T>& req) const
{
  m_bytes_sent += sizeof(T) * count;
  El::mpi::ISend(data,
                 count,
                 get_world_rank(trainer, rank),
                 get_world_comm(),
                 req);
}
template <typename T>
void lbann_comm::nb_tagged_send(const T* const data,
                                const int count,
                                const int rank,
                                const int tag,
                                El::mpi::Request<T>& req,
                                const El::mpi::Comm& c) const
{
  m_bytes_sent += sizeof(T) * count;
  El::mpi::TaggedISend(data, count, rank, tag, c, req);
}
template <typename T>
void lbann_comm::nb_send(const T* const data,
                         const int count,
                         const int trainer,
                         El::mpi::Request<T>& req) const
{
  nb_send(data, count, trainer, m_rank_in_trainer, req);
}

/** Corresponding receive to send. */
template <typename T>
void lbann_comm::recv(T* const data,
                      const int count,
                      const int trainer,
                      const int rank) const
{
  recv(data, count, trainer, rank, El::SyncInfo<El::Device::CPU>{});
}
template <typename T>
void lbann_comm::recv(T* data, const int count, const int trainer) const
{
  recv(data, count, trainer, m_rank_in_trainer);
}
template <typename T>
void lbann_comm::recv(T* const data, const int count) const
{
  recv(data, count, El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::recv(T* const data,
                      const int count,
                      const int trainer,
                      const int rank,
                      El::SyncInfo<D> const& syncInfo) const
{
  El::mpi::Recv(data,
                count,
                get_world_rank(trainer, rank),
                get_world_comm(),
                syncInfo);
  m_bytes_received += sizeof(T) * count;
}
template <typename T, El::Device D>
void lbann_comm::recv(T* const data,
                      const int count,
                      const int trainer,
                      El::SyncInfo<D> const& syncInfo) const
{
  recv(data, count, trainer, m_rank_in_trainer, syncInfo);
}
/** As above, but receive from anyone. */
template <typename T, El::Device D>
void lbann_comm::recv(T* const data,
                      const int count,
                      El::SyncInfo<D> const& syncInfo) const
{
  El::mpi::Recv(data, count, El::mpi::ANY_SOURCE, get_world_comm(), syncInfo);
  m_bytes_received += sizeof(T) * count;
}

/** Corresponding non-blocking receives. */
template <typename T>
void lbann_comm::nb_recv(T* const data,
                         const int count,
                         const int trainer,
                         const int rank,
                         El::mpi::Request<T>& req) const
{
  El::mpi::IRecv(data,
                 count,
                 get_world_rank(trainer, rank),
                 get_world_comm(),
                 req);
  m_bytes_received += sizeof(T) * count;
}
template <typename T>
void lbann_comm::nb_tagged_recv(T* const data,
                                const int count,
                                const int rank,
                                const int tag,
                                El::mpi::Request<T>& req,
                                const El::mpi::Comm& c) const
{
  El::mpi::TaggedIRecv(data, count, rank, tag, c, req);
  m_bytes_received += sizeof(T) * count;
}

template <typename T>
void lbann_comm::nb_recv(T* const data,
                         const int count,
                         const int trainer,
                         El::mpi::Request<T>& req) const
{
  nb_recv(data, count, trainer, m_rank_in_trainer, req);
}
template <typename T>
void lbann_comm::nb_recv(T* const data,
                         const int count,
                         El::mpi::Request<T>& req) const
{
  El::mpi::IRecv(data, count, El::mpi::ANY_SOURCE, get_world_comm(), req);
  m_bytes_received += sizeof(T) * count;
}

/** Send/recv to/from ranks. */
template <typename T, El::Device D>
void lbann_comm::sendrecv(const T* const snd,
                          const int send_count,
                          const int send_trainer,
                          const int send_rank,
                          T* const rcv,
                          const int recv_count,
                          const int recv_trainer,
                          const int recv_rank) const
{
  sendrecv(snd,
           send_count,
           send_trainer,
           send_rank,
           rcv,
           recv_count,
           recv_trainer,
           recv_rank,
           El::SyncInfo<El::Device::CPU>{});
}
template <typename T, El::Device D>
void lbann_comm::sendrecv(const T* const snd,
                          const int send_count,
                          const int send_trainer,
                          T* const rcv,
                          const int recv_count,
                          const int recv_trainer) const
{
  sendrecv(snd,
           send_count,
           send_trainer,
           m_rank_in_trainer,
           rcv,
           recv_count,
           recv_trainer,
           m_rank_in_trainer,
           El::SyncInfo<El::Device::CPU>{});
}

template <typename T, El::Device D>
void lbann_comm::sendrecv(const T* const snd,
                          const int send_count,
                          const int send_trainer,
                          const int send_rank,
                          T* const rcv,
                          const int recv_count,
                          const int recv_trainer,
                          const int recv_rank,
                          El::SyncInfo<D> const& syncInfo) const
{
  m_bytes_sent += sizeof(T) * send_count;
  m_bytes_received += sizeof(T) * recv_count;
  El::mpi::SendRecv(snd,
                    send_count,
                    get_world_rank(send_trainer, send_rank),
                    rcv,
                    recv_count,
                    get_world_rank(recv_trainer, recv_rank),
                    get_world_comm(),
                    syncInfo);
}
template <typename T, El::Device D>
void lbann_comm::sendrecv(const T* const snd,
                          const int send_count,
                          const int send_trainer,
                          T* const rcv,
                          const int recv_count,
                          const int recv_trainer,
                          El::SyncInfo<D> const& syncInfo) const
{
  sendrecv(snd,
           send_count,
           send_trainer,
           m_rank_in_trainer,
           rcv,
           recv_count,
           recv_trainer,
           m_rank_in_trainer,
           syncInfo);
}

/** Determine the size (count) of an incoming message. */
template <typename T>
int lbann_comm::get_count(const int trainer, const int rank) const
{
  MPI_Status status;
  MPI_Probe(get_world_rank(trainer, rank),
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &status);
  return El::mpi::GetCount<T>(status);
}
template <typename T> int lbann_comm::get_count(const int trainer) const
{
  return get_count<T>(trainer, m_rank_in_trainer);
}

template <typename T, bool S>
void lbann_comm::broadcast(const int root, T& val, const El::mpi::Comm& c) const
{
  auto const rank_c = El::mpi::Rank(c);
  if (S) {
    // Avoid linking error from uninstantiated El::mpi routine if !S by
    // converting T to El::byte
    using TT = typename interpret_as_byte_if_needed<S, T>::type;
    broadcast_native<TT>(root, reinterpret_cast<TT&>(val), c);
  }
  else {
    broadcast_custom(root, val, c);
  }
  count_bytes_broadcast(sizeof(T), rank_c, root);
}

template <typename T>
void lbann_comm::broadcast_native(const int root,
                                  T& val,
                                  const El::mpi::Comm& c) const
{
  El::mpi::Broadcast(val, root, c, El::SyncInfo<El::Device::CPU>{});
}

template <typename T>
void lbann_comm::broadcast_custom(const int root,
                                  T& val,
                                  const El::mpi::Comm& c) const
{
  const int bytes = static_cast<int>(sizeof(T));
  El::mpi::Broadcast<El::byte>(reinterpret_cast<El::byte*>(&val),
                               bytes,
                               root,
                               c,
                               El::SyncInfo<El::Device::CPU>{});
}

template <typename T, El::Device D, bool S>
void lbann_comm::broadcast(const int root,
                           T* const data,
                           const int count,
                           const El::mpi::Comm& c,
                           El::SyncInfo<D> const& syncInfo) const
{
  auto const rank_c = El::mpi::Rank(c);
  const int size = static_cast<int>(S ? count : sizeof(T) * count);
  // Avoid linking error from uninstantiated El::mpi routine if !S by converting
  // T to El::byte
  using TT = typename interpret_as_byte_if_needed<S, T>::type;
  El::mpi::Broadcast<TT>(reinterpret_cast<TT*>(data), size, root, c, syncInfo);
  count_bytes_broadcast(sizeof(T) * count, rank_c, root);
}

/// Broadcast std::string over an arbitrary communicator.
template <>
void lbann_comm::broadcast<std::string>(int root,
                                        std::string& str,
                                        const El::mpi::Comm& c) const;

#ifndef LBANN_COMM_INSTANTIATE
#define PROTO(T)                                                               \
  extern template void lbann_comm::allreduce(El::AbstractMatrix<T>& m,         \
                                             const El::mpi::Comm& c,           \
                                             El::mpi::Op op) const;            \
  extern template void lbann_comm::allreduce(El::AbstractDistMatrix<T>& m,     \
                                             const El::mpi::Comm& c,           \
                                             El::mpi::Op op) const;            \
  extern template void lbann_comm::nb_allreduce(El::AbstractMatrix<T>& m,      \
                                                const El::mpi::Comm& c,        \
                                                Al::request& req,              \
                                                El::mpi::Op op) const;         \
  extern template void lbann_comm::nb_allreduce(El::AbstractDistMatrix<T>& m,  \
                                                const El::mpi::Comm& c,        \
                                                Al::request& req,              \
                                                El::mpi::Op op) const

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_COMM_INSTANTIATE

} // namespace lbann

#endif // LBANN_COMM_IMPL_HPP_INCLUDED
