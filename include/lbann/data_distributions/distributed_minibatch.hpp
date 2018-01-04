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

#ifndef LBANN_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/data_distributions/data_distribution.hpp"

namespace lbann {

class data_buffer {
 public:
  /** Which rank is the root of the CircMat */
  int m_root;
  /** Requested maximum number of parallel readers (I/O streams) */
  int m_requested_max_num_parallel_readers;
  bool m_local_reader_done;
  /** Number of samples in the current mini-batch */
  int m_num_samples_in_batch;
  /** Has the layer copied valid data into the local matrix */
  bool m_local_data_valid;
  /** Number of samples in the current mini-batch */
  int m_num_data_per_epoch;

  Mat M_local; /** Local matrix that holds data from data reader */
  Mat M_local_v; /** View of local matrix that holds data from data reader */
  CircMat Ms; /** Distributed matrix used to stage local data to layer output */

  data_buffer(lbann_comm *comm, int num_parallel_readers) :
    m_root(0),
    m_requested_max_num_parallel_readers(num_parallel_readers),
    m_local_reader_done(false),
    m_num_samples_in_batch(0),
    m_local_data_valid(false),
    Ms(comm->get_model_grid()) {}

  data_buffer(
    const data_buffer&) = default;
  data_buffer& operator=(
    const data_buffer&) = default;
  data_buffer* copy() const { return new data_buffer(*this); }
};

/**
 * Parallel I/O routines for managing distributed minibatches
 */
class distributed_minibatch : public generic_data_distribution {
 public:
  typedef std::map<execution_mode, data_buffer *> data_buffer_map_t;
 public:
  distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);
  distributed_minibatch(const distributed_minibatch& other) :
    generic_data_distribution(other) {
    for (auto& buf : m_data_buffers) {
      buf.second = buf.second->copy();
    }
  }
  distributed_minibatch& operator=(const distributed_minibatch& other) {
    generic_data_distribution::operator=(other);
    for (auto& buf : m_data_buffers) {
      buf.second = buf.second->copy();
    }
    return *this;
  }
  virtual ~distributed_minibatch() {
    for (auto buf : m_data_buffers) {
      delete buf.second;
    }
  }

  void set_local_matrix_bypass(Mat *M_local) override {};

  int fetch_to_local_matrix(generic_data_reader *data_reader) override;
  void distribute_from_local_matrix(AbsDistMat& Ms, generic_data_reader *data_reader) override;
  bool is_data_set_processed(generic_data_reader *data_reader) override;

  void calculate_num_iterations_per_epoch(int num_models, int model_rank, int max_mini_batch_size, generic_data_reader *data_reader);
  void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) override;
  void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) override;
  int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const override;

  data_buffer *get_data_buffer(const execution_mode mode) const {
    data_buffer *data_buffer = nullptr;
    data_buffer_map_t::const_iterator it = m_data_buffers.find(mode);
    if (it != m_data_buffers.end()) data_buffer = it->second;

    switch(mode) {
    case execution_mode::training:
      break;
    case execution_mode::validation:
      break;
    case execution_mode::testing:
      break;
    default:
      throw lbann_exception(
                            std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                            " :: distributed_minibatch: invalid execution phase");
    }
    return data_buffer;
  }

  data_buffer *get_data_buffer() const {
    return get_data_buffer(execution_mode::training/*get_execution_mode()*/);
  }

  /// Return the rank of the current root node for the Elemental Distribution
  virtual int current_root_rank() const {
    data_buffer *buf = get_data_buffer();
    return buf->m_root;
  }

  /// Is this rank the current root node for the Elemental Distribution
  bool is_current_root() const {
    data_buffer *buf = get_data_buffer();
    return (m_comm->get_rank_in_model() == buf->m_root);
  }

  /// Is the local reader done
  virtual bool is_local_reader_done() const {
    data_buffer *buf = get_data_buffer();
    return buf->m_local_reader_done;
  }

  // protected:
  data_buffer_map_t m_data_buffers;
};

}  // namespace lbann

#endif  // LBANN_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
