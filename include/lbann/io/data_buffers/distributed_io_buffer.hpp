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

#ifndef LBANN_DISTRIBUTED_IO_BUFFER_HPP_INCLUDED
#define LBANN_DISTRIBUTED_IO_BUFFER_HPP_INCLUDED

#include "lbann/io/data_buffers/generic_io_buffer.hpp"

namespace lbann {

class data_buffer {
 public:
  /** Which rank is the root of the CircMat */
  int m_root;
  bool m_local_reader_done;
  /** Number of samples in the current mini-batch */
  int m_num_samples_in_batch;
  /** Has the layer copied valid data into the local matrix */
  bool m_local_data_valid;
  /** Number of samples in the current mini-batch */
  int m_num_data_per_epoch;

  std::vector<CPUMat*> M_local; /** Local matrix that holds data from data reader */
  std::vector<CPUMat*> M_local_v; /** View of local matrix that holds data from data reader */
  std::vector<CircMat<El::Device::CPU>*> Ms; /** Distributed matrix used to stage local data to layer output */

  data_buffer(lbann_comm *comm) :
    m_root(0),
    m_local_reader_done(false),
    m_num_samples_in_batch(0),
    m_local_data_valid(false) {

    for (int i = 0; i < 2; i++) {
      M_local.push_back(new CPUMat());
      M_local_v.push_back(new CPUMat());
      Ms.push_back(new CircMat<El::Device::CPU>(comm->get_model_grid()));
    }
  }

  data_buffer(
    const data_buffer&) = default;
  data_buffer& operator=(
    const data_buffer&) = default;
  data_buffer* copy() const { return new data_buffer(*this); }
};

/**
 * Parallel I/O routines for managing distributed minibatches
 */
class distributed_io_buffer : public generic_io_buffer {
 public:
  typedef std::map<execution_mode, data_buffer *> data_buffer_map_t;
  /** Requested maximum number of parallel readers (I/O streams) */
  int m_requested_max_num_parallel_readers;
 public:
  distributed_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);
  distributed_io_buffer(const distributed_io_buffer& other) :
    generic_io_buffer(other) {
    m_requested_max_num_parallel_readers = other.m_requested_max_num_parallel_readers;
    for (auto& buf : m_data_buffers) {
      buf.second = buf.second->copy();
    }
  }
  distributed_io_buffer& operator=(const distributed_io_buffer& other) {
    generic_io_buffer::operator=(other);
    m_requested_max_num_parallel_readers = other.m_requested_max_num_parallel_readers;
    for (auto& buf : m_data_buffers) {
      buf.second = buf.second->copy();
    }
    return *this;
  }
  virtual ~distributed_io_buffer() {
    for (auto buf : m_data_buffers) {
      for (auto m : buf.second->M_local) { delete m; }
      for (auto m : buf.second->M_local_v) { delete m; }
      for (auto m : buf.second->Ms) { delete m; }
      delete buf.second;
    }
  }
  distributed_io_buffer* copy() const override { return new distributed_io_buffer(*this); }

  std::string get_type() const override { return "distributed_io_buffer"; }

  void set_local_matrix_bypass(CPUMat *M_local, int idx) override {}

  void set_std_matrix_view(El::Int cur_mini_batch_size, int idx) override {
    for (auto& buf : m_data_buffers) {
      El::View(*buf.second->M_local_v[idx], *buf.second->M_local[idx], El::ALL, El::IR(0, cur_mini_batch_size));
    }
  }

  void setup_data(El::Int num_neurons, El::Int max_minibatch_size) override {
    for (auto& buf : m_data_buffers) {
      buf.second->M_local[0]->Resize(num_neurons, max_minibatch_size);
      buf.second->Ms[0]->Resize(num_neurons, max_minibatch_size);
    }
  }

  int fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) override;
  void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample, AbsDistMat& response) override;
  bool is_data_set_processed(generic_data_reader *data_reader, execution_mode mode) override;

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
                            " :: distributed_io_buffer: invalid execution phase");
    }
    return data_buffer;
  }

  /// Return the rank of the current root node for the Elemental Distribution
  virtual int current_root_rank(execution_mode mode) const {
    data_buffer *buf = get_data_buffer(mode);
    return buf->m_root;
  }

  /// Is this rank the current root node for the Elemental Distribution
  bool is_current_root(execution_mode mode) const {
    data_buffer *buf = get_data_buffer(mode);
    return (m_comm->get_rank_in_model() == buf->m_root);
  }

  /// Is the local reader done
  virtual bool is_local_reader_done(execution_mode mode) const {
    data_buffer *buf = get_data_buffer(mode);
    return buf->m_local_reader_done;
  }

  // protected:
  data_buffer_map_t m_data_buffers;
};

}  // namespace lbann

#endif  // LBANN_DISTRIBUTED_IO_BUFFER_HPP_INCLUDED
