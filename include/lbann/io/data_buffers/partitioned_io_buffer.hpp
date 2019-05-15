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

#ifndef LBANN_PARTITIONED_IO_BUFFER_HPP_INCLUDED
#define LBANN_PARTITIONED_IO_BUFFER_HPP_INCLUDED

#include "lbann/io/data_buffers/generic_io_buffer.hpp"

namespace lbann {

class data_buffer {
 public:
  /** Number of samples in the current mini-batch */
  int m_num_samples_fetched;
  /** Distributed matrix used to stage local data to layer output */
  std::vector<std::unique_ptr<AbsDistMat>> m_input_buffers;
  std::atomic<bool> m_fetch_data_in_background;
  std::future<void> m_data_fetch_future;
  /// 1-D Matrix of which indices were fetched in this mini-batch
  El::Matrix<El::Int> m_indices_fetched_per_mb;

  data_buffer(lbann_comm *comm, int num_child_layers) :
    m_num_samples_fetched(0), m_fetch_data_in_background(false)
  {
    m_input_buffers.clear();
    m_input_buffers.resize(num_child_layers);
    for(int i = 0; i < num_child_layers; i++) {
      m_input_buffers[i].reset(new StarVCMat<El::Device::CPU>(comm->get_trainer_grid()));
    }
  }

  data_buffer(const data_buffer& other) :
    m_num_samples_fetched(other.m_num_samples_fetched)
  {
    m_fetch_data_in_background.store(other.m_fetch_data_in_background);
    m_input_buffers.clear();
    m_input_buffers.reserve(other.m_input_buffers.size());
    for (const auto& ptr : other.m_input_buffers) {
      m_input_buffers.emplace_back(ptr ? ptr->Copy() : nullptr);
    }
  }
  data_buffer& operator=(const data_buffer& other) {
    m_num_samples_fetched = other.m_num_samples_fetched;
    m_fetch_data_in_background.store(other.m_fetch_data_in_background);
    m_input_buffers.clear();
    m_input_buffers.reserve(other.m_input_buffers.size());
    for (const auto& ptr : other.m_input_buffers) {
      m_input_buffers.emplace_back(ptr ? ptr->Copy() : nullptr);
    }
    return *this;
  }
  data_buffer* copy() const { return new data_buffer(*this); }
};

/**
 * Parallel I/O routines for managing partitioned minibatches
 */
class partitioned_io_buffer : public generic_io_buffer {
 public:
  typedef std::map<execution_mode, data_buffer *> data_buffer_map_t;
 public:
  partitioned_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, int num_child_layers);
  partitioned_io_buffer(const partitioned_io_buffer& other);
  partitioned_io_buffer& operator=(const partitioned_io_buffer& other);
  ~partitioned_io_buffer();
  partitioned_io_buffer* copy() const override;

  std::string get_type() const override { return "partitioned"; }

  void fp_setup_data(El::Int cur_mini_batch_size, int idx) override;
  void setup_data(El::Int num_neurons, El::Int num_targets, El::Int max_mini_batch_size) override;

  int fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) override;
  void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample, AbsDistMat& response) override;
  void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample) override;
  bool update_data_set(generic_data_reader *data_reader, execution_mode mode) override;
  void set_fetch_data_in_background(bool flag, execution_mode mode) override;
  bool is_data_fetched_in_background(execution_mode mode) override;
  El::Matrix<El::Int>* get_sample_indices_fetched_per_mb(execution_mode mode) override;
  int num_samples_ready(execution_mode mode) override;
  void set_data_fetch_future(std::future<void> future, execution_mode mode) override;
  std::future<void> get_data_fetch_future(execution_mode mode) override;

  void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) override;
  void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) override;
  int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const override;
  static int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers, const lbann_comm* comm);

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

  /** Input data buffers
   *  There is a buffer for each phase of execution.
   *  Each matrix column corresponds to a flattened mini-batch sample
   *  or label or responase.
   */
  data_buffer_map_t m_data_buffers;
};
}

#endif  // LBANN_PARTITIONED_IO_BUFFER_HPP_INCLUDED
