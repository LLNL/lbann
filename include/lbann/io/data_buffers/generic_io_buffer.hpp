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

#ifndef LBANN_GENERIC_IO_BUFFER_HPP_INCLUDED
#define LBANN_GENERIC_IO_BUFFER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/threads/thread_pool.hpp"

#include <future>

namespace lbann
{
class fetch_data_functor {
 public:
  fetch_data_functor (data_reader_target_mode target_mode) :
    _target_mode(target_mode) {}
  int operator() (CPUMat& samples, CPUMat& responses, El::Matrix<El::Int>& indices_fetched, generic_data_reader* data_reader) const {
    int num_samples_fetched = data_reader->fetch_data(samples, indices_fetched);
    int num_responses_fetched;
    switch(_target_mode) {
    case data_reader_target_mode::REGRESSION:
      num_responses_fetched = data_reader->fetch_responses(responses);
      break;
    case data_reader_target_mode::RECONSTRUCTION:
      El::Copy(samples, responses);
      num_responses_fetched = num_samples_fetched;
      break;
    case data_reader_target_mode::NA:
       throw lbann_exception("Invalid data reader target mode");
    case data_reader_target_mode::CLASSIFICATION:
    default:
      num_responses_fetched = data_reader->fetch_labels(responses);
    }
    if(num_samples_fetched != num_responses_fetched) {
      std::string err = std::string("Number of samples: ") + std::to_string(num_samples_fetched)
        + std::string(" does not match the number of responses: ") + std::to_string(num_responses_fetched);
      throw lbann_exception(err);
    }
    return num_samples_fetched;
  }
  int operator() (CPUMat& samples, El::Matrix<El::Int>& indices_fetched, generic_data_reader* data_reader) const {
    int num_samples_fetched = data_reader->fetch_data(samples, indices_fetched);
    switch(_target_mode) {
    case data_reader_target_mode::NA:
      break;
    case data_reader_target_mode::REGRESSION:
    case data_reader_target_mode::RECONSTRUCTION:
    case data_reader_target_mode::CLASSIFICATION:
    default:
      throw lbann_exception("Invalid data reader target mode");
    }
    return num_samples_fetched;
  }
 private:
  const data_reader_target_mode _target_mode;
};

class update_data_reader_functor {
 public:
  update_data_reader_functor () {}
  int operator() (bool is_active_reader, generic_data_reader* data_reader) const {
    return data_reader->update(is_active_reader);
  }
};

class generic_io_buffer {
public:
  generic_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);
  generic_io_buffer(
    const generic_io_buffer&);
  generic_io_buffer& operator=(
    const generic_io_buffer&);
  virtual ~generic_io_buffer() {
    if(fetch_data_fn != nullptr) {
      delete fetch_data_fn;
    }
    if(update_data_reader_fn != nullptr) {
      delete update_data_reader_fn;
    }
  }
  virtual generic_io_buffer* copy() const = 0;

  /** Return this buffer's type, e.g: "partitioned_io_buffer," etc. */
  virtual std::string get_type() const = 0;
  virtual void fp_setup_data(El::Int cur_mini_batch_size, int idx) = 0;
  virtual void setup_data(El::Int num_neurons, El::Int num_targets, El::Int max_minibatch_size) = 0;

  virtual int fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) = 0;
  virtual void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample, AbsDistMat& response) {}
  virtual void distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample) {}
  virtual bool update_data_set(generic_data_reader *data_reader, execution_mode mode) = 0;
  virtual void set_fetch_data_in_background(bool flag, execution_mode mode) = 0;
  virtual bool is_data_fetched_in_background(execution_mode mode) = 0;
  virtual El::Matrix<El::Int>* get_sample_indices_fetched_per_mb(execution_mode mode) = 0;
  virtual int num_samples_ready(execution_mode mode) = 0;
  virtual void set_data_fetch_future(std::future<void> future, execution_mode mode) = 0;
  virtual std::future<void> get_data_fetch_future(execution_mode mode) = 0;

  virtual void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
  virtual void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) = 0;

  virtual int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const = 0;

  // protected:
 public:
  lbann_comm *m_comm;
  const fetch_data_functor *fetch_data_fn;
  const update_data_reader_functor *update_data_reader_fn;
};
}

#endif // LBANN_GENERIC_IO_BUFFER_HPP_INCLUDED
