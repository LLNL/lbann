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

#ifndef LBANN_GENERIC_IO_BUFFER_HPP_INCLUDED
#define LBANN_GENERIC_IO_BUFFER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/data_readers/data_reader.hpp"

namespace lbann
{
class fetch_data_functor {
 public:
  fetch_data_functor (bool is_input_layer, bool is_for_regression) :
    _is_input_layer(is_input_layer), _is_for_regression(is_for_regression) {}
  int operator() (CPUMat& M_local, generic_data_reader* data_reader) const {
    if (_is_input_layer) {
      return data_reader->fetch_data(M_local);
    } else {
      if (_is_for_regression) {
        return data_reader->fetch_responses(M_local);
      } else {
        return data_reader->fetch_labels(M_local);
      }
    }
  }
 private:
  const bool _is_input_layer;
  const bool _is_for_regression;
};

class update_data_reader_functor {
 public:
  update_data_reader_functor (bool is_input_layer) :
    _is_input_layer(is_input_layer) {}
  int operator() (bool is_active_reader, generic_data_reader* data_reader) const {
    if (_is_input_layer) {
      return data_reader->update(is_active_reader);
    } else {
      return (data_reader->is_data_reader_done(is_active_reader));
    }
  }
 private:
  const bool _is_input_layer;
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

  /** Return this buffer's type, e.g: "partitioned_io_buffer," "distributed_io_buffer," etc. */
  virtual std::string get_type() const = 0;
  virtual void set_local_matrix_bypass(CPUMat *M_local) = 0;
  virtual void set_std_matrix_view(El::Int cur_mini_batch_size) = 0;
  virtual void setup_data(El::Int num_neurons, El::Int max_minibatch_size) = 0;

  virtual int fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) = 0;
  virtual void distribute_from_local_matrix(AbsDistMat& Ms, generic_data_reader *data_reader, execution_mode mode) {}
  virtual bool is_data_set_processed(generic_data_reader *data_reader, execution_mode mode) = 0;

  virtual void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
  virtual void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
;
  virtual int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const = 0;

  // protected:
 public:
  lbann_comm *m_comm;
  const fetch_data_functor *fetch_data_fn;
  const update_data_reader_functor *update_data_reader_fn;
};
}

#endif // LBANN_GENERIC_IO_BUFFER_HPP_INCLUDED
