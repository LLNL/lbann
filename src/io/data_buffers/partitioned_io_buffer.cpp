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

#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

template <typename TensorDataType>
partitioned_io_buffer<TensorDataType>::partitioned_io_buffer(lbann_comm *comm, int num_parallel_readers, int num_child_layers)
  : generic_io_buffer<TensorDataType>(comm, num_parallel_readers) {
  m_data_buffers[execution_mode::training] = new data_buffer<IODataType>(comm, num_child_layers);
  m_data_buffers[execution_mode::validation] = new data_buffer<IODataType>(comm, num_child_layers);
  m_data_buffers[execution_mode::testing] = new data_buffer<IODataType>(comm, num_child_layers);
}

template <typename TensorDataType>
partitioned_io_buffer<TensorDataType>::~partitioned_io_buffer() {
  for (auto& buf : m_data_buffers) {
    delete buf.second;
  }
}

template <typename TensorDataType>
partitioned_io_buffer<TensorDataType>::partitioned_io_buffer(const partitioned_io_buffer& other)
  : generic_io_buffer<TensorDataType>(other) {
  for (const auto& buf : other.m_data_buffers) {
    m_data_buffers[buf.first] = buf.second->copy();
  }
}

template <typename TensorDataType>
partitioned_io_buffer<TensorDataType>* partitioned_io_buffer<TensorDataType>::copy() const {
  return new partitioned_io_buffer<TensorDataType>(*this);
}

template <typename TensorDataType>
partitioned_io_buffer<TensorDataType>& partitioned_io_buffer<TensorDataType>::operator=(const partitioned_io_buffer& other) {
  generic_io_buffer<TensorDataType>::operator=(other);
  for (auto& buf : m_data_buffers) {
    if (buf.second) delete buf.second;
    buf.second = buf.second->copy();
  }
  return *this;
}

template <typename TensorDataType>
void partitioned_io_buffer<TensorDataType>::fp_setup_data(El::Int cur_mini_batch_size, int idx) {
#ifdef LBANN_HAS_DISTCONV
  cur_mini_batch_size *= dc::get_number_of_io_partitions();
#endif
  for (auto& buf : m_data_buffers) {
    buf.second->m_input_buffers[idx]->Resize(buf.second->m_input_buffers[idx]->Height(), cur_mini_batch_size);
  }
}

template <typename TensorDataType>
void partitioned_io_buffer<TensorDataType>::setup_data(El::Int num_neurons, El::Int num_targets, El::Int max_mini_batch_size) {
#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    num_neurons /= dc::get_number_of_io_partitions();
    // TensorDataType is assumed to be 2-byte integer types such as
    // short or int16_t.
    assert_eq(sizeof(TensorDataType), sizeof(short));
    max_mini_batch_size *= dc::get_number_of_io_partitions();
  }
#endif // LBANN_HAS_DISTCONV
  El::Int local_mini_batch_size = max_mini_batch_size / this->m_comm->get_procs_per_trainer();
  El::Int partial_mini_batch_size = max_mini_batch_size % this->m_comm->get_procs_per_trainer();
#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    assert_eq(local_mini_batch_size, 1);
    assert_eq(partial_mini_batch_size, 0);
  }
#endif // LBANN_HAS_DISTCONV
  if(partial_mini_batch_size > 0 && this->m_comm->get_rank_in_trainer() < partial_mini_batch_size) {
    local_mini_batch_size++;
  }
  for (const auto& it : m_data_buffers) {
    data_buffer<IODataType> *data_buffer = it.second;
    int i = 0;
    for (const auto& buf : data_buffer->m_input_buffers) {
      if(i == 0) {
        buf->Resize(num_neurons, max_mini_batch_size);
      }else if(i == 1) {
        buf->Resize(num_targets, max_mini_batch_size);
      }else {
        LBANN_ERROR("Unsupported number of input channels");
      }
      i++;
    }
    /// The amount of space needed will vary based on input layer type,
    /// but the batch size is the maximum space necessary
    El::Zeros_seq(data_buffer->m_indices_fetched_per_mb, local_mini_batch_size, 1);
  }
}

template <typename TensorDataType>
int partitioned_io_buffer<TensorDataType>::fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) {
  int num_parallel_readers = data_reader->get_num_parallel_readers();

  prof_region_begin("fetch_to_local_matrix", prof_colors[2], false);
  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  buf->m_num_samples_fetched = 0;
  if (this->m_comm->get_rank_in_trainer() < num_parallel_readers && (buf->m_input_buffers[0]->Height() != 0 && buf->m_input_buffers[0]->Width() != 0)) {
    /// Each data reader needs to either have independent / split
    /// data, or take an offset / stride
    if(buf->m_input_buffers.size() == 2) {
      buf->m_num_samples_fetched = (*this->fetch_data_fn)(buf->m_input_buffers[0]->Matrix(), buf->m_input_buffers[1]->Matrix(), buf->m_indices_fetched_per_mb, data_reader);
    }else {
      buf->m_num_samples_fetched = (*this->fetch_data_fn)(buf->m_input_buffers[0]->Matrix(), buf->m_indices_fetched_per_mb, data_reader);
    }
    bool data_valid = (buf->m_num_samples_fetched > 0);
    if(data_valid) {
      //      m_num_data_per_epoch+=num_samples_fetched; /// BVE FIXME need to change how this is shared
    }
  }
  prof_region_end("fetch_to_local_matrix", false);
  return buf->m_num_samples_fetched;
}

template <typename TensorDataType>
void partitioned_io_buffer<TensorDataType>::distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMatrixType& sample, AbsDistMatrixType& response) {
  prof_region_begin("distribute_from_local_matrix", prof_colors[3], false);
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  Copy(*buf->m_input_buffers[0], sample);
  Copy(*buf->m_input_buffers[1], response);
#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    response.Resize(response.Height(), response.Width() /
                    dc::get_number_of_io_partitions());
  }
#endif
  buf->m_num_samples_fetched = 0;
  prof_region_end("distribute_from_local_matrix", false);
  return;
}

template <typename TensorDataType>
void partitioned_io_buffer<TensorDataType>::distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMatrixType& sample) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  Copy(*buf->m_input_buffers[0], sample);
  buf->m_num_samples_fetched = 0;
  return;
}

template <typename TensorDataType>
bool partitioned_io_buffer<TensorDataType>::update_data_set(generic_data_reader *data_reader, execution_mode mode) {
  int num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
  int current_step_in_epoch = data_reader->get_current_step_in_epoch(); // Get the current step before the update function increments it

  (*this->update_data_reader_fn)(true, data_reader);

  if(current_step_in_epoch == (num_iterations_per_epoch - 1)) {
    return true;
  } else {
    return false;
  }
}

template <typename TensorDataType>
void partitioned_io_buffer<TensorDataType>::set_fetch_data_in_background(bool flag, execution_mode mode) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  buf->m_fetch_data_in_background = flag;
}

template <typename TensorDataType>
bool partitioned_io_buffer<TensorDataType>::is_data_fetched_in_background(execution_mode mode) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  return buf->m_fetch_data_in_background;
}

/**
 * Return the sample indices fetched in the current mini-batch.
 */
template <typename TensorDataType>
El::Matrix<El::Int>* partitioned_io_buffer<TensorDataType>::get_sample_indices_fetched_per_mb(execution_mode mode) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  return &(buf->m_indices_fetched_per_mb);
}

template <typename TensorDataType>
int partitioned_io_buffer<TensorDataType>::num_samples_ready(execution_mode mode) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  return buf->m_num_samples_fetched;
}

template <typename TensorDataType>
void partitioned_io_buffer<TensorDataType>::set_data_fetch_future(std::future<void> future, execution_mode mode) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  buf->m_data_fetch_future = std::move(future);
}

template <typename TensorDataType>
std::future<void> partitioned_io_buffer<TensorDataType>::get_data_fetch_future(execution_mode mode) {
  data_buffer<IODataType> *buf = get_data_buffer(mode);
  return std::move(buf->m_data_fetch_future);
}

#define PROTO(T)                          \
  template class partitioned_io_buffer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
