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

#include "lbann/data_coordinator/buffered_data_coordinator.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::setup(thread_pool& io_thread_pool, int max_mini_batch_size) {
  data_coordinator::setup(io_thread_pool, max_mini_batch_size);

  auto data_dims = get_data_dims();
  for (auto& io_buffer : m_io_buffers) {
    io_buffer->setup_data(get_linearized_data_size(),
                          get_linearized_label_size(),
                          max_mini_batch_size);
  }

  for (auto& io_buffer : m_io_buffers) {
    // How much data does the buffer fetch, just the sample or the
    // response as well?
    //    for (int i = 0; i < 2; ++i) {
    for(auto idt : input_data_type_iterator()) {
      io_buffer->fp_setup_data(max_mini_batch_size, idt);
    }
  }

  for (auto io_buffer : m_io_buffers) {
    io_buffer->fetch_data_fn = new fetch_data_functor<IODataType>(data_reader_target_mode::CLASSIFICATION/*target_mode*/);
    io_buffer->update_data_reader_fn = new update_data_reader_functor();
  }
}

template <typename TensorDataType>
int buffered_data_coordinator<TensorDataType>::fetch_to_local_matrix(const execution_mode mode, partitioned_io_buffer<TensorDataType>* io_buffer) {
  generic_data_reader *data_reader = get_data_reader(mode);
  int num_parallel_readers = data_reader->get_num_parallel_readers();

  prof_region_begin("fetch_to_local_matrix", prof_colors[2], false);
  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  data_buffer<IODataType> *buf = io_buffer->get_data_buffer(mode);
  buf->m_num_samples_fetched = 0;
  if (this->m_comm->get_rank_in_trainer() < num_parallel_readers
      && (buf->m_input_buffers[input_data_type::SAMPLES]->Height() != 0 && buf->m_input_buffers[input_data_type::SAMPLES]->Width() != 0)) {
    /// Each data reader needs to either have independent / split
    /// data, or take an offset / stride

    /** @brief Each rank will fetch a mini-batch worth of data into it's buffer */
    buf->m_num_samples_fetched = data_reader->fetch_data(buf->m_input_buffers[input_data_type::SAMPLES]->Matrix(), buf->m_indices_fetched_per_mb);
    if(data_reader->has_labels()) {
      int num_labels_fetched = data_reader->fetch_labels(buf->m_input_buffers[input_data_type::LABELS]->Matrix());
      if(num_labels_fetched != buf->m_num_samples_fetched) {
        LBANN_ERROR("Number of samples: ",
                    std::to_string(buf->m_num_samples_fetched),
                    " does not match the number of labels: ",
                    std::to_string(num_labels_fetched));
      }
    }
    if(data_reader->has_responses()) {
      int num_responses_fetched = data_reader->fetch_responses(buf->m_input_buffers[input_data_type::RESPONSES]->Matrix());
      if(num_responses_fetched != buf->m_num_samples_fetched) {
        LBANN_ERROR("Number of samples: ",
                    std::to_string(buf->m_num_samples_fetched),
                    " does not match the number of responses: ",
                    std::to_string(num_responses_fetched));
      }
    }
    // if(buf->m_input_buffers.size() == 2) {
    //   buf->m_num_samples_fetched = (*this->fetch_data_fn)(buf->m_input_buffers[0]->Matrix(), buf->m_input_buffers[1]->Matrix(), buf->m_indices_fetched_per_mb, data_reader);
    // }else {
    //   buf->m_num_samples_fetched = (*this->fetch_data_fn)(buf->m_input_buffers[0]->Matrix(), buf->m_indices_fetched_per_mb, data_reader);
    // }
    bool data_valid = (buf->m_num_samples_fetched > 0);
    if(data_valid) {
      //      m_num_data_per_epoch+=num_samples_fetched; /// BVE FIXME need to change how this is shared
    }
  }
  prof_region_end("fetch_to_local_matrix", false);
  return buf->m_num_samples_fetched;
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data_in_background(int future_active_buffer, execution_mode mode) {
  int active_buffer = future_active_buffer % m_io_buffers.size();
  partitioned_io_buffer<TensorDataType>* io_buffer = m_io_buffers[active_buffer];
  std::lock_guard<std::mutex> guard(dr_mutex);
  setup_next_io_buffer(io_buffer, mode);
  fetch_to_local_matrix(mode, io_buffer);
  return;
}

/// Check for each buffer if there is an outstanding fetch request
template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::collect_background_data_fetch(execution_mode mode) {
  for(auto& io_buffer : m_io_buffers) {
    if(io_buffer->is_data_fetched_in_background(mode)) {
      io_buffer->get_data_fetch_future(mode).get();
      io_buffer->set_fetch_data_in_background(false, mode);
    }
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::setup_next_io_buffer(partitioned_io_buffer<TensorDataType>* io_buffer, execution_mode mode) {
  int mini_batch_size = get_current_mini_batch_size(mode);
  //  for (int i = 0; i < 2/*this->get_num_children()*/; ++i) {
  for(auto idt : input_data_type_iterator()) {
    io_buffer->fp_setup_data(mini_batch_size, idt);
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data(execution_mode mode) {

  increment_active_buffer_idx(mode);

  partitioned_io_buffer<TensorDataType>* io_buffer = m_io_buffers[this->get_active_buffer_idx(mode) % m_io_buffers.size()];

  // If there is no valid data and there is not already a background
  // thread to fetch the data, queue up the background thread
  if(io_buffer->num_samples_ready(mode) == 0 && !io_buffer->is_data_fetched_in_background(mode)) {
    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background, this, this->get_active_buffer_idx(mode), mode));
    io_buffer->set_data_fetch_future(std::move(background_fetch_done), mode);
    io_buffer->set_fetch_data_in_background(true, mode);
  }

  // Wait for the background thread to complete fetching the data
  if(io_buffer->is_data_fetched_in_background(mode)) {
    io_buffer->get_data_fetch_future(mode).get();
    io_buffer->set_fetch_data_in_background(false, mode);
  }

  //  int num_samples_in_batch = 0;
  if(io_buffer->num_samples_ready(mode) > 0) {
    /*num_samples_in_batch = */io_buffer->num_samples_ready(mode);
  }else {
      if(!get_data_reader(mode)->position_is_overrun()) {
        std::stringstream err;
        err << "I/O buffer does not contain valid samples ("/*<< num_samples_in_batch << ")"*/;
        LBANN_ERROR(err.str());
      }
  }
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::epoch_complete(execution_mode mode) {
  partitioned_io_buffer<TensorDataType>* io_buffer = m_io_buffers[this->get_active_buffer_idx(mode) % m_io_buffers.size()];

  m_data_set_processed = io_buffer->update_data_set(get_data_reader(mode), mode);

  if(!m_data_set_processed && m_trainer->background_io_activity_allowed()) {
    int next_active_buffer = this->get_active_buffer_idx(mode) + 1;
    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background, this, next_active_buffer, mode));
    partitioned_io_buffer<TensorDataType>* next_io_buffer = m_io_buffers[next_active_buffer % m_io_buffers.size()];
    next_io_buffer->set_data_fetch_future(std::move(background_fetch_done), mode);
    next_io_buffer->set_fetch_data_in_background(true, mode);
  }
  return m_data_set_processed;
}

template <typename TensorDataType>
partitioned_io_buffer<TensorDataType>* buffered_data_coordinator<TensorDataType>::get_active_buffer(execution_mode mode) {
  return dynamic_cast<partitioned_io_buffer<TensorDataType>*>(m_io_buffers[this->get_active_buffer_idx(mode) % m_io_buffers.size()]);
}

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
template <typename TensorDataType>
El::Matrix<El::Int>* buffered_data_coordinator<TensorDataType>::get_sample_indices_per_mb(execution_mode mode) {
  partitioned_io_buffer<TensorDataType>* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];
  return io_buffer->get_sample_indices_fetched_per_mb(mode);
}

#define PROTO(T)                     \
  template class buffered_data_coordinator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
