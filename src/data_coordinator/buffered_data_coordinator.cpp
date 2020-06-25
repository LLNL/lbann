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
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/distconv.hpp"

#include <cereal/types/utility.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/atomic.hpp>

namespace lbann {

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::setup(thread_pool& io_thread_pool, int max_mini_batch_size) {
  data_coordinator::setup(io_thread_pool, max_mini_batch_size);
  El::Int num_neurons = get_linearized_data_size();
#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    num_neurons /= dc::get_number_of_io_partitions();
    // TODO: Make sure that TensorDatType is equivalent to the HDF5
    // data reader's data type (float as default).
    // TensorDataType is assumed to be 2-byte integer types such as
    // short or int16_t in an older version.
    // assert_eq(sizeof(TensorDataType), sizeof(short));
    max_mini_batch_size *= dc::get_number_of_io_partitions();
  }
#endif // LBANN_HAS_DISTCONV
  auto data_dims = get_data_dims();

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
  for (const auto& buf_map : m_data_buffers) {
    const data_buffer_map_t& buffer_map = buf_map;
    for (const auto& b : buffer_map) {
      observer_ptr<data_buffer<IODataType>> data_buffer = b.second.get();
      data_buffer->m_input_buffers[input_data_type::SAMPLES]->Resize(num_neurons/*get_linearized_data_size()*/, max_mini_batch_size);
      data_buffer->m_input_buffers[input_data_type::LABELS]->Resize(get_linearized_label_size(), max_mini_batch_size);
    /// The amount of space needed will vary based on input layer type,
    /// but the batch size is the maximum space necessary
      El::Zeros_seq(data_buffer->m_indices_fetched_per_mb, local_mini_batch_size, 1);
    }
  }
}

template <typename TensorDataType>
int buffered_data_coordinator<TensorDataType>::fetch_to_local_matrix(data_buffer_map_t& buffer_map, const execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  int num_parallel_readers = data_reader->get_num_parallel_readers();

  prof_region_begin("fetch_to_local_matrix", prof_colors[2], false);
  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  data_buffer<IODataType>& buf = get_data_buffer(buffer_map, mode);
  buf.m_num_samples_fetched = 0;
  if (this->m_comm->get_rank_in_trainer() < num_parallel_readers
      && (buf.m_input_buffers[input_data_type::SAMPLES]->Height() != 0 && buf.m_input_buffers[input_data_type::SAMPLES]->Width() != 0)) {
    /// Create a map of the local matrices to pass into the data reader
    std::map<input_data_type, CPUMat*> local_input_buffers;
    for(auto& b : buf.m_input_buffers) {
      local_input_buffers[b.first] = static_cast<CPUMat*>(&(b.second->Matrix()));
    }
    /** @brief Each rank will fetch a mini-batch worth of data into it's buffer */
    buf.m_num_samples_fetched = data_reader->fetch(local_input_buffers, buf.m_indices_fetched_per_mb);

    bool data_valid = (buf.m_num_samples_fetched > 0);
    if(data_valid) {
      //      m_num_data_per_epoch+=num_samples_fetched; /// BVE FIXME need to change how this is shared
    }
  }
  prof_region_end("fetch_to_local_matrix", false);
  return buf.m_num_samples_fetched;
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fp_setup_data(data_buffer_map_t& buffer_map, El::Int cur_mini_batch_size) {
#ifdef LBANN_HAS_DISTCONV
  cur_mini_batch_size *= dc::get_number_of_io_partitions();
#endif
  for(auto idt : input_data_type_iterator()) {
    for (auto& buf : buffer_map) {
      buf.second->m_input_buffers[idt]->Resize(buf.second->m_input_buffers[idt]->Height(), cur_mini_batch_size);
    }
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data_in_background(int future_active_buffer, execution_mode mode) {
  int active_buffer_idx = future_active_buffer % m_data_buffers.size();
  data_buffer_map_t& buffer_map = m_data_buffers[active_buffer_idx];
  std::lock_guard<std::mutex> guard(dr_mutex);
  int mini_batch_size = get_current_mini_batch_size(mode);
  fp_setup_data(buffer_map, mini_batch_size);
  fetch_to_local_matrix(buffer_map, mode);
  return;
}

/// Check for each buffer if there is an outstanding fetch request
template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::collect_background_data_fetch(execution_mode mode) {
  for(auto& buffer_map : m_data_buffers) {
    typename data_buffer_map_t::const_iterator it = buffer_map.find(mode);
    if (it != buffer_map.end()) {
      data_buffer<IODataType>& io_buffer = *buffer_map[mode];
      if(io_buffer.is_data_fetched_in_background()) {
        io_buffer.get_data_fetch_future().get();
        io_buffer.set_fetch_data_in_background(false);
      }
    }
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data(execution_mode mode) {

  increment_active_buffer_idx(mode);

  data_buffer<IODataType>& active_buffer = get_active_buffer(mode);

  // If there is no valid data and there is not already a background
  // thread to fetch the data, queue up the background thread
  if(active_buffer.num_samples_ready() == 0 && !active_buffer.is_data_fetched_in_background()) {
    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background, this, this->get_active_buffer_idx(mode), mode));
    active_buffer.set_data_fetch_future(std::move(background_fetch_done));
    active_buffer.set_fetch_data_in_background(true);
  }

  // Wait for the background thread to complete fetching the data
  if(active_buffer.is_data_fetched_in_background()) {
    active_buffer.get_data_fetch_future().get();
    active_buffer.set_fetch_data_in_background(false);
  }

  //  int num_samples_in_batch = 0;
  if(active_buffer.num_samples_ready() > 0) {
    /*num_samples_in_batch = */active_buffer.num_samples_ready();
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
  m_data_set_processed = update_data_set(get_data_reader(mode), mode);

  if(!m_data_set_processed && m_trainer->background_io_activity_allowed()) {
    int next_active_buffer = this->get_active_buffer_idx(mode) + 1;
    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background, this, next_active_buffer, mode));
    data_buffer_map_t& next_io_buffer_map = m_data_buffers[next_active_buffer % m_data_buffers.size()];
    data_buffer<IODataType>& next_io_buffer = get_data_buffer(next_io_buffer_map, mode);
    next_io_buffer.set_data_fetch_future(std::move(background_fetch_done));
    next_io_buffer.set_fetch_data_in_background(true);
  }
  return m_data_set_processed;
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_active_buffer_map(execution_mode mode) -> data_buffer_map_t& {
  return m_data_buffers[this->get_active_buffer_idx(mode) % m_data_buffers.size()];
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_data_buffer(data_buffer_map_t& buffer_map, const execution_mode mode) const -> data_buffer<IODataType>&{
  typename data_buffer_map_t::const_iterator it = buffer_map.find(mode);
  if (it == buffer_map.end()) {
    LBANN_ERROR("Attempting to return a buffer for an invalid execution mode ", to_string(mode));
  }
  return *buffer_map[mode];
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_active_buffer(execution_mode mode) -> data_buffer<IODataType>&{
  data_buffer_map_t& active_buffer_map = get_active_buffer_map(mode);
  return get_data_buffer(active_buffer_map, mode);
}

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
template <typename TensorDataType>
El::Matrix<El::Int>* buffered_data_coordinator<TensorDataType>::get_sample_indices_per_mb(execution_mode mode) {
  auto& active_buffer = get_active_buffer(mode);
  return active_buffer.get_sample_indices_fetched_per_mb();
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::update_data_set(generic_data_reader *data_reader, execution_mode mode) {
  int num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
  int current_step_in_epoch = data_reader->get_current_step_in_epoch(); // Get the current step before the update function increments it

  data_reader->update(true);

  if(current_step_in_epoch == (num_iterations_per_epoch - 1)) {
    return true;
  } else {
    return false;
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::distribute_from_local_matrix(execution_mode mode, AbsDistMatrixType& sample, AbsDistMatrixType& response) {
  prof_region_begin("distribute_from_local_matrix", prof_colors[3], false);
  data_buffer<IODataType>& buf = get_active_buffer(mode);
  El::Copy(*buf.m_input_buffers[input_data_type::SAMPLES], sample);
  El::Copy(*buf.m_input_buffers[input_data_type::LABELS], response);
#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    response.Resize(response.Height(), response.Width() /
                    dc::get_number_of_io_partitions());
  }
#endif
  buf.m_num_samples_fetched = 0;
  prof_region_end("distribute_from_local_matrix", false);
  return;
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::distribute_from_local_matrix(execution_mode mode, AbsDistMatrixType& sample) {
  data_buffer<IODataType>& buf = get_active_buffer(mode);
  El::Copy(*buf.m_input_buffers[input_data_type::SAMPLES], sample);
  buf.m_num_samples_fetched = 0;
  return;
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::save_to_checkpoint_shared(persist& p) const {
  data_coordinator::save_to_checkpoint_shared(p);

  if (this->m_comm->am_trainer_master()) {
    write_cereal_archive<const buffered_data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
  }
  return true;
}

// reload state of IO from a checkpoint
template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::load_from_checkpoint_shared(persist& p) {
  data_coordinator::load_from_checkpoint_shared(p);
  std::string buf;
  if (this->m_comm->am_trainer_master()) {
    read_cereal_archive<buffered_data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
    buf = create_cereal_archive_binary_string<buffered_data_coordinator>(*this);
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  this->m_comm->trainer_broadcast(0, buf);

  if (!this->m_comm->am_trainer_master()) {
    unpack_cereal_archive_binary_string<buffered_data_coordinator>(*this, buf);
  }
  return true;
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::save_to_checkpoint_distributed(persist& p) const {
  data_coordinator::save_to_checkpoint_distributed(p);

  write_cereal_archive<const buffered_data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
  return true;
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::load_from_checkpoint_distributed(persist& p) {
  data_coordinator::load_from_checkpoint_distributed(p);

  read_cereal_archive<buffered_data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
  return true;
}

#define PROTO(T)                     \
  template class buffered_data_coordinator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
