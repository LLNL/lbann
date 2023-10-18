////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/buffered_data_coordinator_impl.hpp"
#include "lbann/data_coordinator/data_packer.hpp"
#include "lbann/data_coordinator/io_data_buffer_impl.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/tensor_impl.hpp"

namespace lbann {

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::register_active_data_field(
  data_field_type const& data_field,
  std::vector<El::Int> const& data_field_dim_map)

{
  data_coordinator::register_active_data_field(data_field, data_field_dim_map);
  for (const auto& buf_map : m_data_buffers) {
    const data_buffer_map_t& buffer_map = buf_map;
    for (auto& [mode, buffer] : buffer_map) {
      buffer->initialize_buffer_for_data_field(data_field, m_comm);
    }
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::setup_data_fields(
  int max_mini_batch_size)
{
  if (m_active_data_fields.size() == 0) {
    LBANN_ERROR(
      "Models have not registered data fields with the data coordinator");
  }

#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    // TODO: Make sure that TensorDatType is equivalent to the HDF5
    // data reader's data type (float as default).
    // TensorDataType is assumed to be 2-byte integer types such as
    // short or int16_t in an older version.
    // assert_eq(sizeof(TensorDataType), sizeof(short));
    max_mini_batch_size *= dc::get_number_of_io_partitions();
  }
#endif // LBANN_HAS_DISTCONV

  /// @todo BVE This is where we are going to have to limit how many
  /// ranks are participating in I/O
  El::Int local_mini_batch_size =
    max_mini_batch_size / this->m_comm->get_procs_per_trainer();
  El::Int partial_mini_batch_size =
    max_mini_batch_size % this->m_comm->get_procs_per_trainer();
  if (partial_mini_batch_size > 0 &&
      this->m_comm->get_rank_in_trainer() < partial_mini_batch_size) {
    local_mini_batch_size++;
  }

#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    // Manually resize buffers for CosmoFlow data tensors
    El::Int linearized_size = get_linearized_data_size();
    linearized_size /= dc::get_number_of_io_partitions();
    for (const auto& buf_map : m_data_buffers) {
      const data_buffer_map_t& buffer_map = buf_map;
      for (const auto& [mode, data_buffer] : buffer_map) {
        auto& input_buffers = data_buffer->m_input_buffers;
        if (input_buffers.count(INPUT_DATA_TYPE_SAMPLES) > 0 &&
            input_buffers[INPUT_DATA_TYPE_SAMPLES]->IsEmpty()) {
          input_buffers[INPUT_DATA_TYPE_SAMPLES]->Resize(linearized_size,
                                                         max_mini_batch_size);
          El::Zeros_seq(data_buffer->m_indices_fetched_per_mb,
                        local_mini_batch_size,
                        1);
        }
      }
    }
  }
#endif // LBANN_HAS_DISTCONV

  // Check to see if there are any data fields with unallocated buffers
  for (auto& data_field : m_active_data_fields) {
    for (const auto& buf_map : m_data_buffers) {
      const data_buffer_map_t& buffer_map = buf_map;
      for (const auto& [mode, data_buffer] : buffer_map) {
        auto& phase_io_buffer = data_buffer->m_input_buffers[data_field];
        // Check to see if a buffer has already been allocated.  If
        // not, resize and zero it
        if (phase_io_buffer->IsEmpty() || phase_io_buffer->Width() == 0 ||
            phase_io_buffer->Height() == 0) {
          El::Int linearized_size = get_linearized_size(data_field);
          if (linearized_size == -1) {
            LBANN_ERROR("Invalid value for the linearized size of data field ",
                        data_field);
          }
          data_buffer->m_input_buffers[data_field]->Resize(linearized_size,
                                                           max_mini_batch_size);

          /// The amount of space needed will vary based on input layer type,
          /// but the batch size is the maximum space necessary
          El::Zeros_seq(data_buffer->m_indices_fetched_per_mb,
                        local_mini_batch_size,
                        1);
        }
      }
    }
  }
}

template <typename TensorDataType>
int buffered_data_coordinator<TensorDataType>::fetch_to_local_matrix(
  const execution_mode mode,
  data_buffer<IODataType>& buf,
  El::Int loaded_mini_batch_size,
  El::Int relative_base_position,
  int buffer_id)
{
  generic_data_reader* dr = get_data_reader(mode);
  int num_parallel_readers = dr->get_num_parallel_readers();

  std::string prof_title =
    ("fetch_to_local_matrix " + std::to_string(buffer_id));
  prof_region_begin(prof_title.c_str(), prof_colors[2], false);
  /// Coordinate all available readers so that they perform I/O in the same step
  /// Check to make sure that the local matrix has space for data

  LBANN_MSG(to_string(mode),
            " mode: fetch to local matrix has a current mini-batch size = ",
            loaded_mini_batch_size);

  buf.m_num_samples_fetched = 0;
  /// BVE FIXME change the guard
  // Check to see if this rank has local space for the current mini-batch
  if (this->m_comm->get_rank_in_trainer() < num_parallel_readers &&
      (buf.m_input_buffers[INPUT_DATA_TYPE_SAMPLES]->LocalHeight() != 0 &&
       buf.m_input_buffers[INPUT_DATA_TYPE_SAMPLES]->LocalWidth() != 0)) {
    /// Create a map of the local matrices to pass into the data reader
    std::map<data_field_type, CPUMat*> local_input_buffers;
    for (auto& b : buf.m_input_buffers) {
      local_input_buffers[b.first] =
        static_cast<CPUMat*>(&(b.second->Matrix()));
    }

    // Compute the size of the current local mini-batch
    const int end_pos = std::min(
      static_cast<size_t>(relative_base_position + loaded_mini_batch_size),
      dr->m_shuffled_indices.size());
    const int local_mini_batch_size = std::min(
      El::Int{((end_pos - relative_base_position) + dr->m_sample_stride - 1) /
              dr->m_sample_stride},
      local_input_buffers[INPUT_DATA_TYPE_SAMPLES]->Width());

    /** @brief Each rank will fetch a mini-batch worth of data into its buffer
     */
    if (dr->has_conduit_output()) {
      std::vector<conduit::Node> samples(local_mini_batch_size);
      buf.m_num_samples_fetched = dr->fetch(samples,
                                            buf.m_indices_fetched_per_mb,
                                            relative_base_position,
                                            local_mini_batch_size);
      data_packer::extract_data_fields_from_samples(samples,
                                                    local_input_buffers);
    }
    else {
      buf.m_num_samples_fetched = dr->fetch(local_input_buffers,
                                            buf.m_indices_fetched_per_mb,
                                            relative_base_position,
                                            local_mini_batch_size);
    }

    bool data_valid = (buf.m_num_samples_fetched > 0);
    if (data_valid) {
      //      m_num_data_per_epoch+=num_samples_fetched; /// BVE FIXME need to
      //      change how this is shared
    }
  }
  prof_region_end(prof_title.c_str(), false);
  return buf.m_num_samples_fetched;
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fp_setup_data(
  data_buffer<IODataType>& buffer,
  El::Int cur_mini_batch_size)
{
#ifdef LBANN_HAS_DISTCONV
  cur_mini_batch_size *= dc::get_number_of_io_partitions();
#endif
  for (auto& [data_field, mat] : buffer.m_input_buffers) {
    mat->Resize(mat->Height(), cur_mini_batch_size);
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data_in_background(
  int future_active_buffer,
  data_buffer<IODataType>& buf,
  El::Int loaded_mini_batch_size,
  El::Int relative_base_position,
  execution_mode mode)
{
  int active_buffer_idx = future_active_buffer % m_data_buffers.size();
  std::lock_guard<std::mutex> guard(dr_mutex);
  fetch_to_local_matrix(mode,
                        buf,
                        loaded_mini_batch_size,
                        relative_base_position,
                        active_buffer_idx);
  return;
}

template <typename TensorDataType>
int buffered_data_coordinator<TensorDataType>::get_current_mini_batch_size(
  execution_mode mode) const
{
  // Get the mini-batch size from the active buffer
  int buffer_id = this->get_active_buffer_idx(mode) % m_data_buffers.size();
  return m_current_mini_batch_size.at(buffer_id).at(mode);
}

/// Check for each buffer if there is an outstanding fetch request
template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::collect_background_data_fetch(
  execution_mode mode)
{
  for (auto& buffer_map : m_data_buffers) {
    typename data_buffer_map_t::const_iterator it = buffer_map.find(mode);
    if (it != buffer_map.end()) {
      data_buffer<IODataType>& io_buffer = *buffer_map[mode];
      if (io_buffer.is_background_fetching_in_progress()) {
        io_buffer.get_data_fetch_future().get();
        io_buffer.set_background_fetching_in_progress(false);
      }
    }
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_active_batch_synchronous(
  execution_mode mode)
{
  int idx = this->get_active_buffer_idx(mode);
  int buffer_id = idx % m_data_buffers.size();
  data_buffer<IODataType>& active_buffer = get_active_buffer(mode);

  LBANN_MSG("fetching active batch for buffer ",
            std::to_string(get_active_buffer_idx(mode)));

  generic_data_reader* dr = get_data_reader(mode);
  //************************************************************************
  // Get the current mini-batch from the data reader
  El::Int loaded_mini_batch_size = dr->get_loaded_mini_batch_size();
  LBANN_MSG(to_string(mode),
            " fetch active data ",
            " for index ",
            idx,
            " thinks that for iteration ",
            "current ",
            dr->get_current_mini_batch_index(),
            " index and mb size ",
            dr->get_current_mini_batch_size(),
            " loaded ",
            dr->get_loaded_mini_batch_index(),
            " (",
            dr->get_loaded_mini_batch_index(),
            " +",
            dr->get_iteration_stride(),
            ")",
            " the loaded mini-batch size will be ",
            loaded_mini_batch_size);

  // If the projected mini-batch will not run past the epoch boundary
  // If there is no valid data and there is not already a background
  // thread to fetch the data, queue up the background thread
  if (loaded_mini_batch_size > 0 && active_buffer.num_samples_ready() == 0 &&
      !active_buffer.is_background_fetching_in_progress()) {
    // Start data store exchange if necessary (this should be moved
    // earlier as a future optimization)
    get_data_reader(mode)->start_data_store_mini_batch_exchange();
    // Finish data store exchange before accessing samples
    get_data_reader(mode)->finish_data_store_mini_batch_exchange();

    // ********************************************************************************
    // BVE Get all of the state now about what is being fetched

    // Set the size for the I/O buffers
    fp_setup_data(active_buffer, loaded_mini_batch_size);
    // Store the size of the current mini-batch so that others can obtain it
    // without worrying about where the data reader is currently at.
    m_current_mini_batch_size[buffer_id][mode] = loaded_mini_batch_size;
    El::Int relative_base_position = dr->m_current_pos;

    // ********************************************************************************
    // BVE

    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background,
                this,
                idx,
                std::ref(active_buffer),
                loaded_mini_batch_size,
                relative_base_position,
                mode));
    active_buffer.set_data_fetch_future(std::move(background_fetch_done));
    active_buffer.set_background_fetching_in_progress(true);
  }

  if (loaded_mini_batch_size == 0) {
    LBANN_MSG("I have skipped a fetch active batch");
  }
  // Wait for the background thread to complete fetching the same data
  if (active_buffer.is_background_fetching_in_progress()) {
    active_buffer.get_data_fetch_future().get();
    active_buffer.set_background_fetching_in_progress(false);
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data(execution_mode mode)
{
  data_buffer<IODataType>& current_buffer = get_active_buffer(mode);
  data_buffer<IODataType>& next_buffer = get_next_buffer(mode);
  auto next_buffer_idx = this->get_next_buffer_idx(mode);
  auto next_buffer_id = next_buffer_idx % m_data_buffers.size();

  LBANN_MSG("fetching next batch for buffer ",
            std::to_string(get_next_buffer_idx(mode)),
            " active buffer is ",
            std::to_string(get_active_buffer_idx(mode)));

  // Wait for the background thread to complete fetching the data
  if (current_buffer.is_background_fetching_in_progress()) {
    current_buffer.get_data_fetch_future().get();
    current_buffer.set_background_fetching_in_progress(false);
  }

  generic_data_reader* dr = get_data_reader(mode);
  //************************************************************************
  // Get the next mini-batchs size from the data reader
  El::Int next_loaded_mini_batch_size = dr->get_next_loaded_mini_batch_size();
  LBANN_MSG(to_string(mode),
            " fetch data thinks that for iteration ",
            " for index ",
            this->get_next_buffer_idx(mode),
            " thinks that for iteration ",
            "current ",
            dr->get_current_mini_batch_index(),
            " index and mb size ",
            dr->get_current_mini_batch_size(),
            " loaded ",
            dr->get_loaded_mini_batch_index() + dr->get_iteration_stride(),
            " (",
            dr->get_loaded_mini_batch_index(),
            " + ",
            dr->get_iteration_stride(),
            ")",
            ", the next mini-batch size will be ",
            next_loaded_mini_batch_size);
  // If the projected mini-batch will not run past the epoch boundary
  // If there is no valid data and there is not already a background
  // thread to fetch the data, queue up the background thread
  // NOTE: This may fetch one more sample after the last epoch has completed.
  if (next_loaded_mini_batch_size > 0 && next_buffer.num_samples_ready() == 0 &&
      !next_buffer.is_background_fetching_in_progress()) {
    // BVE FIXME I don't think that the future data fetch should do
    // this.

    // Start data store exchange if necessary (this should be moved
    // earlier as a future optimization)
    get_data_reader(mode)->start_data_store_mini_batch_exchange();
    // Finish data store exchange before accessing samples
    get_data_reader(mode)->finish_data_store_mini_batch_exchange();

    // ********************************************************************************
    // BVE Get all of the state now about what is being fetched

    // Set the size for the I/O buffers
    fp_setup_data(next_buffer, next_loaded_mini_batch_size);
    // Store the size of the current mini-batch so that others can obtain it
    // without worrying about where the data reader is currently at.
    m_current_mini_batch_size[next_buffer_id][mode] =
      next_loaded_mini_batch_size;
    El::Int relative_base_position = dr->get_next_position();
    LBANN_MSG(
      "Fetching local samples for a future buffer with new base position",
      relative_base_position,
      " versus current position ",
      dr->m_current_pos);

    // ********************************************************************************
    // BVE

    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background,
                this,
                next_buffer_idx,
                std::ref(next_buffer),
                next_loaded_mini_batch_size,
                relative_base_position,
                mode));
    next_buffer.set_data_fetch_future(std::move(background_fetch_done));
    next_buffer.set_background_fetching_in_progress(true);
  }

  if (next_loaded_mini_batch_size == 0) {
    LBANN_MSG("I will skipped a fetch batch");
  }
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::ready_for_next_fetch(
  execution_mode mode)
{
  LBANN_MSG("ready for next fetch is cleaning up buffer ",
            std::to_string(get_active_buffer_idx(mode)));
  // Check to see if the data from the sample was actually consumed
  data_buffer<IODataType>& active_buffer = get_active_buffer(mode);
  // for (const auto& buf_map : m_data_buffers) {
  //   const data_buffer_map_t& buffer_map = buf_map;
  for (auto& [field, count] :
       active_buffer.m_num_samples_per_field_distributed) {
    LBANN_MSG("ready for next fetch: for field ",
              field,
              " was distributed ",
              count,
              " samples");
    if (count != active_buffer.m_num_samples_fetched) {
      LBANN_MSG("ready for next fetch: UNUSED field ",
                field,
                " was not distributed ",
                count,
                " samples but ",
                active_buffer.m_num_samples_fetched,
                " available");
    }
  }
  active_buffer.m_num_samples_fetched = 0;

  // Make sure to update the data reader before incrementing the
  // buffer index
  auto is_epoch_complete = this->update_data_reader(mode);
  if (is_epoch_complete) {
    data_buffer<IODataType>& active_buffer = get_active_buffer(mode);
    // Wait for the background thread to complete fetching the same data
    if (active_buffer.is_background_fetching_in_progress()) {
      LBANN_MSG("ready_for_next_fetch has to wait for the data.");
      active_buffer.get_data_fetch_future().get();
      active_buffer.set_background_fetching_in_progress(false);
    }
    // BVE FIXME do we need to collect all background data before
    // going to the next phase
  }
  this->increment_active_buffer_idx(mode);
  LBANN_MSG("ready for next fetch just flipped the index and it is now ",
            std::to_string(get_active_buffer_idx(mode)));
  return is_epoch_complete;
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::update_data_reader(
  execution_mode mode)
{
  // Use the size of the mini-batch to update the data reader how much
  // to advance
  int num_samples_in_batch = get_current_mini_batch_size(mode);
  // BVE When we finish the epoch we can increment the number of
  // samples that have been processed
  update_num_samples_processed(mode, num_samples_in_batch);

  // Returns true if the epoch will be complete
  return update_data_set(get_data_reader(mode), mode);
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_active_buffer_map(
  execution_mode mode) const -> const data_buffer_map_t&
{
  return m_data_buffers.at(get_active_buffer_idx(mode) % m_data_buffers.size());
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_active_buffer_map(
  execution_mode mode) -> data_buffer_map_t&
{
  return m_data_buffers[get_active_buffer_idx(mode) % m_data_buffers.size()];
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_data_buffer(
  const data_buffer_map_t& buffer_map,
  const execution_mode mode) const -> const data_buffer<IODataType>&
{
  typename data_buffer_map_t::const_iterator it = buffer_map.find(mode);
  if (it == buffer_map.end()) {
    LBANN_ERROR("Attempting to return a buffer for an invalid execution mode ",
                to_string(mode));
  }
  return *buffer_map.at(mode);
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_data_buffer(
  data_buffer_map_t& buffer_map,
  const execution_mode mode) -> data_buffer<IODataType>&
{
  typename data_buffer_map_t::const_iterator it = buffer_map.find(mode);
  if (it == buffer_map.end()) {
    LBANN_ERROR("Attempting to return a buffer for an invalid execution mode ",
                to_string(mode));
  }
  return *buffer_map[mode];
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_active_buffer(
  execution_mode mode) const -> const data_buffer<IODataType>&
{
  const data_buffer_map_t& active_buffer_map = get_active_buffer_map(mode);
  return get_data_buffer(active_buffer_map, mode);
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_active_buffer(
  execution_mode mode) -> data_buffer<IODataType>&
{
  data_buffer_map_t& active_buffer_map = get_active_buffer_map(mode);
  return get_data_buffer(active_buffer_map, mode);
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_next_buffer(
  execution_mode mode) const -> const data_buffer<IODataType>&
{
  const data_buffer_map_t& next_buffer_map =
    m_data_buffers.at(get_next_buffer_idx(mode) % m_data_buffers.size());
  return get_data_buffer(next_buffer_map, mode);
}

template <typename TensorDataType>
auto buffered_data_coordinator<TensorDataType>::get_next_buffer(
  execution_mode mode) -> data_buffer<IODataType>&
{
  data_buffer_map_t& next_buffer_map =
    m_data_buffers[get_next_buffer_idx(mode) % m_data_buffers.size()];
  return get_data_buffer(next_buffer_map, mode);
}

/**
 * Return the sample indices fetched in the current mini-batch.
 */
template <typename TensorDataType>
const El::Matrix<El::Int>*
buffered_data_coordinator<TensorDataType>::get_sample_indices_per_mb(
  execution_mode mode) const
{
  const auto& active_buffer = get_active_buffer(mode);
  return active_buffer.get_sample_indices_fetched_per_mb();
}

template <typename TensorDataType>
El::Matrix<El::Int>*
buffered_data_coordinator<TensorDataType>::get_sample_indices_per_mb(
  execution_mode mode)
{
  return const_cast<El::Matrix<El::Int>*>(
    static_cast<const buffered_data_coordinator&>(*this)
      .get_sample_indices_per_mb(mode));
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::update_data_set(
  generic_data_reader* data_reader,
  execution_mode mode)
{
  int num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
  int current_step_in_epoch =
    data_reader->get_current_step_in_epoch(); // Get the current step before the
                                              // update function increments it

  data_reader->update(true);

  if (current_step_in_epoch == (num_iterations_per_epoch - 1)) {
    return true;
  }
  else {
    return false;
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::distribute_from_local_matrix(
  execution_mode mode,
  data_field_type const data_field,
  AbsDistMatrixType& input_buffer)
{
  std::string prof_title = ("distribute_from_local_matrix " +
                            std::to_string(get_active_buffer_idx(mode)));
  prof_region_begin(prof_title.c_str(), prof_colors[3], false);
  LBANN_MSG("distributed from local matrix is looking at buffer ",
            std::to_string(get_active_buffer_idx(mode)));
  data_buffer<IODataType>& buf = get_active_buffer(mode);
  // Wait for the background thread to complete fetching the same data
  if (buf.is_background_fetching_in_progress()) {
    LBANN_MSG("distribute from local matrix has to wait for the data.");
    buf.get_data_fetch_future().get();
    buf.set_background_fetching_in_progress(false);
  }
  if (buf.m_input_buffers.find(data_field) == buf.m_input_buffers.end()) {
    LBANN_ERROR("Unknown data_field_type value requested: " + data_field);
  }
  view_or_copy_tensor(*buf.m_input_buffers[data_field], input_buffer);
#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled() &&
      data_field == INPUT_DATA_TYPE_RESPONSES) {
    El::Int new_width =
      input_buffer.Width() / dc::get_number_of_io_partitions();
    if (input_buffer.Viewing()) {
      El::LockedView(input_buffer, input_buffer, El::ALL, El::IR(0, new_width));
    }
    else {
      input_buffer.Resize(input_buffer.Height(), new_width);
    }
  }
#endif
  // Track if the data field has been pulled out of the buffer
  buf.m_num_samples_per_field_distributed[data_field] =
    buf.m_num_samples_fetched;
  // BVE FIXME I don't think that we should reset this field right now.
  //  buf.m_num_samples_fetched = 0;
  prof_region_end(prof_title.c_str(), false);
  return;
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::save_to_checkpoint_shared(
  persist& p) const
{
  data_coordinator::save_to_checkpoint_shared(p);

  if (this->m_comm->am_trainer_master()) {
    write_cereal_archive<const buffered_data_coordinator>(
      *this,
      p,
      execution_mode::training,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
      "_dc.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
      "_dc.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  return true;
}

// reload state of IO from a checkpoint
template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::load_from_checkpoint_shared(
  persist& p)
{
  data_coordinator::load_from_checkpoint_shared(p);
  std::string buf;
  if (this->m_comm->am_trainer_master()) {
    read_cereal_archive<buffered_data_coordinator>(*this,
                                                   p,
                                                   execution_mode::training,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                                   "_dc.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                                   "_dc.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
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
bool buffered_data_coordinator<TensorDataType>::save_to_checkpoint_distributed(
  persist& p) const
{
  data_coordinator::save_to_checkpoint_distributed(p);

  write_cereal_archive<const buffered_data_coordinator>(
    *this,
    p,
    execution_mode::training,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
    "_dc.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
    "_dc.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

template <typename TensorDataType>
bool buffered_data_coordinator<
  TensorDataType>::load_from_checkpoint_distributed(persist& p)
{
  data_coordinator::load_from_checkpoint_distributed(p);

  read_cereal_archive<buffered_data_coordinator>(*this,
                                                 p,
                                                 execution_mode::training,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                                 "_dc.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                                 "_dc.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

#define PROTO(T) template class buffered_data_coordinator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
