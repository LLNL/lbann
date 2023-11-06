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
#include "lbann/data_ingestion/infrastructure/data_packer.hpp"
#include "lbann/data_ingestion/infrastructure/io_data_buffer_impl.hpp"
#include "lbann/data_ingestion/readers/data_reader.hpp"
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
  uint64_t max_mini_batch_size)
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

  uint64_t local_mini_batch_size =
    max_mini_batch_size / this->m_comm->get_procs_per_trainer();
  uint64_t partial_mini_batch_size =
    max_mini_batch_size % this->m_comm->get_procs_per_trainer();
  if (partial_mini_batch_size > 0 &&
      static_cast<uint64_t>(this->m_comm->get_rank_in_trainer()) <
        partial_mini_batch_size) {
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
      for (auto& [mode, data_buffer] : buffer_map) {
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
          auto& indices_fetched_per_mb = data_buffer->m_indices_fetched_per_mb;
          El::Zeros_seq(indices_fetched_per_mb, local_mini_batch_size, 1);
        }
      }
    }
  }
}

template <typename TensorDataType>
int buffered_data_coordinator<TensorDataType>::fetch_to_local_matrix(
  const execution_mode mode,
  data_buffer<IODataType>& buf,
  uint64_t loaded_mini_batch_size,
  uint64_t relative_base_position,
  int buffer_id)
{
  generic_data_reader* dr = get_data_reader(mode);
  dataset& ds = get_dataset(mode);

  std::string prof_title =
    ("fetch_to_local_matrix " + std::to_string(buffer_id));
  prof_region_begin(prof_title.c_str(), prof_colors[2], false);
  /// Coordinate all available readers so that they perform I/O in the same step
  /// Check to make sure that the local matrix has space for data

  buf.m_num_samples_fetched = 0;
  // Check to see if this rank has local space for the current mini-batch
  if (buf.m_input_buffers[INPUT_DATA_TYPE_SAMPLES]->LocalHeight() != 0 &&
      buf.m_input_buffers[INPUT_DATA_TYPE_SAMPLES]->LocalWidth() != 0) {
    /// Create a map of the local matrices to pass into the data reader
    std::map<data_field_type, CPUMat*> local_input_buffers;
    for (auto& b : buf.m_input_buffers) {
      local_input_buffers[b.first] =
        static_cast<CPUMat*>(&(b.second->Matrix()));
    }

    // Compute the size of the current local mini-batch
    const uint64_t end_pos =
      std::min(relative_base_position + loaded_mini_batch_size,
               dr->m_shuffled_indices.size());
    const uint64_t local_mini_batch_size = std::min(
      ((end_pos - relative_base_position) + ds.get_sample_stride() - 1) /
        ds.get_sample_stride(),
      static_cast<uint64_t>(
        local_input_buffers[INPUT_DATA_TYPE_SAMPLES]->Width()));

    /** @brief Each rank will fetch a mini-batch worth of data into its buffer
     */
    if (dr->has_conduit_output()) {
      std::vector<conduit::Node> samples(local_mini_batch_size);
      buf.m_num_samples_fetched = dr->fetch(samples,
                                            buf.m_indices_fetched_per_mb,
                                            relative_base_position,
                                            ds.get_sample_stride(),
                                            local_mini_batch_size,
                                            mode);
      data_packer::extract_data_fields_from_samples(samples,
                                                    local_input_buffers);
    }
    else {
      buf.m_num_samples_fetched = dr->fetch(local_input_buffers,
                                            buf.m_indices_fetched_per_mb,
                                            relative_base_position,
                                            ds.get_sample_stride(),
                                            local_mini_batch_size,
                                            mode);
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
  uint64_t cur_mini_batch_size)
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
  uint64_t loaded_mini_batch_size,
  uint64_t relative_base_position,
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
uint64_t buffered_data_coordinator<TensorDataType>::get_current_mini_batch_size(
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
  dataset& ds = get_dataset(mode);
  //************************************************************************
  // Get the current mini-batch from the data reader
  uint64_t loaded_mini_batch_size = ds.get_current_mini_batch_size();

  // If there is no valid data and there is not already a background
  // thread to fetch the data, queue up the background thread
  if (loaded_mini_batch_size > 0 && active_buffer.num_samples_ready() == 0 &&
      !active_buffer.is_background_fetching_in_progress()) {
    // Store the size of the current mini-batch so that others can obtain it
    // without worrying about where the data reader is currently at.
    m_current_mini_batch_size[buffer_id][mode] = loaded_mini_batch_size;
    uint64_t relative_base_position = ds.get_position();

    // Start data store exchange if necessary (this should be moved
    // earlier as a future optimization)
    get_data_reader(mode)->start_data_store_mini_batch_exchange(
      // Use the relative position of the mini-batch (adjusted for rank)
      relative_base_position - ds.get_base_offset(),
      loaded_mini_batch_size,
      ds.at_new_epoch());
    // Finish data store exchange before accessing samples
    get_data_reader(mode)->finish_data_store_mini_batch_exchange();

    // Set the size for the I/O buffers
    fp_setup_data(active_buffer, loaded_mini_batch_size);

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

  // Wait for the background thread to complete fetching the same data
  if (active_buffer.is_background_fetching_in_progress()) {
    active_buffer.get_data_fetch_future().get();
    active_buffer.set_background_fetching_in_progress(false);
  }
}

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::fetch_data_asynchronous(
  execution_mode mode)
{
  data_buffer<IODataType>& current_buffer = get_active_buffer(mode);
  data_buffer<IODataType>& next_buffer = get_next_buffer(mode);
  auto next_buffer_idx = this->get_next_buffer_idx(mode);
  auto next_buffer_id = next_buffer_idx % m_data_buffers.size();

  // Wait for the background thread to complete fetching the data
  if (current_buffer.is_background_fetching_in_progress()) {
    current_buffer.get_data_fetch_future().get();
    current_buffer.set_background_fetching_in_progress(false);
  }

  dataset& ds = get_dataset(mode);
  //************************************************************************
  // Get the next mini-batchs size from the data reader
  uint64_t next_mini_batch_size = ds.get_next_mini_batch_size();

  // If there is no valid data and there is not already a background
  // thread to fetch the data, queue up the background thread
  if (next_mini_batch_size > 0 && next_buffer.num_samples_ready() == 0 &&
      !next_buffer.is_background_fetching_in_progress()) {
    // Store the size of the current mini-batch so that others can obtain it
    // without worrying about where the data reader is currently at.
    m_current_mini_batch_size[next_buffer_id][mode] = next_mini_batch_size;
    uint64_t relative_base_position = ds.get_next_position();

    // Start data store exchange if necessary (this should be moved
    // earlier as a future optimization)
    get_data_reader(mode)->start_data_store_mini_batch_exchange(
      // Use the relative position of the mini-batch (adjusted for rank)
      relative_base_position - ds.get_base_offset(),
      next_mini_batch_size,
      ds.at_new_epoch());
    // Finish data store exchange before accessing samples
    get_data_reader(mode)->finish_data_store_mini_batch_exchange();

    // Set the size for the I/O buffers
    fp_setup_data(next_buffer, next_mini_batch_size);

    std::future<void> background_fetch_done = get_io_thread_pool().submit_job(
      std::bind(&buffered_data_coordinator::fetch_data_in_background,
                this,
                next_buffer_idx,
                std::ref(next_buffer),
                next_mini_batch_size,
                relative_base_position,
                mode));
    next_buffer.set_data_fetch_future(std::move(background_fetch_done));
    next_buffer.set_background_fetching_in_progress(true);
  }
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::ready_for_next_fetch(
  execution_mode mode)
{
  // Check to see if the data from the sample was actually consumed
  data_buffer<IODataType>& active_buffer = get_active_buffer(mode);
  for (auto& [field, count] :
       active_buffer.m_num_samples_per_field_distributed) {
    if (count != active_buffer.m_num_samples_fetched) {
      LBANN_WARNING("ready for next fetch: UNUSED field ",
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
    // Wait for the background thread to complete fetching the same data
    if (active_buffer.is_background_fetching_in_progress()) {
      LBANN_WARNING("ready_for_next_fetch has to wait for the data.");
      active_buffer.get_data_fetch_future().get();
      active_buffer.set_background_fetching_in_progress(false);
    }
  }
  this->increment_active_buffer_idx(mode);
  return is_epoch_complete;
}

template <typename TensorDataType>
bool buffered_data_coordinator<TensorDataType>::update_data_reader(
  execution_mode mode)
{
  // Use the size of the mini-batch to update the data reader how much
  // to advance
  uint64_t num_samples_in_batch = get_current_mini_batch_size(mode);
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
  dataset& ds = get_dataset(mode);
  uint64_t num_iterations_per_epoch = ds.get_num_iterations_per_epoch();
  uint64_t current_step_in_epoch =
    ds.get_current_step_in_epoch(); // Get the current step before the
                                    // update function increments it

  bool epoch_complete = ds.update();
  data_reader->update(epoch_complete /*ds.get_next_position(), */);

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
  data_buffer<IODataType>& buf = get_active_buffer(mode);
  // Wait for the background thread to complete fetching the same data
  if (buf.is_background_fetching_in_progress()) {
    LBANN_WARNING("distribute from local matrix has to wait for the data.");
    buf.get_data_fetch_future().get();
    buf.set_background_fetching_in_progress(false);
  }
  if (buf.m_input_buffers.find(data_field) == buf.m_input_buffers.end()) {
    LBANN_ERROR("Unknown data_field_type value requested: " + data_field);
  }
  view_or_copy_tensor(*buf.m_input_buffers[data_field], input_buffer, false);
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
