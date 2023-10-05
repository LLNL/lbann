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

#include <lbann/comm_impl.hpp>
#include <lbann/data_coordinator/data_coordinator.hpp>
#include <lbann/data_readers/data_reader.hpp>
#include <lbann/data_store/data_store_conduit.hpp>
#include <lbann/execution_algorithms/execution_context.hpp>
#include <lbann/trainers/trainer.hpp>
#include <lbann/utils/dim_helpers.hpp>
#include <lbann/utils/distconv.hpp>
#include <lbann/utils/serialize.hpp>

namespace lbann {

data_coordinator::~data_coordinator()
{
  // Synchronize the I/O thread pool
  // Note: The thread pool may still be running asynchronously if the
  // trainer is destroyed in the middle of an epoch. The thread pool
  // needs to interact with data readers, etc., so it needs to be
  // synchronized before any of them are destroyed.
  if (m_io_thread_pool != nullptr) {
    m_io_thread_pool->reap_threads();
  }
  // Data coordinator always frees data readers.
  for (auto& dr : m_data_readers) {
    delete dr.second;
  }
}

// Data Coordinators copy their data readers.
data_coordinator::data_coordinator(const data_coordinator& other)
  : m_comm(other.m_comm),
    m_datasets(other.m_datasets),
    m_data_readers(other.m_data_readers),
    m_data_set_processed(other.m_data_set_processed),
    m_execution_context(other.m_execution_context)
{
  for (auto& dr : m_data_readers) {
    dr.second = dr.second ? dr.second->copy() : nullptr;
  }
}

data_coordinator& data_coordinator::operator=(const data_coordinator& other)
{
  for (auto& dr : m_data_readers) {
    dr.second = dr.second ? dr.second->copy() : nullptr;
  }
  return *this;
}

template <class Archive>
void data_coordinator::serialize(Archive& ar)
{
  ar(/*CEREAL_NVP(m_io_buffer),*/
     CEREAL_NVP(m_datasets)/*,
     CEREAL_NVP(m_active_data_fields),
     CEREAL_NVP(m_data_readers),
     CEREAL_NVP(m_data_set_processed)*/);
}

void data_coordinator::setup(
  thread_pool& io_thread_pool,
  int max_mini_batch_size,
  std::map<execution_mode, generic_data_reader*> data_readers)
{
  m_io_thread_pool = &io_thread_pool;

  m_data_readers = data_readers;

  // Initialize the data sets
  for (auto m : execution_mode_iterator()) {
    if (this->m_data_readers.count(m)) {
      this->m_datasets[m].total_samples() = m_data_readers[m]->get_num_data();
    }
  }

  /// @todo BVE FIXME the list of execution modes should not include
  // ones will null data readers.  Fix this in next PR.
  // Setup data readers
  for (auto&& dr : m_data_readers) {
    if (!dr.second)
      continue;
    dr.second->setup(m_io_thread_pool->get_num_threads(), m_io_thread_pool);
  }

  /** Calculate how many iterations are required for training, testing,
   *  and validation given a specified mini-batch size.
   */
  for (auto&& dr : m_data_readers) {
    if (!dr.second)
      continue;
    calculate_num_iterations_per_epoch(max_mini_batch_size, dr.second);
  }

  auto& arg_parser = global_argument_parser();
  if (arg_parser.get<bool>(LBANN_OPTION_USE_DATA_STORE) ||
      arg_parser.get<bool>(LBANN_OPTION_PRELOAD_DATA_STORE) ||
      arg_parser.get<bool>(LBANN_OPTION_DATA_STORE_CACHE) ||
      arg_parser.get<std::string>(LBANN_OPTION_DATA_STORE_SPILL) != "") {
    bool master = m_comm->am_world_master();
    if (master) {
      std::cout << "\nUSING DATA STORE!\n\n";
    }
    for (auto&& r : m_data_readers) {
      if (!r.second)
        continue;
      r.second->setup_data_store(max_mini_batch_size);
    }
  }
}

void data_coordinator::calculate_num_iterations_per_epoch(
  int max_mini_batch_size,
  generic_data_reader* data_reader)
{
  if (data_reader == nullptr) {
    return;
  }
  // If the data reader does not have any data bail out (e.g. unused validation
  // reader)
  if (data_reader->get_num_data() == 0) {
    return;
  }

  if (max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel
  /// readers
  int num_parallel_readers_per_model =
    compute_max_num_parallel_readers(data_reader->get_num_data(),
                                     max_mini_batch_size,
                                     this->m_comm->get_procs_per_trainer());
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if (num_parallel_readers_per_model == 0 ||
      (num_parallel_readers_per_model !=
         this->m_comm->get_procs_per_trainer() &&
       num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_distribution: number of parallel readers is zero");
  }

#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    // #trainers is assumed to be 1.
    assert_eq(this->m_comm->get_num_trainers(), 1);
  }
#endif

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = max_mini_batch_size;
  int base_offset = this->m_comm->get_rank_in_trainer();
#ifdef LBANN_HAS_DISTCONV
  base_offset =
    dc::get_input_rank(*(this->m_comm)) / dc::get_number_of_io_partitions();
#endif
  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
#ifdef LBANN_HAS_DISTCONV
  data_reader->set_sample_stride(num_parallel_readers_per_model /
                                 dc::get_number_of_io_partitions());
#else
  data_reader->set_sample_stride(num_parallel_readers_per_model);
#endif
  data_reader->set_iteration_stride(1);
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(0);
  data_reader->set_initial_position();

  /// By default each data reader will plan to process the entire data set
  int num_iterations_per_epoch =
    ceil((float)data_reader->get_num_data() / (float)max_mini_batch_size);
  int last_mini_batch_size = data_reader->get_num_data() % max_mini_batch_size;
  if (last_mini_batch_size == 0) {
    last_mini_batch_size = max_mini_batch_size;
  }
  data_reader->set_num_iterations_per_epoch(num_iterations_per_epoch);
  data_reader->set_last_mini_batch_size(last_mini_batch_size);
  data_reader->set_stride_to_last_mini_batch(
    data_reader->get_stride_to_next_mini_batch());
  return;
}

generic_data_reader*
data_coordinator::get_data_reader(const execution_mode mode) const
{
  generic_data_reader* data_reader = nullptr;
  auto it = m_data_readers.find(mode);
  if (it != m_data_readers.end())
    data_reader = it->second;
  return data_reader;
}

TargetModeDimMap data_coordinator::get_data_dims() const
{
  TargetModeDimMap map;
  generic_data_reader* dr;
  for (execution_mode mode : execution_mode_iterator()) {
    dr = get_data_reader(mode);
    if (dr != nullptr) {
      map[data_reader_target_mode::INPUT] = dr->get_data_dims();
      if (dr->has_labels()) {
        map[data_reader_target_mode::CLASSIFICATION] =
          std::vector<El::Int>(1, dr->get_num_labels());
      }
      else {
        map[data_reader_target_mode::CLASSIFICATION] =
          std::vector<El::Int>(1, 0);
      }
      if (dr->has_responses()) {
        map[data_reader_target_mode::REGRESSION] =
          std::vector<El::Int>(1, dr->get_num_responses());
      }
      else {
        map[data_reader_target_mode::REGRESSION] = std::vector<El::Int>(1, 0);
      }
      map[data_reader_target_mode::RECONSTRUCTION] = dr->get_data_dims();
      map[data_reader_target_mode::LABEL_RECONSTRUCTION] = dr->get_data_dims();
      map[data_reader_target_mode::NA] = std::vector<El::Int>(1, 0);
      return map;
    }
  }
  LBANN_ERROR("get_data_dims: no available data readers");
  return {};
}

/**
 * Get the dimensions of the underlying data.
 */
SPModeSlicePoints data_coordinator::get_slice_points() const
{
  SPModeSlicePoints map;
  generic_data_reader* dr;
  for (execution_mode mode : execution_mode_iterator()) {
    dr = get_data_reader(mode);
    if (dr != nullptr) {
      for (slice_points_mode sp_mode : slice_points_mode_iterator()) {
        bool is_supported;
        std::vector<El::Int> tmp = dr->get_slice_points(sp_mode, is_supported);
        if (is_supported) {
          map[sp_mode] = tmp;
        }
      }
      return map;
    }
  }
  LBANN_ERROR("get_data_dims: no available data readers");
  return {};
}

DataReaderMetaData data_coordinator::get_dr_metadata() const
{
  if (m_mock_data_reader_metadata)
    return *m_mock_data_reader_metadata;

  DataReaderMetaData drm;
  drm.data_dims = get_data_dims();
  drm.slice_points = get_slice_points();
#ifdef LBANN_HAS_DISTCONV
  const auto training_dr = m_data_readers.at(execution_mode::training);
  drm.shuffle_required = training_dr->is_tensor_shuffle_required();
#endif // LBANN_HAS_DISTCONV
  return drm;
}

void data_coordinator::set_mock_dr_metadata(const DataReaderMetaData& drm)
{
  m_mock_data_reader_metadata = std::make_unique<DataReaderMetaData>(drm);
}

void data_coordinator::clear_mock_dr_metadata()
{
  m_mock_data_reader_metadata.reset();
}

/**
 * Check to see if the data readers have labels
 */
bool data_coordinator::has_labels() const
{
  bool flag = false;
  generic_data_reader* dr;
  for (auto mode : execution_mode_iterator()) {
    dr = get_data_reader(mode);
    if (dr != nullptr) {
      flag = dr->has_labels();
      if (flag) {
        return flag;
      }
    }
  }
  return flag;
}

/**
 * Check to see if the data readers have responses
 */
bool data_coordinator::has_responses() const
{
  bool flag = false;
  generic_data_reader* dr;
  for (auto mode : execution_mode_iterator()) {
    dr = get_data_reader(mode);
    if (dr != nullptr) {
      flag = dr->has_responses();
      if (flag) {
        return flag;
      }
    }
  }
  return flag;
}

/**
 * Get the linearized size of the underlying data.
 */
long data_coordinator::get_linearized_size(
  data_field_type const& data_field) const
{
  long linearized_size = -1;
  for (auto mode : execution_mode_iterator()) {
    if (generic_data_reader const* const dr = get_data_reader(mode)) {
      long tmp_size = dr->get_linearized_size(data_field);
      if (linearized_size != -1 && linearized_size != tmp_size) {
        LBANN_ERROR(
          "data_coordinator: ",
          to_string(mode),
          " data set size (",
          std::to_string(tmp_size),
          ") does not match the currently established data set size (",
          std::to_string(linearized_size),
          ")");
      }
      linearized_size = tmp_size;
    }
  }
  auto it = m_active_data_fields_dim_map.find(data_field);
  if (it != m_active_data_fields_dim_map.end()) {
    auto& dim_map = it->second;
    if (linearized_size != get_linear_size(dim_map)) {
      if (linearized_size == -1) {
        LBANN_WARNING("Unable to find data readers; using data field map for "
                      "linearized size for data field: ",
                      data_field,
                      " = ",
                      get_linear_size(dim_map));
        linearized_size = get_linear_size(dim_map);
      }
      else {
        LBANN_ERROR("The data readers and data field map disagree on the "
                    "linearized size of the field: ",
                    data_field,
                    ": ",
                    linearized_size,
                    " != ",
                    get_linear_size(dim_map));
      }
    }
  }
  return linearized_size;
}

/**
 * Get the linearized size of the underlying data.
 */
long data_coordinator::get_linearized_data_size() const
{
  return get_linearized_size(INPUT_DATA_TYPE_SAMPLES);
}

/**
 * Get the linearized size of the labels for the underlying data.
 */
long data_coordinator::get_linearized_label_size() const
{
  return get_linearized_size(INPUT_DATA_TYPE_LABELS);
}

/**
 * Get the linearized size of the responses for the underlying data.
 */
long data_coordinator::get_linearized_response_size() const
{
  return get_linearized_size(INPUT_DATA_TYPE_RESPONSES);
}

// At the start of the epoch, set the execution mode and make sure
// that each layer points to this model
void data_coordinator::reset_mode(ExecutionContext& context)
{
  m_execution_context = static_cast<observer_ptr<ExecutionContext>>(&context);
}

dataset& data_coordinator::get_dataset(execution_mode m)
{
  if (m_datasets.count(m)) {
    return m_datasets.at(m);
  }
  else {
    LBANN_ERROR("get_dataset: invalid execution mode");
  }
}

const dataset& data_coordinator::get_dataset(execution_mode m) const
{
  if (m_datasets.count(m)) {
    return m_datasets.at(m);
  }
  else {
    LBANN_ERROR("get_dataset: invalid execution mode");
  }
}

dataset* data_coordinator::select_first_valid_dataset()
{
  for (auto m : execution_mode_iterator()) {
    if (m_datasets.count(m)) {
      return &m_datasets.at(m);
    }
  }
  return nullptr;
}

long data_coordinator::get_num_samples(execution_mode m) const
{
  if (m_datasets.count(m)) {
    return m_datasets.at(m).get_num_samples_processed();
  }
  else {
    return 0;
  }
}

long data_coordinator::get_total_num_samples(execution_mode m) const
{
  if (m_datasets.count(m)) {
    return m_datasets.at(m).get_total_samples();
  }
  else {
    return 0;
  }
}

long data_coordinator::update_num_samples_processed(execution_mode mode,
                                                    long num_samples)
{
  dataset& ds = get_dataset(mode);
  ds.num_samples_processed() += num_samples;
  return ds.get_num_samples_processed();
}

bool data_coordinator::is_execution_mode_valid(execution_mode mode) const
{
  return (get_total_num_samples(mode) != static_cast<long>(0));
}

void data_coordinator::calculate_num_iterations_per_epoch(int mini_batch_size)
{
  for (auto&& dr : m_data_readers) {
    if (!dr.second)
      continue;
    calculate_num_iterations_per_epoch(mini_batch_size, dr.second);
  }
}

int data_coordinator::compute_max_num_parallel_readers(
  long data_set_size,
  int mini_batch_size,
  int requested_num_parallel_readers) const
{
  return compute_max_num_parallel_readers(data_set_size,
                                          mini_batch_size,
                                          requested_num_parallel_readers,
                                          this->m_comm);
}

int data_coordinator::compute_max_num_parallel_readers(
  long data_set_size,
  int mini_batch_size,
  int requested_num_parallel_readers,
  const lbann_comm* comm)
{
  int num_parallel_readers = requested_num_parallel_readers;

  if (comm->get_procs_per_trainer() != num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers << " does not match the grid size "
                << comm->get_procs_per_trainer()
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = comm->get_procs_per_trainer();
  }

#if 0
  if(mini_batch_size < num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " is larger than the requested mini-batch size "
                << mini_batch_size
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = mini_batch_size;
  }
#endif
  return num_parallel_readers;
}

int data_coordinator::get_num_parallel_readers(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_num_parallel_readers() : 0;
}

bool data_coordinator::at_new_epoch(execution_mode mode) const
{
  const generic_data_reader* dr = get_data_reader(mode);
  return (dr != nullptr && dr->at_new_epoch());
}

bool data_coordinator::at_new_epoch() const
{
  return at_new_epoch(execution_mode::training);
}

void data_coordinator::register_active_data_field(
  data_field_type const& data_field,
  std::vector<El::Int> const& data_field_dim_map)
{
  m_active_data_fields.insert(data_field);
  m_active_data_fields_dim_map[data_field] = data_field_dim_map;
}

size_t data_coordinator::get_num_iterations_per_epoch(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_num_iterations_per_epoch()
                                  : 0;
}

int data_coordinator::get_current_step_in_epoch(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_current_step_in_epoch()
                                  : 0;
}

int data_coordinator::get_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_mini_batch_size() : 0;
}

int data_coordinator::get_last_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_last_mini_batch_size() : 0;
}

int data_coordinator::get_current_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_current_mini_batch_size()
                                  : 0;
}

// save state of IO to a checkpoint
bool data_coordinator::save_to_checkpoint_shared(persist& p) const
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  if (p.get_cb_type() == callback_type::execution_context_only ||
      p.get_cb_type() == callback_type::full_checkpoint) {

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_shared(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_shared(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_shared(p, execution_mode::validation);
    }

    // if (this->m_comm->am_trainer_master()) {
    //   write_cereal_archive<const data_coordinator>(*this, p,
    //   execution_mode::training, "_dc.xml");
    // }
  }
  return true;
}

// reload state of IO from a checkpoint
bool data_coordinator::load_from_checkpoint_shared(persist& p)
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  if (p.get_cb_type() == callback_type::execution_context_only ||
      p.get_cb_type() == callback_type::full_checkpoint) {

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::validation);
    }

    // std::string buf;
    // if (this->m_comm->am_trainer_master()) {
    //   read_cereal_archive<data_coordinator>(*this, p,
    //   execution_mode::training, "_dc.xml"); buf =
    //   create_cereal_archive_binary_string<data_coordinator>(*this);
    // }

    // // TODO: this assumes homogeneous processors
    // // broadcast state from rank 0
    // this->m_comm->trainer_broadcast(0, buf);

    // if (!this->m_comm->am_trainer_master()) {
    //   unpack_cereal_archive_binary_string<data_coordinator>(*this, buf);
    // }
  }

  return true;
}

bool data_coordinator::save_to_checkpoint_distributed(persist& p) const
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  if (p.get_cb_type() == callback_type::execution_context_only ||
      p.get_cb_type() == callback_type::full_checkpoint) {

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_distributed(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_distributed(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)
        ->save_to_checkpoint_distributed(p, execution_mode::validation);
    }

    // write_cereal_archive<const data_coordinator>(*this, p,
    // execution_mode::training, "_dc.xml");
  }
  return true;
}

bool data_coordinator::load_from_checkpoint_distributed(persist& p)
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  it = this->m_data_readers.find(execution_mode::training);
  if ((it != this->m_data_readers.end()) && it->second) {
    (it->second)->load_from_checkpoint_distributed(p, execution_mode::training);
  }
  it = this->m_data_readers.find(execution_mode::testing);
  if ((it != this->m_data_readers.end()) && it->second) {
    (it->second)->load_from_checkpoint_distributed(p, execution_mode::testing);
  }
  it = this->m_data_readers.find(execution_mode::validation);
  if ((it != this->m_data_readers.end()) && it->second) {
    (it->second)
      ->load_from_checkpoint_distributed(p, execution_mode::validation);
  }

  // read_cereal_archive<data_coordinator>(*this, p, execution_mode::training,
  // "_dc.xml");
  return true;
}

// only used in LTFB; from that file:
// "Note that this is a temporary fix
// for the current use of the tournament"
void data_coordinator::make_data_store_preloaded(execution_mode mode)
{
  auto* dr = this->get_data_reader(mode);
  auto* data_store = dr->get_data_store_ptr();
  if (data_store != nullptr && !data_store->is_fully_loaded()) {
    dr->get_data_store_ptr()->set_loading_is_complete();
    dr->get_data_store_ptr()->set_is_explicitly_loading(false);
  }
}

// only used in LTFB; from that file:
// "Note that this is a temporary fix
// for the current use of the tournament"
void data_coordinator::mark_data_store_explicitly_loading(execution_mode mode)
{
  auto* dr = this->get_data_reader(mode);
  auto* data_store = dr->get_data_store_ptr();
  if (data_store != nullptr && !data_store->is_fully_loaded()) {
    dr->get_data_store_ptr()->set_is_explicitly_loading(true);
  }
}

} // namespace lbann

#define LBANN_CLASS_NAME data_coordinator
#include <lbann/macros/register_class_with_cereal.hpp>
