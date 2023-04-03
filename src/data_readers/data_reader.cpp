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
//
// lbann_data_reader .hpp .cpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/timer.hpp"

#include "conduit/conduit_node.hpp"

#include <future>
#include <omp.h>

namespace lbann {

#undef DEBUG
// #define DEBUG

generic_data_reader::~generic_data_reader()
{
  if (m_data_store != nullptr) {
    delete m_data_store;
    m_data_store = nullptr;
  }
}

template <class Archive>
void generic_data_reader::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_supported_input_types));
}

void generic_data_reader::setup(observer_ptr<thread_pool> io_thread_pool)
{
  m_io_thread_pool = io_thread_pool;
}

int lbann::generic_data_reader::fetch(std::vector<conduit::Node>& samples,
                                      El::Matrix<El::Int>& indices_fetched,
                                      size_t mb_size)
{
  // Check to make sure that a valid map was passed
  if (samples.empty()) {
    LBANN_ERROR("fetch function called with no valid buffers");
  }

  if (!position_valid()) {
    if (position_is_overrun()) {
      return 0;
    }
    else {
      LBANN_ERROR(std::string{} +
                  "generic data reader load error: !position_valid" +
                  " -- current pos = " + std::to_string(m_current_pos) +
                  " and there are " +
                  std::to_string(m_shuffled_indices.size()) + " indices");
    }
  }

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads());
       t++) {
    preprocess_data_source(t);
  }

  // Fetch data is executed by the thread pool so it has to dispatch
  // work to other threads in the thread pool and do some work locally
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads());
       t++) {
    // Queue up work into other threads and then finish off the
    // mini-batch in the active thread
    if (t == m_io_thread_pool->get_local_thread_id()) {
      continue;
    }
    else {
      m_io_thread_pool->submit_job_to_work_group(
        std::bind(&generic_data_reader::fetch_data_block_conduit,
                  this,
                  std::ref(samples),
                  t,
                  m_io_thread_pool->get_num_threads(),
                  mb_size,
                  std::ref(indices_fetched)));
    }
  }
  fetch_data_block_conduit(samples,
                           m_io_thread_pool->get_local_thread_id(),
                           m_io_thread_pool->get_num_threads(),
                           mb_size,
                           indices_fetched);

  // Wait for all of the threads to finish
  m_io_thread_pool->finish_work_group();

  /// Allow each thread to perform any postprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads());
       t++) {
    postprocess_data_source(t);
  }

  return mb_size;
}

int lbann::generic_data_reader::fetch(
  std::map<data_field_type, CPUMat*>& input_buffers,
  El::Matrix<El::Int>& indices_fetched,
  size_t mb_size)
{
  // Check to make sure that a valid map was passed
  if (input_buffers.empty()) {
    LBANN_ERROR("fetch function called with no valid buffer");
  }
  //  Check that all buffers within the map are valid and hold the
  //  same number of samples
  El::Int buffer_width = 0;
  for (auto& [data_field, buf] : input_buffers) {
    if (buf == nullptr || buf->Height() == 0 || buf->Width() == 0) {
      LBANN_ERROR("fetch function called with invalid buffer: h=",
                  buf->Height(),
                  " x ",
                  buf->Width(),
                  " for data field ",
                  data_field);
    }
    if (buffer_width == 0) {
      buffer_width = buf->Width();
    }
    else {
      if (buffer_width != buf->Width()) {
        LBANN_ERROR("fetch function called with a set of buffers that have "
                    "mismatched widths: h=",
                    buf->Height(),
                    " x ",
                    buf->Width(),
                    " for data field ",
                    data_field);
      }
    }
  }

#ifdef DEBUG
  if (m_current_pos == 0) {
    if (get_comm()->am_world_master()) {
      std::cout << "role: " << get_role()
                << " model: " << get_trainer().get_name()
                << " shuffled indices: ";
      for (size_t j = 0; j < 15; j++) {
        std::cout << m_shuffled_indices[j] << " ";
      }
      std::cout << "\n";
    }
  }
#endif

  if (!position_valid()) {
    if (position_is_overrun()) {
      return 0;
    }
    else {
      LBANN_ERROR(std::string{} +
                  "generic data reader load error: !position_valid" +
                  " -- current pos = " + std::to_string(m_current_pos) +
                  " and there are " +
                  std::to_string(m_shuffled_indices.size()) + " indices");
    }
  }

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads());
       t++) {
    preprocess_data_source(t);
  }

  // BVE FIXME - for the time being certain data fields, such as the
  // labels have to be zeroed out because they will typically only
  // set the single index corresponding to the categorical value.
  // With general data fields this will have to be the responsibilty
  // of the concrete data reader.
  if (has_labels() &&
      input_buffers.find(INPUT_DATA_TYPE_LABELS) != input_buffers.end()) {
    auto& buf = input_buffers[INPUT_DATA_TYPE_LABELS];
    El::Zeros_seq(*buf, buf->Height(), buf->Width());
  }

  // Fetch data is executed by the thread pool so it has to dispatch
  // work to other threads in the thread pool and do some work locally
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads());
       t++) {
    // Queue up work into other threads and then finish off the
    // mini-batch in the active thread
    if (t == m_io_thread_pool->get_local_thread_id()) {
      continue;
    }
    else {
      m_io_thread_pool->submit_job_to_work_group(
        std::bind(&generic_data_reader::fetch_data_block,
                  this,
                  std::ref(input_buffers),
                  t,
                  m_io_thread_pool->get_num_threads(),
                  mb_size,
                  std::ref(indices_fetched)));
    }
  }
  fetch_data_block(input_buffers,
                   m_io_thread_pool->get_local_thread_id(),
                   m_io_thread_pool->get_num_threads(),
                   mb_size,
                   indices_fetched);

  // Wait for all of the threads to finish
  m_io_thread_pool->finish_work_group();

  /// Allow each thread to perform any postprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads());
       t++) {
    postprocess_data_source(t);
  }

  return mb_size;
}

void lbann::generic_data_reader::start_data_store_mini_batch_exchange()
{
  int loaded_batch_size = get_loaded_mini_batch_size();

  // Make sure that every rank participates in the data store prior
  // to seeing if the local rank's position is valid.  Note that
  // every rank will hold data that may be used in the last mini-batch
  if (data_store_active()) {
    m_data_store->start_exchange_mini_batch_data(m_current_pos - m_base_offset -
                                                   m_model_offset,
                                                 loaded_batch_size);
  }
  return;
}

void lbann::generic_data_reader::finish_data_store_mini_batch_exchange()
{
  // Make sure that every rank participates in the data store prior
  // to seeing if the local rank's position is valid.  Note that
  // every rank will hold data that may be used in the last mini-batch
  if (data_store_active()) {
    m_data_store->finish_exchange_mini_batch_data();
  }
  return;
}

bool lbann::generic_data_reader::fetch_data_block(
  std::map<data_field_type, CPUMat*>& input_buffers,
  El::Int block_offset,
  El::Int block_stride,
  El::Int mb_size,
  El::Matrix<El::Int>& indices_fetched)
{
  locked_io_rng_ref io_rng = set_io_generators_local_index(block_offset);

  //  CPUMat& X
  for (int s = block_offset; s < mb_size; s += block_stride) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    indices_fetched.Set(s, 0, index);

    for (auto& [data_field, buf] : input_buffers) {
      bool valid = false;
      if (data_field == INPUT_DATA_TYPE_SAMPLES) {
        if (buf == nullptr || buf->Height() == 0 || buf->Width() == 0) {
          LBANN_ERROR(
            "fetch_data_block function called with invalid buffer: h=",
            buf->Height(),
            " x ",
            buf->Width());
        }
        valid = fetch_datum(*buf, index, s);
        if (!valid) {
          LBANN_ERROR("invalid datum (index ", std::to_string(index), ")");
        }
      }
      else if (data_field == INPUT_DATA_TYPE_LABELS && has_labels()) {
        if (buf == nullptr || buf->Height() == 0 || buf->Width() == 0) {
          LBANN_ERROR(
            "fetch_data_block function called with invalid buffer: h=",
            buf->Height(),
            " x ",
            buf->Width());
        }
        valid = fetch_label(*buf, index, s);
        if (!valid) {
          LBANN_ERROR("invalid datum (index ", std::to_string(index), ")");
        }
      }
      else if (data_field == INPUT_DATA_TYPE_RESPONSES && has_responses()) {
        if (buf == nullptr || buf->Height() == 0 || buf->Width() == 0) {
          LBANN_ERROR(
            "fetch_data_block function called with invalid buffer: h=",
            buf->Height(),
            " x ",
            buf->Width());
        }
        valid = fetch_response(*buf, index, s);
        if (!valid) {
          LBANN_ERROR("invalid datum (index ", std::to_string(index), ")");
        }
      }
      else if (has_data_field(data_field)) {
        if (buf == nullptr || buf->Height() == 0 || buf->Width() == 0) {
          LBANN_ERROR(
            "fetch_data_block function called with invalid buffer: h=",
            buf->Height(),
            " x ",
            buf->Width());
        }
        valid = fetch_data_field(data_field, *buf, index, s);
        if (!valid) {
          LBANN_ERROR("invalid datum (index ",
                      std::to_string(index),
                      ") for field ",
                      data_field);
        }
      }
      else {
        LBANN_ERROR("Unsupported data_field ", data_field);
      }
    }
  }

  return true;
}

bool lbann::generic_data_reader::fetch_data_block_conduit(
  std::vector<conduit::Node>& samples,
  El::Int block_offset,
  El::Int block_stride,
  El::Int mb_size,
  El::Matrix<El::Int>& indices_fetched)
{
  locked_io_rng_ref io_rng = set_io_generators_local_index(block_offset);

  if (static_cast<size_t>(mb_size) > samples.size()) {
    LBANN_ERROR("unable to fetch data to conduit nodes, vector length ",
                samples.size(),
                " is smaller than mini-batch size",
                mb_size);
  }
  //  CPUMat& X
  for (int s = block_offset; s < mb_size; s += block_stride) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    indices_fetched.Set(s, 0, index);

    auto& sample = samples[s];
    bool valid = fetch_conduit_node(sample, index);
    if (!valid) {
      LBANN_ERROR("invalid datum (index ", std::to_string(index), ")");
    }
  }
  return true;
}

int generic_data_reader::get_linearized_size(
  data_field_type const& data_field) const
{
  if (data_field == INPUT_DATA_TYPE_SAMPLES) {
    return get_linearized_data_size();
  }
  else if (data_field == INPUT_DATA_TYPE_LABELS) {
    return get_linearized_label_size();
  }
  else if (data_field == INPUT_DATA_TYPE_RESPONSES) {
    return get_linearized_response_size();
  }
  else if (data_field == INPUT_DATA_TYPE_LABEL_RECONSTRUCTION) {
    return get_linearized_data_size();
  }
  else {
    LBANN_ERROR("Unknown data_field_type value provided: " + data_field);
  }
  return 0;
}


bool generic_data_reader::save_to_checkpoint_shared(persist& p,
                                                    execution_mode mode)
{
  if (get_comm()->am_trainer_master()) {
    write_cereal_archive<generic_data_reader>(*this,
                                              p,
                                              mode,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                              "_dr.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                               "_dr.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  return true;
}

bool lbann::generic_data_reader::load_from_checkpoint_shared(
  persist& p,
  execution_mode mode)
{
  load_from_shared_cereal_archive<generic_data_reader>(*this,
                                                       p,
                                                       mode,
                                                       *get_comm(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                                       "_dr.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                                        "_dr.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  // Adjust current position to deal with fact that it was just loaded to all
  // ranks from rank 0 (differs by rank #)
  m_current_pos += m_comm->get_rank_in_trainer();
  return true;
}

bool generic_data_reader::save_to_checkpoint_distributed(persist& p,
                                                         execution_mode mode)
{
  write_cereal_archive<generic_data_reader>(*this,
                                            p,
                                            mode,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                            "_dr.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                             "_dr.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

  );
  return true;
}

bool lbann::generic_data_reader::load_from_checkpoint_distributed(
  persist& p,
  execution_mode mode)
{
  read_cereal_archive<generic_data_reader>(*this,
                                           p,
                                           mode,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                           "_dr.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                            "_dr.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

const data_store_conduit& generic_data_reader::get_data_store() const
{
  if (m_data_store == nullptr) {
    LBANN_ERROR("m_data_store is nullptr");
  }
  return *m_data_store;
}

data_store_conduit& generic_data_reader::get_data_store()
{
  return const_cast<data_store_conduit&>(
    static_cast<const generic_data_reader&>(*this).get_data_store());
}

void generic_data_reader::do_preload_data_store()
{
  LBANN_ERROR("Not implemented.");
}

void generic_data_reader::set_file_dir(std::string s)
{
  if (endsWith(s, "/")) {
    m_file_dir = s;
  }
  else {
    m_file_dir = s + "/";
  }
}

void generic_data_reader::set_local_file_dir(std::string s)
{
  if (endsWith(s, "/")) {
    m_local_file_dir = s;
  }
  else {
    m_local_file_dir = s + "/";
  }
}

std::string generic_data_reader::get_file_dir() const { return m_file_dir; }

std::string generic_data_reader::get_local_file_dir() const
{
  return m_local_file_dir;
}

void generic_data_reader::set_data_sample_list(std::string s)
{
  m_data_sample_list = s;
}

std::string generic_data_reader::get_data_sample_list() const
{
  return m_data_sample_list;
}

void generic_data_reader::keep_sample_order(bool same_order)
{
  // The sample_list::keep_sample_order() should be called using this
  // flag. By doing so, it will add additional step to re-shuffle the
  // sample order to restore it to the original before the loading
  // with interleaving accesses by multiple ranks in a trainer.
  m_keep_sample_order = same_order;
}

void generic_data_reader::set_data_filename(std::string s) { m_data_fn = s; }

std::string generic_data_reader::get_data_filename() const
{
  if (m_data_fn == "") {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you apparently did not call set_data_filename; error!");
  }
  return m_data_fn;
}

void generic_data_reader::set_label_filename(std::string s) { m_label_fn = s; }

std::string generic_data_reader::get_label_filename() const
{
  if (m_label_fn == "") {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you apparently did not call set_label_filename; error!");
  }
  return m_label_fn;
}


void generic_data_reader::instantiate_data_store()
{
  double tm1 = get_time();
  auto& arg_parser = global_argument_parser();
  if (!(arg_parser.get<bool>(LBANN_OPTION_USE_DATA_STORE) ||
        arg_parser.get<bool>(LBANN_OPTION_PRELOAD_DATA_STORE) ||
        arg_parser.get<bool>(LBANN_OPTION_DATA_STORE_CACHE) ||
        arg_parser.get<std::string>(LBANN_OPTION_DATA_STORE_SPILL) != "")) {
    if (m_data_store != nullptr) {
      delete m_data_store;
      m_data_store = nullptr;
    }
    return;
  }

  if (get_comm()->am_world_master()) {
    std::cout << "\nUSING DATA_STORE\n\n";
  }
  m_data_store = new data_store_conduit(this); // *data_store_conduit
  if (m_shuffled_indices.size() == 0) {
    LBANN_ERROR("shuffled_indices.size() == 0");
  }

  if (arg_parser.get<bool>(LBANN_OPTION_NODE_SIZES_VARY)) {
    m_data_store->set_node_sizes_vary();
  }

  m_data_store->set_shuffled_indices(&m_shuffled_indices);

  std::stringstream s;
  s << "generic_data_reader::instantiate_data_store time: : "
    << (get_time() - tm1);
  m_data_store->set_profile_msg(s.str());
}

void generic_data_reader::setup_data_store(int mini_batch_size)
{
  if (m_data_store == nullptr) {
    LBANN_ERROR("m_data_store == nullptr; you shouldn't be here");
  }
  // optionally preload the data store
  if (m_data_store->is_preloading()) {
    m_data_store->set_profile_msg(
      "generic_data_reader::instantiate_data_store - starting the preload");
    double tm2 = get_time();
    preload_data_store();
    std::stringstream s;
    s << "Preload complete; time: " << get_time() - tm2;
    m_data_store->set_profile_msg(s.str());
  }

  m_data_store->setup(mini_batch_size);
}

bool generic_data_reader::data_store_active() const
{
  if (m_data_store != nullptr && m_data_store->is_fully_loaded()) {
    return true;
  }

  const auto& c = static_cast<const SGDExecutionContext&>(
    get_trainer().get_data_coordinator().get_execution_context());
  /// Use the data store for all modes except testing
  /// i.e. training, validation, tournament
  return (m_data_store != nullptr &&
          (((c.get_execution_mode() == execution_mode::training) &&
            c.get_epoch() > 0) ||
           ((c.get_execution_mode() == execution_mode::validation) &&
            c.get_epoch() > 0)));
}

bool generic_data_reader::priming_data_store() const
{
  const auto& c = static_cast<const SGDExecutionContext&>(
    get_trainer().get_data_coordinator().get_execution_context());
  if (m_data_store != nullptr && m_data_store->is_fully_loaded()) {
    return false;
  }

  /// Use the data store for all modes except testing
  /// i.e. training, validation, tournament
  return (m_data_store != nullptr &&
          (((c.get_execution_mode() == execution_mode::training) &&
            c.get_epoch() == 0) ||
           ((c.get_execution_mode() == execution_mode::validation) &&
            c.get_epoch() == 0) ||
           m_data_store->is_explicitly_loading()));
}

void generic_data_reader::set_data_store(data_store_conduit* g)
{
  if (m_data_store != nullptr) {
    delete m_data_store;
  }
  m_data_store = g;
}

void generic_data_reader::preload_data_store()
{
  if (m_data_store->is_local_cache()) {
    m_data_store->set_profile_msg(
      "generic_data_reader::preload_data_store() calling "
      "m_data_store->preload_local_cache()");
    m_data_store->preload_local_cache();
  }

  else {
    std::vector<int> local_list_sizes;
    int np = m_comm->get_procs_per_trainer();
    int base_files_per_rank = m_shuffled_indices.size() / np;
    int extra = m_shuffled_indices.size() - (base_files_per_rank * np);
    if (extra > np) {
      LBANN_ERROR("extra > np");
    }
    local_list_sizes.resize(np, 0);
    for (int j = 0; j < np; j++) {
      local_list_sizes[j] = base_files_per_rank;
      if (j < extra) {
        local_list_sizes[j] += 1;
      }
    }
    m_data_store->set_profile_msg(
      "generic_data_reader::preload_data_store() calling "
      "m_data_store->build_preloaded_owner_map()");
    m_data_store->build_preloaded_owner_map(local_list_sizes);
    m_data_store->set_profile_msg("generic_data_reader::preload_data_store() "
                                  "calling do_preload_data_store()");
    do_preload_data_store();
    m_data_store->set_loading_is_complete();
  }
}

void generic_data_reader::print_get_methods(const std::string filename)
{
  if (!get_comm()->am_world_master()) {
    return;
  }
  std::ofstream out(filename.c_str());
  if (!out) {
    LBANN_ERROR("failed to open ", filename, " for writing");
  }

  out << "get_file_dir " << get_file_dir() << std::endl;
  out << "get_local_file_dir " << get_local_file_dir() << std::endl;
  out << "get_data_sample_list " << get_data_sample_list() << std::endl;
  out << "get_data_filename " << get_data_filename() << std::endl;
  out << "get_label_filename " << get_label_filename() << std::endl;
  out << "get_role " << get_role() << std::endl;
  out << "get_type " << get_type() << std::endl;
  out << "get_num_data " << get_num_data() << std::endl;
  out << "get_absolute_sample_count" << get_absolute_sample_count()
      << std::endl;
  out << "get_use_percent " << get_use_percent() << std::endl;
  for (auto m : execution_mode_iterator()) {
    out << "get_execution_mode_split_percent[" << to_string(m) << "] "
        << get_execution_mode_split_percent(m) << std::endl;
  }
  out.close();
}

} // namespace lbann
