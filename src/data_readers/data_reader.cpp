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
  ar(CEREAL_NVP(m_current_mini_batch_idx),
     CEREAL_NVP(m_current_pos),
     CEREAL_NVP(m_shuffled_indices),
     CEREAL_NVP(m_supported_input_types));
}

void generic_data_reader::shuffle_indices()
{
  shuffle_indices(get_data_seq_generator());
}

void generic_data_reader::shuffle_indices(rng_gen& gen)
{
  // Shuffle the data
  if (m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(), gen);
  }
}

void generic_data_reader::setup(int num_io_threads,
                                observer_ptr<thread_pool> io_thread_pool)
{
  m_base_offset = 0;
  m_sample_stride = 1;
  m_stride_to_next_mini_batch = 0;
  m_stride_to_last_mini_batch = 0;
  m_current_mini_batch_idx = 0;
  m_num_iterations_per_epoch = 0;

  set_initial_position();

  shuffle_indices();

  m_io_thread_pool = io_thread_pool;
}

int lbann::generic_data_reader::fetch(std::vector<conduit::Node>& samples,
                                      El::Matrix<El::Int>& indices_fetched,
                                      El::Int current_position_in_data_set,
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
                  current_position_in_data_set,
                  t,
                  m_io_thread_pool->get_num_threads(),
                  mb_size,
                  std::ref(indices_fetched)));
    }
  }
  fetch_data_block_conduit(samples,
                           current_position_in_data_set,
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
  El::Int current_position_in_data_set,
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
                  current_position_in_data_set,
                  t,
                  m_io_thread_pool->get_num_threads(),
                  mb_size,
                  std::ref(indices_fetched)));
    }
  }
  fetch_data_block(input_buffers,
                   current_position_in_data_set,
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

void lbann::generic_data_reader::start_data_store_mini_batch_exchange(
  El::Int current_position_in_data_set,
  El::Int current_mini_batch_size)
{
  // Make sure that every rank participates in the data store prior
  // to seeing if the local rank's position is valid.  Note that
  // every rank will hold data that may be used in the last mini-batch
  if (data_store_active()) {
    m_data_store->start_exchange_mini_batch_data(current_position_in_data_set -
                                                   m_base_offset,
                                                 current_mini_batch_size);
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
  El::Int current_position_in_data_set,
  El::Int block_offset,
  El::Int block_stride,
  El::Int mb_size,
  El::Matrix<El::Int>& indices_fetched)
{
  locked_io_rng_ref io_rng = set_io_generators_local_index(block_offset);

  //  CPUMat& X
  for (int s = block_offset; s < mb_size; s += block_stride) {
    // LBANN_MSG(
    //   "I am fetching a data block and I think that the current position is ",
    //   current_position_in_data_set,
    //   " instead of ",
    //   m_current_pos);
    int n = current_position_in_data_set + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    indices_fetched.Set(s, 0, index);

    // LBANN_MSG("fetch data block is getting s = ",
    //           s,
    //           " with offset = ",
    //           block_offset,
    //           " stride = ",
    //           block_stride,
    //           " current_position ",
    //           m_current_pos,
    //           " n = ",
    //           n,
    //           " and index = ",
    //           index);
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
  El::Int current_position_in_data_set,
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
    int n = current_position_in_data_set + (s * m_sample_stride);
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

bool generic_data_reader::update(bool is_active_reader)
{
  bool reader_not_done = true; // BVE The sense of this should be fixed
  m_current_mini_batch_idx++;

  if (is_active_reader) {
    m_current_pos = get_next_position();
  }
  // if (m_current_mini_batch_idx >= m_num_iterations_per_epoch) {
  //   //  if (m_loaded_mini_batch_idx >= m_num_iterations_per_epoch) {
  //   reader_not_done = false;
  // }
  if ((size_t)m_current_pos >= m_shuffled_indices.size()) {
    reader_not_done = false;
  }
  if (m_current_mini_batch_idx == m_num_iterations_per_epoch) {
    // for working with 1B jag samples, we may not process all the data
    if (m_current_pos < (int)m_shuffled_indices.size()) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic data reader update error: the epoch is complete," +
        " but not all of the data has been used -- current pos = " +
        std::to_string(m_current_pos) + " and there are " +
        std::to_string(m_shuffled_indices.size()) + " indices" +
        " : iteration=" + std::to_string(m_current_mini_batch_idx) + "C of" +
        std::to_string(m_num_iterations_per_epoch) + "+" +
        " index stride=" + std::to_string(m_stride_to_next_mini_batch) + "/" +
        std::to_string(m_stride_to_last_mini_batch));
    }

    shuffle_indices();
    if (m_data_store != nullptr && priming_data_store()) {
      m_data_store->set_shuffled_indices(&m_shuffled_indices);
    }

    set_initial_position();
  }

  return reader_not_done;
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

int generic_data_reader::get_next_mini_batch_size() const
{
  if (m_current_mini_batch_idx + 1 > (m_num_iterations_per_epoch - 1)) {
    return 0;
  }
  else if (m_current_mini_batch_idx + 1 == (m_num_iterations_per_epoch - 1)) {
    return m_last_mini_batch_size;
  }
  else {
    return m_mini_batch_size;
  }
}

int generic_data_reader::get_current_mini_batch_size() const
{
  if (m_current_mini_batch_idx > (m_num_iterations_per_epoch - 1)) {
    return 0;
  }
  else if (m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_last_mini_batch_size;
  }
  else {
    return m_mini_batch_size;
  }
}

int generic_data_reader::get_next_position() const
{
  /// If the next mini-batch for this rank is going to be the last
  /// mini-batch, take the proper (possibly reduced) step to
  /// setup for the last mini-batch
  if (m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_current_pos + m_stride_to_last_mini_batch;
  }
  else {
    return m_current_pos + m_stride_to_next_mini_batch;
  }
}

int generic_data_reader::get_num_unused_data(execution_mode m) const
{
  if (m_unused_indices.count(m)) {
    return (int)m_unused_indices.at(m).size();
  }
  else {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }
}
/// Get a pointer to the start of the unused sample indices.
int* generic_data_reader::get_unused_data(execution_mode m)
{
  if (m_unused_indices.count(m)) {
    return &(m_unused_indices[m][0]);
  }
  else {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }
}
const std::vector<int>&
generic_data_reader::get_unused_indices(execution_mode m)
{
  if (m_unused_indices.count(m)) {
    return m_unused_indices.at(m);
  }
  else {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }
}

void generic_data_reader::error_check_counts() const
{
  size_t count = get_absolute_sample_count();
  double use_fraction = get_use_fraction();
  if (count == 1 and use_fraction == 0.0) {
    LBANN_ERROR("get_use_fraction() and get_absolute_sample_count() are both "
                "zero; exactly one must be zero");
  }
  if (!(count == 0 or use_fraction == 0.0)) {
    LBANN_ERROR("get_use_fraction() and get_absolute_sample_count() are both "
                "non-zero; exactly one must be zero");
  }
  if (count != 0) {
    if (count > static_cast<size_t>(get_num_data())) {
      LBANN_ERROR("absolute_sample_count=" + std::to_string(count) +
                  " is > get_num_data=" + std::to_string(get_num_data()));
    }
  }
}

size_t generic_data_reader::get_num_indices_to_use() const
{
  error_check_counts();
  // note: exactly one of the following is guaranteed to be non-zero
  size_t count = get_absolute_sample_count();
  double use_fraction = get_use_fraction();

  size_t r = 0.;
  if (count) {
    r = count;
  }
  else if (use_fraction) {
    r = use_fraction * get_num_data();
    if (r == 0) {
      LBANN_ERROR("get_num_indices_to_use() computed zero indices; probably: "
                  "fraction_of_data_to_use is too small WRT num_data;  "
                  "get_absolute_sample_count: ",
                  get_absolute_sample_count(),
                  " use_fraction: ",
                  get_use_fraction(),
                  " num data: ",
                  get_num_data(),
                  " for role: ",
                  get_role());
    }
  }
  else {
    LBANN_ERROR("it's impossible to be here");
  }

  return r;
}

void generic_data_reader::resize_shuffled_indices()
{
  size_t num_indices = get_num_indices_to_use();
  shuffle_indices();
  m_shuffled_indices.resize(num_indices);
}

void generic_data_reader::select_subset_of_data()
{
  // Calculate the total number of samples for subsets
  double total_split_fraction = 0.;
  for (auto m : execution_mode_iterator()) {
    total_split_fraction += get_execution_mode_split_fraction(m);
  }
  long total_num_data = get_num_data();
  long total_unused = total_split_fraction * total_num_data;
  long total_used = total_num_data - total_unused;
  auto starting_unused_offset = m_shuffled_indices.begin() + total_used;
  for (auto m : execution_mode_iterator()) {
    double split_fraction = get_execution_mode_split_fraction(m);

    if (split_fraction == 0.) {
      continue;
    }

    long split = split_fraction * total_num_data;
    if (split == 0) {
      LBANN_ERROR(to_string(m),
                  " % of ",
                  split_fraction,
                  " was requested, but the number of validation indices was "
                  "computed as zero. Probably: % ",
                  to_string(m),
                  " requested is too small wrt num_indices (aka, num samples)");
    }
    if (split > 0) {
      if (starting_unused_offset + split > m_shuffled_indices.end()) {
        LBANN_ERROR(
          "Split range exceeds the maximun numbrer of shuffled indices");
      }
      m_unused_indices[m] = std::vector<int>(starting_unused_offset,
                                             starting_unused_offset + split);
      starting_unused_offset += split;
    }
  }
  m_shuffled_indices.resize(total_used);

  if (!m_shuffle) {
    std::sort(m_shuffled_indices.begin(), m_shuffled_indices.end());
    for (auto m : execution_mode_iterator()) {
      if (m_unused_indices.count(m)) {
        std::sort(m_unused_indices[m].begin(), m_unused_indices[m].end());
      }
    }
  }
}

void generic_data_reader::use_unused_index_set(execution_mode m)
{
  if (m_unused_indices.count(m) == 0) {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }

  m_shuffled_indices.swap(m_unused_indices[m]);
  if (m_data_store != nullptr) {
    /// Update the data store's pointer to the shuffled indices
    m_data_store->set_shuffled_indices(&m_shuffled_indices);
  }
  m_unused_indices[m].clear();
  std::vector<int>().swap(
    m_unused_indices[m]); // Trick to force memory reallocation
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

void generic_data_reader::set_first_n(int n) { m_first_n = n; }

void generic_data_reader::set_absolute_sample_count(size_t s)
{
  m_absolute_sample_count = s;
}

size_t generic_data_reader::get_absolute_sample_count() const
{
  return m_absolute_sample_count;
}

void generic_data_reader::set_execution_mode_split_fraction(execution_mode m,
                                                            double s)
{
  if (s < 0 or s > 1.0) {
    throw lbann_exception(std::string{} + __FILE__ + " " +
                          std::to_string(__LINE__) +
                          " :: set_validation_fraction() - must be: s >= 0, s "
                          "<= 1.0; you passed: " +
                          std::to_string(s));
  }
  m_execution_mode_split_fraction[m] = s;
}

double
generic_data_reader::get_execution_mode_split_fraction(execution_mode m) const
{
  if (m_execution_mode_split_fraction.count(m)) {
    return m_execution_mode_split_fraction.at(m);
  }
  else {
    return 0;
  }
}

void generic_data_reader::set_use_fraction(double s)
{
  if (s < 0 or s > 1.0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: set_use_fraction() - must be: s >= 0, s <= 1.0; you passed: " +
      std::to_string(s));
  }
  m_use_fraction = s;
}

double generic_data_reader::get_use_fraction() const { return m_use_fraction; }

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
  if (m_data_store == nullptr) {
    return false;
  }
  if (m_data_store->is_fully_loaded()) {
    return true;
  }

  const auto& c = static_cast<const SGDExecutionContext&>(
    get_trainer().get_data_coordinator().get_execution_context());
  /// Use the data store for all modes except testing
  /// i.e. training, validation, tournament
  return ((((c.get_execution_mode() == execution_mode::training) &&
            c.get_epoch() > 0) ||
           ((c.get_execution_mode() == execution_mode::validation) &&
            c.get_epoch() > 0)));
}

bool generic_data_reader::priming_data_store() const
{
  if (m_data_store != nullptr && m_data_store->is_fully_loaded()) {
    return false;
  }
  const auto& c = static_cast<const SGDExecutionContext&>(
    get_trainer().get_data_coordinator().get_execution_context());

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

void generic_data_reader::set_mini_batch_size(const int s)
{
  m_mini_batch_size = s;
}

void generic_data_reader::set_role(std::string role) { m_role = role; }

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

void generic_data_reader::print_config()
{
  if (!get_comm()->am_world_master()) {
    return;
  }
  LBANN_MSG("\n",
            " role                       = ",
            m_role,
            "\n",
            " mini_batch_size            = ",
            m_mini_batch_size,
            "\n",
            " stride_to_next_mini_batch  = ",
            m_stride_to_next_mini_batch,
            "\n",
            " base_offset                = ",
            m_base_offset,
            "\n",
            " sample_stride              = ",
            m_sample_stride,
            "\n",
            " last_mini_batch_size       = ",
            m_last_mini_batch_size,
            "\n",
            " stride_to_last_mini_batch  = ",
            m_stride_to_last_mini_batch,
            "\n",
            " current_mini_batch_idx     = ",
            m_current_mini_batch_idx,
            "\n",
            " num_iterations_per_epoch   = ",
            m_num_iterations_per_epoch,
            "\n");
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
  out << "get_use_fraction " << get_use_fraction() << std::endl;
  for (auto m : execution_mode_iterator()) {
    out << "get_execution_mode_split_fraction[" << to_string(m) << "] "
        << get_execution_mode_split_fraction(m) << std::endl;
  }
  out.close();
}

} // namespace lbann
