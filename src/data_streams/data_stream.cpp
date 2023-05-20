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

#include "lbann/data_streams/data_stream.hpp"
#include "lbann/io/persist_impl.hpp"

namespace lbann {

template <class Archive>
void data_stream::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_current_mini_batch_idx),
     CEREAL_NVP(m_current_pos),
     CEREAL_NVP(m_shuffled_indices));
}

void data_stream::shuffle_indices()
{
  shuffle_indices(get_data_seq_generator());
}

void data_stream::shuffle_indices(rng_gen& gen)
{
  // Shuffle the data
  if (m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(), gen);
  }
}

// BVE Deprecate the num_io_threads here
void data_stream::setup(int num_io_threads)
{
  m_base_offset = 0;
  m_sample_stride = 1;
  m_stride_to_next_mini_batch = 0;
  m_stride_to_last_mini_batch = 0;
  m_current_mini_batch_idx = 0;
  m_num_iterations_per_epoch = 0;
  m_global_mini_batch_size = 0;
  m_global_last_mini_batch_size = 0;
  m_world_master_mini_batch_adjustment = 0;

  set_initial_position();

  shuffle_indices();

}

bool data_stream::update(bool is_active_reader)
{
  bool reader_not_done = true; // BVE The sense of this should be fixed
  m_current_mini_batch_idx++;

  if (is_active_reader) {
    m_current_pos = get_next_position();
    m_loaded_mini_batch_idx += m_iteration_stride;
  }
  if (m_loaded_mini_batch_idx >= m_num_iterations_per_epoch) {
    reader_not_done = false;
  }
  if ((size_t)m_current_pos >= m_shuffled_indices.size()) {
    reader_not_done = false;
  }
  if (m_current_mini_batch_idx == m_num_iterations_per_epoch) {
    // for working with 1B jag samples, we may not process all the data
    if ((m_comm->get_rank_in_trainer() < m_num_parallel_readers) &&
        (m_current_pos < (int)m_shuffled_indices.size())) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic data reader update error: the epoch is complete," +
        " but not all of the data has been used -- current pos = " +
        std::to_string(m_current_pos) + " and there are " +
        std::to_string(m_shuffled_indices.size()) + " indices" +
        " : iteration=" + std::to_string(m_current_mini_batch_idx) + "C [" +
        std::to_string(m_loaded_mini_batch_idx) + "L] of " +
        std::to_string(m_num_iterations_per_epoch) + "+" +
        std::to_string(m_iteration_stride) + " : " +
        " index stride=" + std::to_string(m_stride_to_next_mini_batch) + "/" +
        std::to_string(m_stride_to_last_mini_batch));
    }

    shuffle_indices();
    if (priming_data_store()) {
      m_data_store->set_shuffled_indices(&m_shuffled_indices);
    }

    set_initial_position();
  }

  return reader_not_done;
}

int data_stream::get_loaded_mini_batch_size() const
{
  if (m_loaded_mini_batch_idx >= (m_num_iterations_per_epoch - 1)) {
    return m_last_mini_batch_size;
  }
  else {
    return m_mini_batch_size;
  }
}

int data_stream::get_current_mini_batch_size() const
{
  if (m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_last_mini_batch_size + m_world_master_mini_batch_adjustment;
  }
  else {
    return m_mini_batch_size;
  }
}

int data_stream::get_current_global_mini_batch_size() const
{
  if (m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_global_last_mini_batch_size;
  }
  else {
    return m_global_mini_batch_size;
  }
}

/// Returns the current adjustment to the mini-batch size based on if
/// the world master (model 0) has any extra samples
/// Note that any rank in model 0 does not need to add in this offset
/// since the model will already be aware of the extra samples
int data_stream::get_current_world_master_mini_batch_adjustment(
  int model_rank) const
{
  if (model_rank != 0 &&
      m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_world_master_mini_batch_adjustment;
  }
  else {
    return 0;
  }
}

int data_stream::get_next_position() const
{
  /// If the next mini-batch for this rank is going to be the last
  /// mini-batch, take the proper (possibly reduced) step to
  /// setup for the last mini-batch
  if ((m_current_mini_batch_idx + m_iteration_stride - 1) ==
      (m_num_iterations_per_epoch - 1)) {
    return m_current_pos + m_stride_to_last_mini_batch;
  }
  else {
    return m_current_pos + m_stride_to_next_mini_batch;
  }
}

int data_stream::get_num_unused_data(execution_mode m) const
{
  if (m_unused_indices.count(m)) {
    return (int)m_unused_indices.at(m).size();
  }
  else {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }
}
/// Get a pointer to the start of the unused sample indices.
int* data_stream::get_unused_data(execution_mode m)
{
  if (m_unused_indices.count(m)) {
    return &(m_unused_indices[m][0]);
  }
  else {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }
}
const std::vector<int>&
data_stream::get_unused_indices(execution_mode m)
{
  if (m_unused_indices.count(m)) {
    return m_unused_indices.at(m);
  }
  else {
    LBANN_ERROR("Invalid execution mode ", to_string(m), " for unused indices");
  }
}

void data_stream::error_check_counts() const
{
  size_t count = get_absolute_sample_count();
  double use_percent = get_use_percent();
  if (count == 1 and use_percent == 0.0) {
    LBANN_ERROR("get_use_percent() and get_absolute_sample_count() are both "
                "zero; exactly one must be zero");
  }
  if (!(count == 0 or use_percent == 0.0)) {
    LBANN_ERROR("get_use_percent() and get_absolute_sample_count() are both "
                "non-zero; exactly one must be zero");
  }
  if (count != 0) {
    if (count > static_cast<size_t>(get_num_data())) {
      LBANN_ERROR("absolute_sample_count=" + std::to_string(count) +
                  " is > get_num_data=" + std::to_string(get_num_data()));
    }
  }
}

size_t data_stream::get_num_indices_to_use() const
{
  error_check_counts();
  // note: exactly one of the following is guaranteed to be non-zero
  size_t count = get_absolute_sample_count();
  double use_percent = get_use_percent();

  size_t r = 0.;
  if (count) {
    r = count;
  }
  else if (use_percent) {
    r = use_percent * get_num_data();
    if (r == 0) {
      LBANN_ERROR("get_num_indices_to_use() computed zero indices; probably: "
                  "percent_of_data_to_use is too small WRT num_data;  "
                  "get_absolute_sample_count: ",
                  get_absolute_sample_count(),
                  " use_percent: ",
                  get_use_percent(),
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

void data_stream::resize_shuffled_indices()
{
  size_t num_indices = get_num_indices_to_use();
  shuffle_indices();
  m_shuffled_indices.resize(num_indices);
}

void data_stream::select_subset_of_data()
{
  // Calculate the total number of samples for subsets
  double total_split_percent = 0.;
  for (auto m : execution_mode_iterator()) {
    total_split_percent += get_execution_mode_split_percent(m);
  }
  long total_num_data = get_num_data();
  long total_unused = total_split_percent * total_num_data;
  long total_used = total_num_data - total_unused;
  auto starting_unused_offset = m_shuffled_indices.begin() + total_used;
  for (auto m : execution_mode_iterator()) {
    double split_percent = get_execution_mode_split_percent(m);

    if (split_percent == 0.) {
      continue;
    }

    long split = split_percent * total_num_data;
    if (split == 0) {
      LBANN_ERROR(to_string(m),
                  " % of ",
                  split_percent,
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

void data_stream::use_unused_index_set(execution_mode m)
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

bool data_stream::save_to_checkpoint_shared(persist& p,
                                                    execution_mode mode)
{
  if (get_comm()->am_trainer_master()) {
    write_cereal_archive<data_stream>(*this,
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

bool lbann::data_stream::load_from_checkpoint_shared(
  persist& p,
  execution_mode mode)
{
  load_from_shared_cereal_archive<data_stream>(*this,
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

bool data_stream::save_to_checkpoint_distributed(persist& p,
                                                         execution_mode mode)
{
  write_cereal_archive<data_stream>(*this,
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

bool lbann::data_stream::load_from_checkpoint_distributed(
  persist& p,
  execution_mode mode)
{
  read_cereal_archive<data_stream>(*this,
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

void data_stream::set_first_n(int n) { m_first_n = n; }

void data_stream::set_absolute_sample_count(size_t s)
{
  m_absolute_sample_count = s;
}

size_t data_stream::get_absolute_sample_count() const
{
  return m_absolute_sample_count;
}

void data_stream::set_execution_mode_split_percent(execution_mode m,
                                                           double s)
{
  if (s < 0 or s > 1.0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: set_validation_percent() - must be: s >= 0, s <= 1.0; you passed: " +
      std::to_string(s));
  }
  m_execution_mode_split_percentage[m] = s;
}

double
data_stream::get_execution_mode_split_percent(execution_mode m) const
{
  if (m_execution_mode_split_percentage.count(m)) {
    return m_execution_mode_split_percentage.at(m);
  }
  else {
    return 0;
  }
}

void data_stream::set_use_percent(double s)
{
  if (s < 0 or s > 1.0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: set_use_percent() - must be: s >= 0, s <= 1.0; you passed: " +
      std::to_string(s));
  }
  m_use_percent = s;
}

double data_stream::get_use_percent() const { return m_use_percent; }

void data_stream::set_mini_batch_size(const int s)
{
  m_mini_batch_size = s;
}

void data_stream::set_role(std::string role) { m_role = role; }

} // namespace lbann
