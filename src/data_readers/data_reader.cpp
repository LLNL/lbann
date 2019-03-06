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
//
// lbann_data_reader .hpp .cpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_store/generic_data_store.hpp"
#include "lbann/utils/omp_pragma.hpp"
#include "lbann/models/model.hpp"
#include <omp.h>
#include <future>

namespace lbann {

#undef DEBUG
//#define DEBUG

void generic_data_reader::shuffle_indices() {
  shuffle_indices(get_data_seq_generator());
}

void generic_data_reader::shuffle_indices(rng_gen& gen) {
  // Shuffle the data
  if (m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(),
                 gen);
  }
}

  void generic_data_reader::setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) {
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

  m_thread_buffer.resize(num_io_threads, std::vector<char>());
  for(int tid = 0; tid < num_io_threads; ++tid) {
    m_thread_buffer[tid].resize(get_linearized_data_size());
  }
  m_io_thread_pool = io_thread_pool;
}


bool lbann::generic_data_reader::fetch_data_block(CPUMat& X, El::Int thread_id, El::Int mb_size, El::Matrix<El::Int>& indices_fetched) {
  std::string error_message;
  for (int s = thread_id; s < mb_size; s+=m_io_thread_pool->get_num_threads()) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    bool valid = fetch_datum(X, index, s);
    if (!valid) {
      error_message = "invalid datum (index " + std::to_string(index) + ")";
    }
    if (!error_message.empty()) { LBANN_ERROR(error_message); }
    indices_fetched.Set(s, 0, index);
  }
  return true;
}

int lbann::generic_data_reader::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
  #ifdef DEBUG
  if (m_current_pos == 0) {
    if (is_master()) {
      std::cout << "role: " << get_role() << " model: " << m_model->get_name()
                << " shuffled indices: ";
      for (size_t j=0; j<15; j++) {
        std::cout << m_shuffled_indices[j] << " ";
      }
      std::cout << "\n";
    }
  }
  #endif

  int loaded_batch_size = get_loaded_mini_batch_size();

  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size), m_shuffled_indices.size());
  const int mb_size = std::min(El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
      X.Width());

  El::Zeros_seq(X, X.Height(), X.Width());
  El::Zeros_seq(indices_fetched, mb_size, 1);

  /// Make sure that every rank participates in the data store prior
  /// to seeing if the local rank's position is valid.  Note that
  /// every rank will hold data that may be used in the last mini-batch
  if (data_store_active()) {
    m_data_store->exchange_mini_batch_data(m_current_pos-m_base_offset-m_model_offset, loaded_batch_size);
  }

  if(!position_valid()) {
    if(position_is_overrun()) {
      return 0;
    }else {
      LBANN_ERROR(std::string{} + "generic data reader load error: !position_valid"
                  + " -- current pos = " + std::to_string(m_current_pos)
                  + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
    }
  }

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads()); t++) {
    preprocess_data_source(t);
  }

  static bool fix_jag = true;
  if (m_jag_partitioned && fix_jag) {
    fix_jag = false;
    set_jag_variables(mb_size);
  }

  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads()); t++) {
    // Queue up work into other threads and then finish off the
    // mini-batch in the active thread
    if(t == m_io_thread_pool->get_local_thread_id()) {
      continue;
    }else {
      m_io_thread_pool->submit_job_to_work_group(
        std::bind(&generic_data_reader::fetch_data_block, this, std::ref(X), t,
                  mb_size, std::ref(indices_fetched)));
    }
  }
  fetch_data_block(X, m_io_thread_pool->get_local_thread_id(), mb_size, indices_fetched);

  // Wait for all of the threads to finish
  m_io_thread_pool->finish_work_group();

  /// Allow each thread to perform any postprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < static_cast<int>(m_io_thread_pool->get_num_threads()); t++) {
    postprocess_data_source(t);
  }

  return mb_size;
}

void lbann::generic_data_reader::set_jag_variables(int mb_size) {
  // all min_batches have the same number of indices;
  // this probably causes a few indices to be discarded,
  // but with 1B indices, who cares?
  int mb_max = m_comm->trainer_allreduce<int>(mb_size, El::mpi::MAX);
  m_num_iterations_per_epoch = m_shuffled_indices.size() / mb_max;

  m_last_mini_batch_size = m_mini_batch_size;
  m_global_mini_batch_size = m_mini_batch_size;
  m_global_last_mini_batch_size = m_mini_batch_size;

  m_reset_mini_batch_index = 0;
  m_loaded_mini_batch_idx = 0;
  m_current_mini_batch_idx = 0;

  m_stride_to_next_mini_batch = mb_size;
  m_stride_to_last_mini_batch = mb_size;

  m_base_offset = 0;
  m_model_offset = 0;
  m_sample_stride = 1;
  m_iteration_stride = 1;

  m_world_master_mini_batch_adjustment = 0;
}

int lbann::generic_data_reader::fetch_labels(CPUMat& Y) {
  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    Y.Width());

  El::Zeros_seq(Y, Y.Height(), Y.Width());

  if(!position_valid()) {
    if(position_is_overrun()) {
      return 0;
    }else {
      LBANN_ERROR(std::string{} + "generic data reader load error: !position_valid"
                  + " -- current pos = " + std::to_string(m_current_pos)
                  + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
    }
  }

  std::string error_message;
  for (int s = 0; s < mb_size; s++) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    bool valid = fetch_label(Y, index, s);
    if (!valid) {
      error_message = "invalid label (index " + std::to_string(index) + ")";
    }
  }
  if (!error_message.empty()) { LBANN_ERROR(error_message); }

  return mb_size;
}

int lbann::generic_data_reader::fetch_responses(CPUMat& Y) {
  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    Y.Width());

  El::Zeros_seq(Y, Y.Height(), Y.Width());

  if(!position_valid()) {
    if(position_is_overrun()) {
      return 0;
    }else {
      LBANN_ERROR(std::string{} + "generic data reader load error: !position_valid"
                  + " -- current pos = " + std::to_string(m_current_pos)
                  + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
    }
  }

  std::string error_message;
  for (int s = 0; s < mb_size; s++) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    bool valid = fetch_response(Y, index, s);
    if (!valid) {
      error_message = "invalid response (index " + std::to_string(index) + ")";
    }
  }
  if (!error_message.empty()) { LBANN_ERROR(error_message); }
  return mb_size;
}

bool generic_data_reader::update(bool is_active_reader) {
  bool reader_not_done = true; // BVE The sense of this should be fixed
  m_current_mini_batch_idx++;

  if(is_active_reader) {
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
    if ((get_rank() < m_num_parallel_readers) && (m_current_pos < (int)m_shuffled_indices.size()) && !m_jag_partitioned) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: generic data reader update error: the epoch is complete,"
        + " but not all of the data has been used -- current pos = " + std::to_string(m_current_pos)
        + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices"
        + " : iteration="
        + std::to_string(m_current_mini_batch_idx) + "C ["
        + std::to_string(m_loaded_mini_batch_idx) +"L] of "
        + std::to_string(m_num_iterations_per_epoch) + "+"
        + std::to_string(m_iteration_stride) + " : "
        + " index stride="
        + std::to_string(m_stride_to_next_mini_batch) + "/"
        + std::to_string(m_stride_to_last_mini_batch));
    }

    shuffle_indices();
    if (priming_data_store()) {
      m_data_store->set_shuffled_indices(&m_shuffled_indices);
    }

    set_initial_position();

  }

  post_update();

  return reader_not_done;
}

int generic_data_reader::get_loaded_mini_batch_size() const {
  if (m_loaded_mini_batch_idx >= (m_num_iterations_per_epoch-1)) {
    return m_last_mini_batch_size;
  } else {
    return m_mini_batch_size;
  }
}

int generic_data_reader::get_current_mini_batch_size() const {
  if (m_current_mini_batch_idx == (m_num_iterations_per_epoch-1)) {
    return m_last_mini_batch_size + m_world_master_mini_batch_adjustment;
  } else {
    return m_mini_batch_size;
  }
}

int generic_data_reader::get_current_global_mini_batch_size() const {
  if (m_current_mini_batch_idx == (m_num_iterations_per_epoch-1)) {
    return m_global_last_mini_batch_size;
  } else {
    return m_global_mini_batch_size;
  }
}

/// Returns the current adjustment to the mini-batch size based on if
/// the world master (model 0) has any extra samples
/// Note that any rank in model 0 does not need to add in this offset
/// since the model will already be aware of the extra samples
int generic_data_reader::get_current_world_master_mini_batch_adjustment(int model_rank) const {
  if (model_rank != 0 && m_current_mini_batch_idx == (m_num_iterations_per_epoch-1)) {
    return m_world_master_mini_batch_adjustment;
  } else {
    return 0;
  }
}

int generic_data_reader::get_next_position() const {
  /// If the next mini-batch for this rank is going to be the last
  /// mini-batch, take the proper (possibly reduced) step to
  /// setup for the last mini-batch
  if ((m_current_mini_batch_idx + m_iteration_stride - 1) == (m_num_iterations_per_epoch-1)) {
    return m_current_pos + m_stride_to_last_mini_batch;
  } else {
    return m_current_pos + m_stride_to_next_mini_batch;
  }
}

void generic_data_reader::select_subset_of_data_partitioned() {

  //sanity checks
  if (get_absolute_sample_count()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_reader - absolute_sample_count is not supported "
      + "for partitioned data_set");
  }
  double use_percent = get_use_percent();
  if (use_percent <= 0.0 || use_percent > 1.0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_reader - percent_of_data_to_use must be > 0 "
      + "and <= 1");
  }
  if (! (m_partition_mode == 1 || m_partition_mode == 2)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_reader - overlap mode must be 1 or 2\n"
      " 1 - share overlap data with one neighboring models;\n"
      " 2 - a set of overlap indices is common to (is shared by) all models");
  }

  shuffle_indices();

  //optionally only use a portion of the data (useful during development
  //and testing)
  m_shuffled_indices.resize( get_use_percent() * m_shuffled_indices.size());

  std::vector<int> common_pool;
  //case where there's an overlap set that is common to all models
  if (m_partition_overlap && m_partition_mode == 2) {
    // Let x be the percent of indices from shuffled_indices that will be
    //   assigned to the common pool.
    // Let p be the number of models.
    // Let v be the requested percent overlap.
    // Let n = m_shuffled_indices.size(). Then each  model will have
    //  xn + n(1-x)/p indices, and we want:
    //   xn / ( xn + n(1-x)/p ) = v solving for x:
    //
    //         x = v / (-pv+p+v)
    //
    double v = m_partition_overlap;
    double p = m_num_partitions;
    double x = v / (-p*v + p + v);
    int x1 = x*(m_shuffled_indices.size() - get_validation_percent()*m_shuffled_indices.size());
    if (x1 < 1) {
      x1 = 1;
    }
    int x3 = m_shuffled_indices.size() - x1;
    common_pool.resize(x1);
    std::copy(
      m_shuffled_indices.begin() + x3,
      m_shuffled_indices.end(),
      common_pool.begin());
    m_shuffled_indices.resize(x3);
  }

  // hack: possibly drop a few indices to avoid dealing with edge cases;
  // number dropped is less than the number of models
  size_t partition_size = m_shuffled_indices.size() / m_num_partitions;
  if (partition_size*m_num_partitions < m_shuffled_indices.size() && is_master()) {
    std::cout
      << "select_subset_of_data_partitioned; data set is partitioned; dropping "
      << m_shuffled_indices.size() - (partition_size*m_num_partitions)
      << " to avoid dealing with edge cases (hack)\n";
  }

  // make temp copy of indices; need this to compute overlap for mode 1 (below)
  std::vector<int> s_indices = m_shuffled_indices;

  //partition the data
  if (m_my_partition > 0) {
    std::copy(
      m_shuffled_indices.begin() + partition_size*m_my_partition,
      m_shuffled_indices.begin() + partition_size*(m_my_partition+1),
      m_shuffled_indices.begin());
  }
  m_shuffled_indices.resize(partition_size);

  //pull out validation set; note that we pull the validation set from
  //the end of the index vector
  long unused = get_validation_percent()*get_num_data();
  long use_me = get_num_data() - unused;
  if (unused > 0) {
      m_unused_indices=std::vector<int>(m_shuffled_indices.begin() + use_me, m_shuffled_indices.end());
      m_shuffled_indices.resize(use_me);
  }

  int shared_index_count = common_pool.size();
  if (m_partition_overlap > 0.) {
    if (m_partition_overlap > 1.) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic_data_reader - overlap must be >= 0 and <= 1");
    }

    if (m_partition_mode == 2) {
      int s = m_shuffled_indices.size();
      m_shuffled_indices.resize(s + common_pool.size());
      std::copy(common_pool.begin(), common_pool.end(), m_shuffled_indices.begin() + s);
    }

    else { //m_partition_mode = 1 or 3

      double x = m_partition_overlap / (1-m_partition_overlap);
      size_t overlap_count = x*use_me;

      //ensure there's at least one overlap at each end of a proc's partition;
      //this is only needed to ensure that, when testing with smallish data sets,
      //rounding error doesn't set overlap to 0.
      if (overlap_count < 2) {
        overlap_count = 2;
      }
      //we exchange 1/2 of the overlap with left & right nabore
      overlap_count /= 2;

      size_t start_of_prior_partition = (m_my_partition-1)*partition_size;
      if (m_my_partition == 0) {
        start_of_prior_partition = (m_num_partitions-1)*partition_size;
      }
      size_t start_of_next_partition = (m_my_partition+1)*partition_size;
      if (m_my_partition == m_num_partitions-1) {
        start_of_next_partition = 0;
      }

      shared_index_count = 0;
      for (size_t j = 0; j<overlap_count; j++) {
        m_shuffled_indices.push_back(s_indices[start_of_prior_partition+j]);
        ++shared_index_count;
      }
      for (size_t j = 0; j<overlap_count; j++) {
        m_shuffled_indices.push_back(s_indices[start_of_next_partition+j]);
        ++shared_index_count;
      }
    }
    if (is_master()) {
      double s = 100.0 * shared_index_count / m_shuffled_indices.size();
      std::cout << "Actual overlap percentage: " << s << "%\n";
    }
  }
}

void generic_data_reader::select_subset_of_data() {
  // ensure that all readers have the same number of indices
  if (m_jag_partitioned) {
    size_t n = m_comm->trainer_allreduce<size_t>(m_shuffled_indices.size(), El::mpi::MIN);
    m_shuffled_indices.resize(n);
  }

  // optionally partition data set amongst the models
  if (m_is_partitioned) {
    select_subset_of_data_partitioned();
    return ;
  }

  shuffle_indices();

  size_t count = get_absolute_sample_count();
  double use_percent = get_use_percent();
  if (count == 0 and use_percent == 0.0) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic_data_reader::select_subset_of_data() get_use_percent() "
        + "and get_absolute_sample_count() are both zero; exactly one "
        + "must be zero");
  }
  if (!(count == 0 or use_percent == 0.0)) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic_data_reader::select_subset_of_data() get_use_percent() "
        "and get_absolute_sample_count() are both non-zero; exactly one "
        "must be zero");
  }

  if (count != 0) {
    if(count > static_cast<size_t>(get_num_data())) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic_data_reader::select_subset_of_data() - absolute_sample_count=" +
        std::to_string(count) + " is > get_num_data=" +
        std::to_string(get_num_data()));
    }
    m_shuffled_indices.resize(get_absolute_sample_count());
  }

  if (use_percent) {
    m_shuffled_indices.resize(get_use_percent()*get_num_data());
  }

  long unused = get_validation_percent()*get_num_data(); //get_num_data() = m_shuffled_indices.size()
  long use_me = get_num_data() - unused;
  if (unused > 0) {
      m_unused_indices=std::vector<int>(m_shuffled_indices.begin() + use_me, m_shuffled_indices.end());
      m_shuffled_indices.resize(use_me);
  }

  if(!m_shuffle) {
    std::sort(m_shuffled_indices.begin(), m_shuffled_indices.end());
    std::sort(m_unused_indices.begin(), m_unused_indices.end());
  }
}

void generic_data_reader::use_unused_index_set() {
  m_shuffled_indices.swap(m_unused_indices);
  m_unused_indices.clear();
  std::vector<int>().swap(m_unused_indices); // Trick to force memory reallocation
}

/** \brief Given directory to store checkpoint files, write state to file and add to number of bytes written */
bool generic_data_reader::save_to_checkpoint_shared(persist& p, const char *name) {
  // rank 0 writes the training state file
  if (m_comm->am_trainer_master()) {
    pack_scalars(p,name);
  }
  return true;
}

/** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
bool lbann::generic_data_reader::load_from_checkpoint_shared(persist& p, const char *name) {
  // rank 0 reads the training state file
  struct packing_header header;
  if (m_comm->am_trainer_master()) {
    unpack_scalars(p,&header,name);
  }
  m_comm->trainer_broadcast(0, header);
  unpack_header(header);

  m_comm->trainer_broadcast(0, m_shuffled_indices);

  // Adjust current position to deal with fact that it was just loaded to all ranks from rank 0 (differs by rank #)
  m_current_pos += m_comm->get_rank_in_trainer();
  return true;
}

bool generic_data_reader::save_to_checkpoint_distributed(persist& p, const char *name) {
  pack_scalars(p,name);
  return true;
}

bool lbann::generic_data_reader::load_from_checkpoint_distributed(persist& p, const char *name) {
  struct packing_header header;
  unpack_scalars(p,&header,name);
  return true;
}

void generic_data_reader::set_file_dir(std::string s) {
  if(endsWith(s, "/")) {
    m_file_dir = s;
  }else {
    m_file_dir = s + "/";
  }
}

void generic_data_reader::set_local_file_dir(std::string s) {
  if(endsWith(s, "/")) {
    m_local_file_dir = s;
  }else {
    m_local_file_dir = s + "/";
  }
}

std::string generic_data_reader::get_file_dir() const {
  return m_file_dir;
}

std::string generic_data_reader::get_local_file_dir() const {
  return m_local_file_dir;
}

void generic_data_reader::set_data_index_list(std::string s) {
  m_data_index_list = s;
}

std::string generic_data_reader::get_data_index_list() const {
  if (m_data_index_list == "") {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you apparently did not call set_data_index_list; error!");
  }
  return m_data_index_list;
}

void generic_data_reader::set_data_filename(std::string s) {
  m_data_fn = s;
}

std::string generic_data_reader::get_data_filename() const {
  if (m_data_fn == "") {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you apparently did not call set_data_filename; error!");
  }
  return m_data_fn;
}

void generic_data_reader::set_label_filename(std::string s) {
  m_label_fn = s;
}

std::string generic_data_reader::get_label_filename() const {
  if (m_label_fn == "") {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you apparently did not call set_label_filename; error!");
  }
  return m_label_fn;
}

void generic_data_reader::set_first_n(int n) {
  m_first_n = n;
}

void generic_data_reader::set_absolute_sample_count(size_t s) {
  m_absolute_sample_count = s;
}

size_t generic_data_reader::get_absolute_sample_count() const {
  return m_absolute_sample_count;
}


void generic_data_reader::set_validation_percent(double s) {
  if (s < 0 or s > 1.0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: set_validation_percent() - must be: s >= 0, s <= 1.0; you passed: " +
      std::to_string(s));
  }
  m_validation_percent = s;
}


double generic_data_reader::get_validation_percent() const {
  return m_validation_percent;
}

void generic_data_reader::set_use_percent(double s) {
  if (s < 0 or s > 1.0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: set_use_percent() - must be: s >= 0, s <= 1.0; you passed: " +
      std::to_string(s));
  }
  m_use_percent = s;
}


double generic_data_reader::get_use_percent() const {
  return m_use_percent;
}

void generic_data_reader::setup_data_store(model *m, int mini_batch_size) {
  m_data_store = nullptr;
}

bool generic_data_reader::data_store_active() const {
  return (m_data_store != nullptr
          && (m_model->get_execution_mode() == execution_mode::training)
          && m_model->get_epoch() > 0);
}

bool generic_data_reader::priming_data_store() const {
  return (m_data_store != nullptr
          && (m_model->get_execution_mode() == execution_mode::training)
          && m_model->get_epoch() == 0);
}

void generic_data_reader::set_data_store(generic_data_store *g) {
    if (m_data_store != nullptr) {
      delete m_data_store;
    }
    m_data_store = g;
}

void generic_data_reader::set_partitioned(bool partitioned_yes, double overlap, int mode) {
  if (m_comm->get_num_trainers() == 1 || m_comm->get_procs_in_world() == 1) {
    m_is_partitioned  = false;
    return;
  }
  m_is_partitioned = partitioned_yes;
  //n.b. the following params have no affect if m_is_partitioned is false
  m_partition_overlap = overlap;
  m_partition_mode = mode;
  m_procs_per_partition = m_comm->get_procs_per_trainer();
  m_num_partitions = m_comm->get_num_trainers();
  m_my_partition = m_comm->get_trainer_rank();
}

}  // namespace lbann
