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
#include <omp.h>

namespace lbann {

void generic_data_reader::shuffle_indices() {
  // Shuffle the data
  if (m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(),
                 get_data_seq_generator());
  }
}

void generic_data_reader::setup() {
  m_base_offset = 0;
  m_sample_stride = 1;
  m_stride_to_next_mini_batch = 0;
  m_stride_to_last_mini_batch = 0;
  m_current_mini_batch_idx = 0;
  m_num_iterations_per_epoch = 0;
  m_global_mini_batch_size = 0;
  m_global_last_mini_batch_size = 0;
  m_world_master_mini_batch_adjustment = 0;

  /// The amount of space needed will vary based on input layer type,
  /// but the batch size is the maximum space necessary
  El::Zeros(m_indices_fetched_per_mb, m_mini_batch_size, 1);

  set_initial_position();

  shuffle_indices();
}

int lbann::generic_data_reader::fetch_data(CPUMat& X) {
  int nthreads = omp_get_max_threads();
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: generic data reader load error: !position_valid"
      + " -- current pos = " + std::to_string(m_current_pos)
      + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
  }

  if (!m_save_minibatch_indices) {
    /// Allow each thread to perform any preprocessing necessary on the
    /// data source prior to fetching data
    LBANN_OMP_TASKLOOP
    for (int t = 0; t < nthreads; t++) {
      preprocess_data_source(omp_get_thread_num());
    }
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    X.Width());

  if (!m_save_minibatch_indices) {
    El::Zeros(X, X.Height(), X.Width());
    El::Zeros(m_indices_fetched_per_mb, mb_size, 1);
  }

  if (m_save_minibatch_indices) {
    m_my_minibatch_indices.resize(m_my_minibatch_indices.size() + 1);
    for (int s = 0; s < mb_size; s++) {
      int n = m_current_pos + (s * m_sample_stride);
      m_my_minibatch_indices.back().push_back(n);
    }
  }

  else {
    LBANN_OMP_TASKLOOP
    for (int s = 0; s < mb_size; s++) {
      // Catch exceptions within the OpenMP thread.
      try {
        int n = m_current_pos + (s * m_sample_stride);
        int index = m_shuffled_indices[n];
        bool valid = fetch_datum(X, index, s, omp_get_thread_num());
        if (!valid) {
          throw lbann_exception(
            std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
            " :: generic data reader load error: datum not valid");
        }
        m_indices_fetched_per_mb.Set(s, 0, index);
      } catch (lbann_exception& e) {
        lbann_report_exception(e);
      } catch (std::exception& e) {
        El::ReportException(e);
      }
    }

    /// Allow each thread to perform any postprocessing necessary on the
    /// data source prior to fetching data
    LBANN_OMP_TASKLOOP
    for (int t = 0; t < nthreads; t++) {
      postprocess_data_source(omp_get_thread_num());
    }
  }

  return mb_size;
}

int lbann::generic_data_reader::fetch_labels(CPUMat& Y) {
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic data reader load error: !position_valid");
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    Y.Width());

  El::Zeros(Y, Y.Height(), Y.Width());

//  if (m_data_store != nullptr) {
    //@todo: get it to work, then add omp support
    //m_data_store->fetch_labels(...);
 // }

//  else {
    LBANN_OMP_TASKLOOP
    for (int s = 0; s < mb_size; s++) {
      // Catch exceptions within the OpenMP thread.
      try {
        int n = m_current_pos + (s * m_sample_stride);
        int index = m_shuffled_indices[n];

        bool valid = fetch_label(Y, index, s, omp_get_thread_num());
        if (!valid) {
          throw lbann_exception(
            std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
            " :: generic data reader load error: label not valid");
        }
      } catch (lbann_exception& e) {
        lbann_report_exception(e);
      } catch (std::exception& e) {
        El::ReportException(e);
      }
    }
  //}
  return mb_size;
}

int lbann::generic_data_reader::fetch_responses(CPUMat& Y) {
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic data reader load error: !position_valid");
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    Y.Width());

  El::Zeros(Y, Y.Height(), Y.Width());
  LBANN_OMP_TASKLOOP
  for (int s = 0; s < mb_size; s++) {
    // Catch exceptions within the OpenMP thread.
    try {
      int n = m_current_pos + (s * m_sample_stride);
      int index = m_shuffled_indices[n];

      bool valid = fetch_response(Y, index, s, omp_get_thread_num());
      if (!valid) {
        throw lbann_exception(
          std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
          " :: generic data reader load error: response not valid");
      }
    } catch (lbann_exception& e) {
      lbann_report_exception(e);
    } catch (std::exception& e) {
      El::ReportException(e);
    }
  }
  return mb_size;
}

bool generic_data_reader::is_data_reader_done(bool is_active_reader) {
  bool reader_not_done = true;
  if(is_active_reader) {
    reader_not_done = !((m_loaded_mini_batch_idx + m_iteration_stride) >= m_num_iterations_per_epoch);
  }else {
    reader_not_done = !(m_loaded_mini_batch_idx >= m_num_iterations_per_epoch);
  }
  return reader_not_done;
}

bool generic_data_reader::update(bool is_active_reader) {
  bool reader_not_done = true; // BVE The sense of this should be fixed
  m_current_mini_batch_idx++;

  if(is_active_reader) {
    m_current_pos = get_next_position();

    /// Maintain the current height of the matrix
    if (!m_save_minibatch_indices) {
      El::Zeros(m_indices_fetched_per_mb, m_indices_fetched_per_mb.Height(), 1);
    }

    m_loaded_mini_batch_idx += m_iteration_stride;
  }
  if (m_loaded_mini_batch_idx >= m_num_iterations_per_epoch) {
    reader_not_done = false;
  }
  if ((size_t)m_current_pos >= m_shuffled_indices.size()) {
    reader_not_done = false;
  }
  if (m_current_mini_batch_idx == m_num_iterations_per_epoch) {
    if ((get_rank() < m_num_parallel_readers) && (m_current_pos < (int)m_shuffled_indices.size())) {
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

    if (!m_save_minibatch_indices) {
      shuffle_indices();
    }

    set_initial_position();

    if (!m_save_minibatch_indices) {
      if (m_data_store) {
        m_data_store->set_shuffled_indices(&m_shuffled_indices);
      }
    }
  }
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

void generic_data_reader::select_subset_of_data() {
  m_num_global_indices = m_shuffled_indices.size();
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
  if (m_comm->am_model_master()) {
    pack_scalars(p,name);
  }
  return true;
}

/** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
bool lbann::generic_data_reader::load_from_checkpoint_shared(persist& p, const char *name) {
  // rank 0 reads the training state file
  struct packing_header header;
  if (m_comm->am_model_master()) {
    unpack_scalars(p,&header,name);
  }
  m_comm->model_broadcast(0, header);
  unpack_header(header);

  m_comm->model_broadcast(0, m_shuffled_indices);

  // Adjust current position to deal with fact that it was just loaded to all ranks from rank 0 (differs by rank #)
  m_current_pos += m_comm->get_rank_in_model();
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

void generic_data_reader::setup_data_store(model *m) {
  m_data_store = nullptr;
}

void generic_data_reader::set_save_minibatch_entries(bool b) {
  m_save_minibatch_indices = b;
  if (b) {
    m_my_minibatch_indices.reserve(get_num_iterations_per_epoch());
  }
}

void generic_data_reader::set_data_store(generic_data_store *g) {
    if (m_data_store != nullptr) {
      delete m_data_store;
    }
    m_data_store = g;
}

void generic_data_reader::init_minibatch() {
  if (m_data_store != nullptr) {
    m_data_store->init_minibatch();
  }
}

}  // namespace lbann
