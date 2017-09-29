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
#include <omp.h>

namespace lbann {

void generic_data_reader::setup() {
  m_base_offset = 0;
  m_sample_stride = 1;
  m_mini_batch_stride = 0;
  m_last_mini_batch_stride = 0;
  m_current_mini_batch_idx = 0;
  m_num_iterations_per_epoch = 0;
  m_global_mini_batch_size = 0;
  m_global_last_mini_batch_size = 0;

  /// The amount of space needed will vary based on input layer type,
  /// but the batch size is the maximum space necessary
  El::Zeros(m_indices_fetched_per_mb, m_mini_batch_size, 1);

  set_initial_position();

  // Shuffle the data
  if (not m_first_n) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(),
                 get_data_seq_generator());
  }

}

int lbann::generic_data_reader::fetch_data(Mat& X) {
  int nthreads = omp_get_max_threads();
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: generic data reader load error: !position_valid"
      + " -- current pos = " + std::to_string(m_current_pos)
      + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
  }

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  #pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nthreads; t++) {
    preprocess_data_source(omp_get_thread_num());
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    X.Width());

  El::Zeros(X, X.Height(), X.Width());
  El::Zeros(m_indices_fetched_per_mb, mb_size, 1);
  #pragma omp parallel for
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
    } catch (exception& e) {
      El::ReportException(e);
    }
  }

  /// Allow each thread to perform any postprocessing necessary on the
  /// data source prior to fetching data
  #pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nthreads; t++) {
    postprocess_data_source(omp_get_thread_num());
  }

  return mb_size;
}

int lbann::generic_data_reader::fetch_labels(Mat& Y) {
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
  #pragma omp parallel for
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
    } catch (exception& e) {
      El::ReportException(e);
    }
  }
  return mb_size;
}

int lbann::generic_data_reader::fetch_responses(Mat& Y) {
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
  #pragma omp parallel for
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
    } catch (exception& e) {
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

    /// Maintain the current width of the matrix
    El::Zeros(m_indices_fetched_per_mb, m_indices_fetched_per_mb.Width(), 1);

    m_loaded_mini_batch_idx += m_iteration_stride;
  }
  if (m_loaded_mini_batch_idx >= m_num_iterations_per_epoch) {
    reader_not_done = false;
  }
  if (m_current_mini_batch_idx == m_num_iterations_per_epoch) {
    if (m_current_pos < (int)m_shuffled_indices.size()) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: generic data reader update error: the epoch is complete,"
        + " but not all of the data has been used -- current pos = " + std::to_string(m_current_pos)
        + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
    }
    if (not m_first_n) {
      std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(),
                   get_data_seq_generator());
    }
    set_initial_position();
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
    return m_last_mini_batch_size;
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

int generic_data_reader::get_next_position() const {
  /// Is the mini-batch being loaded corresponds to the second to last mini-batch
  /// If so, get the last mini-batch stride
  if (m_loaded_mini_batch_idx >= (m_num_iterations_per_epoch-2)) {
    return m_current_pos + m_last_mini_batch_stride;
  } else {
    return m_current_pos + m_mini_batch_stride;
  }
}

void generic_data_reader::select_subset_of_data() {
  if(!get_firstN()) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(), get_data_seq_generator());
  }

  if (not (has_max_sample_count() or has_use_percent() or has_validation_percent())) {
    return;
  }

  if (has_max_sample_count()) {
    size_t count = get_max_sample_count();
    if(count > static_cast<size_t>(get_num_data())) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: generic_data_reader::select_subset_of_data() - max_sample_count=" +
        std::to_string(count) + " is > get_num_data=" +
        std::to_string(get_num_data()));
    }
    m_shuffled_indices.resize(get_max_sample_count());
  } else if (has_use_percent()) {
    m_shuffled_indices.resize(get_use_percent()*get_num_data());
  }

  if (has_validation_percent()) {
    long unused = get_validation_percent()*get_num_data(); //get_num_data() = m_shuffled_indices.size()
    long use_me = get_num_data() - unused;
    if (unused > 0) {
      m_unused_indices=std::vector<int>(m_shuffled_indices.begin() + use_me, m_shuffled_indices.end());
      m_shuffled_indices.resize(use_me);
    }
  }

  if(get_firstN()) {
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
bool generic_data_reader::saveToCheckpointShared(persist& p, const char *name) {
  // rank 0 writes the training state file
  if (p.get_rank() == 0) {
    char fieldname[1024];

    // record minibatch index
    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_current_mini_batch_idx);

    // get size of list of training examples
    int size = m_shuffled_indices.size();

    // record size of ShuffleIndices
    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) size);

    // TODO: each model may have a different position, need to gather and write these
    // record current position within training data
    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_current_pos);

    // write list of indices
    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.write_int32_contig(persist_type::train, fieldname, &m_shuffled_indices[0], (uint64_t) size);
  }

  return true;
}

/** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
bool lbann::generic_data_reader::loadFromCheckpointShared(persist& p, const char *name) {
  // rank 0 reads the training state file
  if (p.get_rank() == 0) {
    char fieldname[1024];

    // record minibatch index
    uint64_t val;
    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_current_mini_batch_idx = (int) val;

    // get size of ShuffleIndices
    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    int size = (int) val;

    // get current position within data
    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_current_pos = (int) val;

    // resize shuffled index array to hold values
    m_shuffled_indices.resize(size);

    // read list of indices
    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.read_int32_contig(persist_type::train, fieldname, &m_shuffled_indices[0], (uint64_t) size);
  }

  // broadcast minibatch index
  MPI_Bcast(&m_current_mini_batch_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // TODO: with multiple readers, make this a scatter
  // broadcast current position
  MPI_Bcast(&m_current_pos, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // broadcast values from rank 0
  int size = m_shuffled_indices.size();
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // resize shuffled index array to hold values
  if (p.get_rank() != 0) {
    m_shuffled_indices.resize(size);
  }

  // broadcast index array
  MPI_Bcast(&m_shuffled_indices[0], size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void generic_data_reader::set_file_dir(std::string s) {
  m_file_dir = s;
}

std::string generic_data_reader::get_file_dir() const {
  return m_file_dir;
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

string generic_data_reader::get_label_filename() const {
  if (m_label_fn == "") {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you apparently did not call set_label_filename; error!");
  }
  return m_label_fn;
}

void generic_data_reader::set_max_sample_count(size_t s) {
  m_max_sample_count = s;
  m_max_sample_count_was_set = true;
}

size_t generic_data_reader::get_max_sample_count() const {
  return m_max_sample_count;
}

bool generic_data_reader::has_max_sample_count() const {
  return m_max_sample_count_was_set;
}

void generic_data_reader::set_firstN(bool b) {
  m_first_n = b;
}

bool generic_data_reader::get_firstN() const {
  return m_first_n;
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

bool generic_data_reader::has_validation_percent() const {
  if (m_validation_percent == -1) {
    return false;
  }
  return true;
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

bool generic_data_reader::has_use_percent() const {
  if (m_use_percent == -1) {
    return false;
  }
  return true;
}

double generic_data_reader::get_use_percent() const {
  if (!has_use_percent()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: you must call set_use_percent(); error!");
  }
  return m_use_percent;
}

}  // namespace lbann
