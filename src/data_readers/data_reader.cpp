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
#include "lbann/data_store/data_store_imagenet.hpp"
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include <omp.h>

namespace lbann {

void generic_data_reader::setup() {
  m_base_offset = 0;
  m_sample_stride = 1;
  m_stride_to_next_mini_batch = 0;
  m_stride_to_last_mini_batch = 0;
  m_current_mini_batch_idx = 0;
  m_num_iterations_per_epoch = 0;
  m_global_mini_batch_size = 0;
  m_global_last_mini_batch_size = 0;

  /// The amount of space needed will vary based on input layer type,
  /// but the batch size is the maximum space necessary
  El::Zeros(m_indices_fetched_per_mb, m_mini_batch_size, 1);

  set_initial_position();

  // Shuffle the data
  if (m_shuffle) {
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
    } catch (std::exception& e) {
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

//  if (m_data_store != nullptr) {
    //@todo: get it to work, then add omp support
    //m_data_store->fetch_labels(...);
 // }

//  else {
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
      } catch (std::exception& e) {
        El::ReportException(e);
      }
    }
  //}
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

bool generic_data_reader::update(bool is_active_reader, bool fake) {
  bool reader_not_done = true; // BVE The sense of this should be fixed
  m_current_mini_batch_idx++;

  if(is_active_reader) {
    m_current_pos = get_next_position();

    /// Maintain the current height of the matrix
    if (!fake) {
      El::Zeros(m_indices_fetched_per_mb, m_indices_fetched_per_mb.Height(), 1);
    }  

    m_loaded_mini_batch_idx += m_iteration_stride;
  }
  if (m_loaded_mini_batch_idx >= m_num_iterations_per_epoch) {
    reader_not_done = false;
  }
  if (fake) {
    if ((size_t)m_current_pos >= m_shuffled_indices.size()) {
      reader_not_done = false;
    }  
  }
  if (m_current_mini_batch_idx == m_num_iterations_per_epoch) {
    if ((get_rank() < m_num_parallel_readers) && (m_current_pos < (int)m_shuffled_indices.size())) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: generic data reader update error: the epoch is complete,"
        + " but not all of the data has been used -- current pos = " + std::to_string(m_current_pos)
        + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
    }

    if (!fake) {
      if (m_shuffle) {
        std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(),
                     get_data_seq_generator());
      }
      set_initial_position();

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
  /// Is the mini-batch that is finishing corresponds to the second to
  /// last mini-batch, take the proper (possibly reduced) step to
  /// setup for the last mini-batch
  if (m_loaded_mini_batch_idx >= (m_num_iterations_per_epoch-2)) {
    return m_current_pos + m_stride_to_last_mini_batch;
  } else {
    return m_current_pos + m_stride_to_next_mini_batch;
  }
}

void generic_data_reader::select_subset_of_data() {
  if(m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(), get_data_seq_generator());
  }

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
bool generic_data_reader::saveToCheckpointShared(persist& p, const char *name) const {
  // rank 0 writes the training state file
  if (p.get_rank() == 0) {
    char fieldname[1024];

    // Closest to non checkpoint run only saves m_current_pos

    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_current_mini_batch_idx);
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

    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_last_mini_batch", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_stride_to_last_mini_batch);

    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_next_mini_batch", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_stride_to_next_mini_batch);

    snprintf(fieldname, sizeof(fieldname), "%s_base_offset", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_base_offset);

    snprintf(fieldname, sizeof(fieldname), "%s_model_offset", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_model_offset);

    snprintf(fieldname, sizeof(fieldname), "%s_sample_stride", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_sample_stride);

    snprintf(fieldname, sizeof(fieldname), "%s_iteration_stride", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_iteration_stride);

    snprintf(fieldname, sizeof(fieldname), "%s_loaded_mini_batch_idx", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_loaded_mini_batch_idx);

    snprintf(fieldname, sizeof(fieldname), "%s_reset_mini_batch_index", name);
    p.write_uint64(persist_type::train, fieldname, (uint64_t) m_reset_mini_batch_index);
    //printf("%d\n", m_current_mini_batch_idx);
    //printf("%d\n", m_shuffled_indices[0]);
    //printf("%d\n", m_model_offset);
  }
  return true;
}

/** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
bool lbann::generic_data_reader::loadFromCheckpointShared(persist& p, const char *name) {
  // rank 0 reads the training state file
  if (p.get_rank() == 0) {
    char fieldname[1024];

    // Closest to non checkpoint run only loads m_current_pos

    // record minibatch index
    uint64_t val;
    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_current_mini_batch_idx = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    auto size = (int) val;

    // get current position within data
    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_current_pos = (int) val;
    //resize shuffled index array to hold values
    m_shuffled_indices.resize(size);

     //read list of indices
    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.read_int32_contig(persist_type::train, fieldname, &m_shuffled_indices[0], (uint64_t) size);

    /* Everything below is things i have tried loading to see if it was needed. No impact as far as I could tell*/
    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_last_mini_batch", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_stride_to_last_mini_batch = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_stride_to_next_mini_batch", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_stride_to_next_mini_batch = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_base_offset", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_base_offset = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_model_offset", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_model_offset = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_sample_stride", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_sample_stride= (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_iteration_stride", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_iteration_stride= (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_loaded_mini_batch_idx", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_loaded_mini_batch_idx = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_reset_mini_batch_index", name);
    p.read_uint64(persist_type::train, fieldname, &val);
    m_reset_mini_batch_index = (int) val;

    // get size of ShuffleIndices
    //
    //printf("%d\n", m_current_mini_batch_idx);
    //printf("%d\n", m_current_pos);
    //printf("%d\n", m_shuffled_indices[0]);
  }

  // broadcast minibatch index
  MPI_Bcast(&m_current_mini_batch_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&m_stride_to_last_mini_batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_stride_to_next_mini_batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_base_offset, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&m_model_offset, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&m_sample_stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_iteration_stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_loaded_mini_batch_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_reset_mini_batch_index, 1, MPI_INT, 0, MPI_COMM_WORLD);


  // TODO: with multiple readers, make this a scatter
  // broadcast current position
  MPI_Bcast(&m_current_pos, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //printf("%d\n", m_current_pos);
  // broadcast values from rank 0
  int size = m_shuffled_indices.size();
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // resize shuffled index array to hold values
  if (p.get_rank() != 0) {
    m_shuffled_indices.resize(size);
  }

  /// broadcast index array
  MPI_Bcast(&m_shuffled_indices[0], size, MPI_INT, 0, MPI_COMM_WORLD);
  //set_initial_position();
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

void generic_data_reader::setup_data_store(lbann_comm *comm) {

  generic_data_reader *the_reader = dynamic_cast<imagenet_reader*>(this);

  if (the_reader == nullptr && m_master) {
    std::cerr << "WARNING: " << __FILE__ << " " << __LINE__
              << " dynamic_cast<imagenet_reader*> failed; NOT using data_store\n";
    return;
  }

  generic_data_store *store = new data_store_imagenet(comm, this);
  m_data_store = store; 
  store->setup();
}

int lbann::generic_data_reader::fetch_data_indices(std::vector<int> &indicies) {
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: generic data reader load error: !position_valid"
      + " -- current pos = " + std::to_string(m_current_pos)
      + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
  }

  indicies.clear();

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    (El::Int)get_mini_batch_size());

  for (int s = 0; s < mb_size; s++) {
      int n = m_current_pos + (s * m_sample_stride);
      indicies.push_back(n);
  }
  return mb_size;
}

}  // namespace lbann
