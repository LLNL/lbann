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

#ifndef LBANN_DATASET_HPP_INCLUDED
#define LBANN_DATASET_HPP_INCLUDED

#include "lbann/utils/exception.hpp"
#include <string>

namespace lbann {

class dataset
{
public:
  dataset()
    : m_num_samples_processed(0),
      m_total_samples(0),
      m_mini_batch_size(0),
      m_current_pos(0),
      m_stride_to_next_mini_batch(0),
      m_base_offset(0),
      m_sample_stride(1),
      m_last_mini_batch_size(0),
      m_stride_to_last_mini_batch(0),
      m_current_mini_batch_idx(0),
      m_num_iterations_per_epoch(0),
      m_initialized(false){};

  // The associated model/IO layer using this dataset is responsible for copying
  // the data reader.
  dataset(const dataset& other) = default;
  dataset& operator=(const dataset& other) = default;
  template <class Archive>
  void serialize(Archive& ar);

  void setup(long total_samples, std::string role);
  void print_config();

  /**
   * Set an idenifier for the dataset.
   * The role should be one of "train", "test", or "validate".
   */
  void set_role(std::string role);

  /**
   * Get the role for this dataset.
   */
  std::string get_role() const { return m_role; }

  long get_num_samples_processed() const { return m_num_samples_processed; }
  long& num_samples_processed() { return m_num_samples_processed; }
  long get_total_samples() const { return m_total_samples; }
  long& total_samples() { return m_total_samples; }

  /// Get the number of samples in this dataset.
  long get_num_data() const { return m_total_samples; }

  /// True if the data reader's current position is valid.
  bool position_valid() const { return (m_current_pos < get_num_data()); }
  /// True if the data reader's current position is not valid but within # ranks
  /// per model of the end of the data set (e.g. it is a rank with no valid data
  /// on the last iteration)
  bool position_is_overrun() const
  {
    int end_pos = get_num_data();
    return (m_current_pos >= end_pos &&
            (m_current_pos - end_pos) <
              m_sample_stride /*m_comm->get_procs_per_trainer()*/);
  }
  /// True if the data reader is at the start of an epoch.
  bool at_new_epoch() const { return (m_current_mini_batch_idx == 0); }
  /// Set the mini batch size
  void set_mini_batch_size(const int s);
  /// Get the mini batch size
  int get_mini_batch_size() const { return m_mini_batch_size; }
  /// Get the size of the next mini-batch that will be loaded by an
  /// asynchronous, background, I/O thread (one fetch in the future)
  int get_next_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_mini_batch_size() const;
  /// Return the full mini_batch_size.
  int get_mini_batch_max() const { return m_mini_batch_size; }
  /// Set the mini batch stride
  void set_stride_to_next_mini_batch(const int s)
  {
    m_stride_to_next_mini_batch = s;
  }
  /// Return the mini batch stride.
  int get_stride_to_next_mini_batch() const
  {
    return m_stride_to_next_mini_batch;
  }
  /// Set the sample stride
  void set_sample_stride(const int s) { m_sample_stride = s; }
  /// Return the sample stride.
  int get_sample_stride() const { return m_sample_stride; }
  /// Return the base offset.
  void set_base_offset(const int s) { m_base_offset = s; }
  /// Return the base offset.
  int get_base_offset() const { return m_base_offset; }
  /// Set the last mini batch size
  void set_last_mini_batch_size(const int s) { m_last_mini_batch_size = s; }
  /// Return the last mini batch size
  int get_last_mini_batch_size() const { return m_last_mini_batch_size; }
  /// Set the last mini batch stride
  void set_stride_to_last_mini_batch(const int s)
  {
    m_stride_to_last_mini_batch = s;
  }
  /// Return the last mini batch stride
  int get_stride_to_last_mini_batch() const
  {
    return m_stride_to_last_mini_batch;
  }
  /// Return the current mini-batch index for the epoch
  int get_current_mini_batch_index() const { return m_current_mini_batch_idx; }
  /// Set the current position based on the base and model offsets
  void set_initial_position()
  {
    m_current_pos = m_base_offset;
    m_current_mini_batch_idx = 0;
  }
  /// Get the current position in the data reader.
  int get_position() const { return m_current_pos; }
  /// Get the next position in the data reader.
  int get_next_position() const;

  /// Set the number of iterations in each epoch.
  void set_num_iterations_per_epoch(int num_iterations_per_epoch)
  {
    m_num_iterations_per_epoch =
      num_iterations_per_epoch; /// @todo BVE FIXME merge this with alternate
                                /// approach
  }
  /// Get the number of iterations in each epoch.
  int get_num_iterations_per_epoch() const
  {
    return m_num_iterations_per_epoch; /// @todo BVE FIXME merge this with
                                       /// alternate approach
  }

  /// Return the index of the current iteration step in the epoch (also the
  /// mini-batch index)
  int get_current_step_in_epoch() const { return m_current_mini_batch_idx; }

  bool initialized() const { return m_initialized; }

  /**
   * During the network's update phase, the dataset will
   * advanced the current position pointer and mini-batch index.
   * If the pointer wraps around, then epoch is complete.
   */
  bool update();

protected:
  long m_num_samples_processed;
  long m_total_samples;

  int m_mini_batch_size;
  int m_current_pos;
  /// Batch Stride is typically batch_size, but may be a multiple of batch size
  /// if there are multiple readers
  int m_stride_to_next_mini_batch;
  /// If there are multiple instances of the reader,
  /// then it may not reset to zero
  int m_base_offset;
  /// Sample stride is used when a mini-batch is finely interleaved across a
  /// DATA_PARALLEL distribution.
  int m_sample_stride;

  int m_last_mini_batch_size;
  int m_stride_to_last_mini_batch;
  /// The index of the current mini-batch that is being processed
  /// (train/test/validate)
  int m_current_mini_batch_idx;
  int
    m_num_iterations_per_epoch; /// How many iterations all readers will execute

  bool m_initialized;
  std::string m_role;
};

} // namespace lbann

#endif // LBANN_DATASET_HPP_INCLUDED
