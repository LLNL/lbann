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

#ifndef LBANN_DATA_STREAM_HPP
#define LBANN_DATA_STREAM_HPP

#include "lbann/base.hpp"
#include "lbann/utils/random_number_generators.hpp"
#include "lbann/utils/serialize.hpp"

// #include <algorithm>
// #include <cassert>
#include <string>
#include <unistd.h>
// #include <unordered_set>
#include <map>
#include <vector>

namespace lbann {

// Forward declarations
class persist;

class data_stream
{
public:
  using unused_index_map_t = std::map<execution_mode, std::vector<int>>;

  data_stream(bool shuffle = true)
    : m_mini_batch_size(0),
      m_current_pos(0),
      m_stride_to_next_mini_batch(0),
      m_base_offset(0),
      m_model_offset(0),
      m_sample_stride(1),
      m_iteration_stride(1),
      m_last_mini_batch_size(0),
      m_stride_to_last_mini_batch(0),
      m_reset_mini_batch_index(0),
      m_loaded_mini_batch_idx(0),
      m_current_mini_batch_idx(0),
      m_num_iterations_per_epoch(0),
      m_global_mini_batch_size(0),
      m_global_last_mini_batch_size(0),
      m_world_master_mini_batch_adjustment(0),
      m_num_parallel_readers(0),
      m_shuffle(shuffle),
      m_absolute_sample_count(0),
      m_use_percent(1.0)
  {
  }

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /**
   * Prepare to start processing an epoch of data.
   * If shuffle is true, then shuffle the indices of the data set
   * If the base offset is not specified set it to 0
   * If the stride is not specified set it to batch size
   */
  virtual void setup(int num_io_threads);

  /**
   * If set to false, indices (data samples) are not shuffled
   * default (in ctor) is true.
   */
  void set_shuffle(bool b) { m_shuffle = b; }

  /**
   * Returns true if data samples are shuffled.
   */
  bool is_shuffled() const { return m_shuffle; }

  /**
   * Set shuffled indices; primary use is for testing
   * and reproducibility
   */
  void set_shuffled_indices(const std::vector<int>& indices)
  {
    m_shuffled_indices = indices;
  }

  /**
   * Returns the shuffled indices; primary use is for testing.
   */
  const std::vector<int>& get_shuffled_indices() const
  {
    return m_shuffled_indices;
  }

  /**
   * During the network's update phase, the data reader will
   * advanced the current position pointer.  If the pointer wraps
   * around, then reshuffle the data indicies.
   */
  virtual bool update(bool is_active_reader);

  /**
   * Read the first 'n' samples. If nonzero, this over-rides
   * set_absolute_sample_count, set_use_percent. The intent
   * is to use this for testing. A problem with set_absolute_sample_count
   * and set_use_percent is that the entire data set is read in, then
   * a subset is selected
   */
  void set_first_n(int n);

  /**
   * Sets the absolute number of data samples that will be used for training or
   * testing.
   */
  void set_absolute_sample_count(size_t s);

  /**
   * Set the percentage of the data set to use for training and validation or
   * testing.
   * @param s The percentage used, in the range [0, 1].
   */
  void set_use_percent(double s);

  /**
   * Sets the percentage of the dataset to be used for validation.
   * @param m The execution mode.
   * @param s The percentage used, in the range [0, 1].
   */
  virtual void set_execution_mode_split_percent(execution_mode m, double s);

  /**
   * Set an idenifier for the dataset.
   * The role should be one of "train", "test", or "validate".
   */
  virtual void set_role(std::string role);

  /**
   * Get the role for this dataset.
   */
  std::string get_role() const { return m_role; }

  /// True if the data reader's current position is valid.
  virtual bool position_valid() const
  {
    return (m_current_pos < get_num_data());
  }
  /// True if the data reader's current position is not valid but within # ranks
  /// per model of the end of the data set (e.g. it is a rank with no valid data
  /// on the last iteration)
  virtual bool position_is_overrun(num_procs_per_trainer) const
  {
    int end_pos = (int)m_shuffled_indices.size();
    return (m_current_pos >= end_pos &&
            (m_current_pos - end_pos) < num_procs_per_trainer /*m_comm->get_procs_per_trainer()*/);
  }
  /// True if the data reader is at the start of an epoch.
  bool at_new_epoch() const
  {
    /// Note that data readers can start at a non-zero index if there
    /// are parallel data readers in a model
    return ((m_loaded_mini_batch_idx == m_reset_mini_batch_index) &&
            (m_current_mini_batch_idx == 0));
  }
  /// Set the mini batch size
  void set_mini_batch_size(const int s);
  /// Get the mini batch size
  int get_mini_batch_size() const { return m_mini_batch_size; }
  /// Get the loaded mini-batch size
  int get_loaded_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_mini_batch_size() const;
  /// Get the current global mini-batch size.
  int get_current_global_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_world_master_mini_batch_adjustment(int model_rank) const;
  /// Return the full mini_batch_size.
  int get_mini_batch_max() const { return m_mini_batch_size; }
  /// Set the mini batch size across all models (global)
  void set_global_mini_batch_size(const int s) { m_global_mini_batch_size = s; }
  /// Return the mini_batch_size across all models (global)
  int get_global_mini_batch_size() const { return m_global_mini_batch_size; }
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
  /// Set the iteration stride
  void set_iteration_stride(const int s) { m_iteration_stride = s; }
  /// Return the iteration stride.
  int get_iteration_stride() const { return m_iteration_stride; }
  /// Return the base offset.
  virtual void set_base_offset(const int s) { m_base_offset = s; }
  /// Return the base offset.
  int get_base_offset() const { return m_base_offset; }
  /// Set the model offset
  void set_model_offset(const int s) { m_model_offset = s; }
  /// Return the model offset.
  int get_model_offset() const { return m_model_offset; }
  /// Set the last mini batch size
  void set_last_mini_batch_size(const int s) { m_last_mini_batch_size = s; }
  /// Return the last mini batch size
  int get_last_mini_batch_size() const { return m_last_mini_batch_size; }
  /// Set the last mini batch size across all models (global)
  void set_global_last_mini_batch_size(const int s)
  {
    m_global_last_mini_batch_size = s;
  }
  /// Return the last mini batch size across all models (global)
  int get_global_last_mini_batch_size() const
  {
    return m_global_last_mini_batch_size;
  }
  /// Set the world master mini batch adjustment (global)
  void set_world_master_mini_batch_adjustment(const int s)
  {
    m_world_master_mini_batch_adjustment = s;
  }
  /// Return the world master mini batch adjustment (global)
  int get_world_master_mini_batch_adjustment() const
  {
    return m_world_master_mini_batch_adjustment;
  }
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
  /// Set the number of parallel readers per model
  void set_num_parallel_readers(const int s) { m_num_parallel_readers = s; }
  /// Return the number of parallel readers per model
  int get_num_parallel_readers() const { return m_num_parallel_readers; }
  /// Set the starting mini-batch index for the epoch
  virtual void set_reset_mini_batch_index(const int s)
  {
    m_reset_mini_batch_index = s;
  }
  /// Return the starting mini-batch index for the epoch
  int get_reset_mini_batch_index() const { return m_reset_mini_batch_index; }
  /// Return the current mini-batch index for the epoch
  int get_loaded_mini_batch_index() const { return m_loaded_mini_batch_idx; }
  /// Return the current mini-batch index for the epoch
  int get_current_mini_batch_index() const { return m_current_mini_batch_idx; }
  /// Set the current position based on the base and model offsets
  void set_initial_position()
  {
    m_current_pos = m_base_offset + m_model_offset;
    m_loaded_mini_batch_idx = m_reset_mini_batch_index;
    m_current_mini_batch_idx = 0;
  }
  /// Get the current position in the data reader.
  int get_position() const { return m_current_pos; }
  /// Get the next position in the data reader.
  int get_next_position() const;
  /// Get a pointer to the start of the shuffled indices.
  int* get_indices() { return &m_shuffled_indices[0]; }
  /// Get the number of samples in this dataset.
  virtual int get_num_data() const { return (int)m_shuffled_indices.size(); }
  /// Get the number of unused samples in this dataset.
  int get_num_unused_data(execution_mode m) const;

  /// Get a pointer to the start of the unused sample indices.
  int* get_unused_data(execution_mode m);

  const std::vector<int>& get_unused_indices(execution_mode m);

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

  /**
   * Optionally resizes the shuffled indices based on the data reader
   * prototext settings: absolute_sample_count, percent_of_data_to_use.
   * (dah - this was formerly part of select_subset_of_data)
   */
  void resize_shuffled_indices();

  /**
   * Select the appropriate subset of data for the additional
   * execution modes such as validation or tournament  set based on
   * the data reader prototext setting: validation_percent or
   * tournament_percent
   */
  void select_subset_of_data();

  /**
   * Replaced the shuffled index set with the unused index set, empying the
   * unused set.
   */
  virtual void use_unused_index_set(execution_mode m);

  /** \brief Given directory to store checkpoint files, write state to file and
   * add to number of bytes written */
  bool save_to_checkpoint_shared(persist& p, execution_mode mode);

  /** \brief Given directory to store checkpoint files, read state from file and
   * add to number of bytes read */
  bool load_from_checkpoint_shared(persist& p, execution_mode mode);

  bool save_to_checkpoint_distributed(persist& p, execution_mode mode);

  /** \brief Given directory to store checkpoint files, read state from file and
   * add to number of bytes read */
  bool load_from_checkpoint_distributed(persist& p, execution_mode mode);

  /// Shuffle indices (uses the data_seq_generator)
  virtual void shuffle_indices();
  /// Shuffle indices and profide a random number generator
  virtual void shuffle_indices(rng_gen& gen);

public:
  int m_mini_batch_size;
  int m_current_pos;
  /// Batch Stride is typically batch_size, but may be a multiple of batch size
  /// if there are multiple readers
  int m_stride_to_next_mini_batch;
  /// If there are multiple instances of the reader,
  /// then it may not reset to zero
  int m_base_offset;
  /// If there are multiple models with multiple instances of the reader,
  /// each model's set of readers may not reset to zero
  /// Provide a set of size, strides, and thresholds to handle the last mini
  /// batch of a dataset
  int m_model_offset;
  /// Sample stride is used when a mini-batch is finely interleaved across a
  /// DATA_PARALELL distribution.
  int m_sample_stride;
  /// Stride used by parallel data readers within the model
  int m_iteration_stride;

  std::string m_role;

  std::vector<int> m_shuffled_indices;
  /// Record of the indicies that are not being used for training
  unused_index_map_t m_unused_indices;

  int m_last_mini_batch_size;
  int m_stride_to_last_mini_batch;
  /// The index at which this data reader starts its epoch
  int m_reset_mini_batch_index;
  /// The index of the current mini-batch that has been loaded
  int m_loaded_mini_batch_idx;
  /// The index of the current mini-batch that is being processed
  /// (train/test/validate)
  int m_current_mini_batch_idx;
  int
    m_num_iterations_per_epoch; /// How many iterations all readers will execute

  int m_global_mini_batch_size;
  int m_global_last_mini_batch_size;
  int m_world_master_mini_batch_adjustment;

  int m_num_parallel_readers; /// How many parallel readers are being used

  bool m_shuffle;
  size_t m_absolute_sample_count;
  std::map<execution_mode, double> m_execution_mode_split_percentage;
  double m_use_percent;
  int m_first_n;
};

} // namespace lbann

#endif // LBANN_DATA_STREAM_HPP
