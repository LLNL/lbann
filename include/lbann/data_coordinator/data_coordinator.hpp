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

#ifndef LBANN_DATA_COORDINATOR_HPP
#define LBANN_DATA_COORDINATOR_HPP

#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"
#include "lbann/utils/dataset.hpp"
#include "lbann/utils/threads/thread_pool.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/data_readers/data_reader_hdf5_legacy.hpp"
#endif // LBANN_HAS_DISTCONV

/** Design docs:
 num_parallel_readers - used by the partitioned io buffer to control
 how many ranks will access data.  Can be set by either the user, or
 by the size of the mini-batch????

 * IO buffers should go away and be rolled into the data coordinator.

 * Buffered data coordinator knows about the native data size / for
 * the data reader and how to store it

 * input layer should take responsibility for the "distribute from
 * local matrix code.  That should be the copy out of the data
 * coordinator into the input layer.

 * num children layers for the IO buffers is a property of the data
 * reader
 * there should be one input layer for each type of data to read.
 */
namespace lbann {

// Forward declaration
class ExecutionContext;
class generic_data_reader;
class persist;
class trainer;

class data_coordinator
{
public:
  using dataset_map_t = std::map<execution_mode, dataset>;
  using data_reader_map_t = std::map<execution_mode, generic_data_reader*>;
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

public:
  data_coordinator(lbann_comm* comm)
    : m_trainer(nullptr),
      m_comm(comm),
      m_data_set_processed(false),
      m_execution_context(nullptr),
      m_io_thread_pool(nullptr)
  {}

  virtual ~data_coordinator();

  // Data Coordinators copy their data readers.
  data_coordinator(const data_coordinator& other);

  data_coordinator& operator=(const data_coordinator& other);

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** Setup the thread pool and data readers within the data coordinator */
  virtual void
  setup(thread_pool& io_thread_pool,
        int max_mini_batch_size,
        std::map<execution_mode, generic_data_reader*> data_readers);

  /** Once all of the models that are served by this data coordinator are
   *  setup and have registered which data fields are required, setup the local
   *  buffers in the data coordinator for each data field.
   */
  virtual void setup_data_fields(int max_mini_batch_size) = 0;

  void set_trainer(trainer& trainer) { m_trainer = &trainer; }

  /** Check to see if there is a valid training context for the data coordinator
   */
  bool has_valid_execution_context() const
  {
    return (m_execution_context != nullptr);
  }

  /** Grab the training context of the data coordinator */
  const ExecutionContext& get_execution_context() const
  {
    if (m_execution_context == nullptr) {
      LBANN_ERROR("execution context is not set");
    }
    return *m_execution_context;
  }

  /** Grab the training context of the data coordinator */
  ExecutionContext& get_execution_context()
  {
    return const_cast<ExecutionContext&>(
      static_cast<const data_coordinator&>(*this).get_execution_context());
  }

  /** Return the I/O thread pool */
  thread_pool& get_io_thread_pool() const
  {
    if (!m_io_thread_pool) {
      LBANN_ERROR("m_io_thread_pool is null");
    }
    return *m_io_thread_pool;
  }

  virtual void fetch_data(execution_mode mode) = 0;

  /** @brief Complete any background I/O data fetch for the execution
      mode requested */
  virtual void collect_background_data_fetch(execution_mode mode) = 0;

  /// @todo BVE FIXME this should probably be a property of the
  /// execution mode
  virtual bool epoch_complete(execution_mode mode) = 0;

  //************************************************************************
  // Helper functions to access the statistics about the data set
  //************************************************************************

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
  virtual const El::Matrix<El::Int>*
  get_sample_indices_per_mb(execution_mode mode) const = 0;
  virtual El::Matrix<El::Int>*
  get_sample_indices_per_mb(execution_mode mode) = 0;

  virtual size_t get_num_iterations_per_epoch(execution_mode mode) const;

  virtual int get_current_step_in_epoch(execution_mode mode) const;

  virtual int get_mini_batch_size(execution_mode mode) const;

  virtual int get_last_mini_batch_size(execution_mode mode) const;

  virtual int get_current_mini_batch_size(execution_mode mode) const;

  virtual int get_global_mini_batch_size(execution_mode mode) const;

  virtual int get_current_global_mini_batch_size(execution_mode mode) const;

  virtual int get_global_last_mini_batch_size(execution_mode mode) const;

  virtual int get_world_master_mini_batch_adjustment(execution_mode mode) const;

  virtual int
  get_current_world_master_mini_batch_adjustment(execution_mode mode,
                                                 int model_rank) const;

  //************************************************************************
  // Helper functions to access the data readers
  //************************************************************************

  generic_data_reader* get_data_reader(const execution_mode mode) const;

  /**
   * Get the dimensions of the underlying data.
   */
  TargetModeDimMap get_data_dims();

  /**
   * Get the dimensions of the underlying data.
   */
  SPModeSlicePoints get_slice_points();

  DataReaderMetaData get_dr_metadata();

  /**
   * Check to see if the data readers have labels
   */
  bool has_labels();

  /**
   * Check to see if the data readers have responses
   */
  bool has_responses();

  /**
   * Get the linearized size of the underlying data.
   */
  long get_linearized_size(data_field_type const& data_field) const;

  /**
   * Get the linearized size of the underlying data.
   */
  long get_linearized_data_size() const;

  /**
   * Get the linearized size of the labels for the underlying data.
   */
  long get_linearized_label_size() const;

  /**
   * Get the linearized size of the responses for the underlying data.
   */
  long get_linearized_response_size() const;

  // At the start of the epoch, set the execution mode and make sure
  // that each layer points to this model
  void reset_mode(ExecutionContext& context);

  /** @name Helper functions to access the dataset statistics */
  ///@{
  /** @brief Return the dataset for the given execution mode. */
  dataset& get_dataset(execution_mode m);

  const dataset& get_dataset(execution_mode m) const;

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  dataset* select_first_valid_dataset();

  long get_num_samples(execution_mode m) const;

  long get_total_num_samples(execution_mode m) const;

  /**
   * Update the number of samples processed for the current execution mode.
   */
  long update_num_samples_processed(execution_mode mode, long num_samples);

  /** @brief Check if the execution mode is valid (i.e. has data). */
  bool is_execution_mode_valid(execution_mode mode) const;
  ///@}

  //************************************************************************
  //
  //************************************************************************

  void calculate_num_iterations_per_epoch(int max_mini_batch_size,
                                          generic_data_reader* data_reader);
  void calculate_num_iterations_per_epoch(int mini_batch_size);

  int compute_max_num_parallel_readers(
    long data_set_size,
    int mini_batch_size,
    int requested_num_parallel_readers) const;
  static int
  compute_max_num_parallel_readers(long data_set_size,
                                   int mini_batch_size,
                                   int requested_num_parallel_readers,
                                   const lbann_comm* comm);

  virtual int get_num_parallel_readers(execution_mode mode) const;

  bool at_new_epoch(execution_mode mode) const;

  bool at_new_epoch() const;

  virtual void
  register_active_data_field(data_field_type const& data_field,
                             std::vector<El::Int> const& data_field_dim_map);

  //************************************************************************
  //
  //************************************************************************

  // save state of IO to a checkpoint
  virtual bool save_to_checkpoint_shared(persist& p) const;
  // reload state of IO from a checkpoint
  virtual bool load_from_checkpoint_shared(persist& p);
  virtual bool save_to_checkpoint_distributed(persist& p) const;
  virtual bool load_from_checkpoint_distributed(persist& p);

protected:
  /** Pointer to hosting trainer */
  trainer* m_trainer;
  /** Pointer to LBANN communicator. */
  lbann_comm* m_comm;

  /// Datasets hold the active statistics and metadata for each data reader
  dataset_map_t m_datasets;

  data_reader_map_t m_data_readers;
  //  std::map<execution_mode, dataset_stats> m_dataset_stats;

  data_field_dim_map_type m_active_data_fields_dim_map;

  std::set<data_field_type> m_active_data_fields;

public: // @todo BVE FIXME
  bool m_data_set_processed;
  std::mutex dr_mutex;

  /** Pointer to the execution context object used for training or evaluating
   * this model */
  observer_ptr<ExecutionContext> m_execution_context;

  observer_ptr<thread_pool> m_io_thread_pool;
};

} // namespace lbann

#endif // LBANN_DATA_COORDINATOR_HPP
