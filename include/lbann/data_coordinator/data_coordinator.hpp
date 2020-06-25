////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/dataset.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/data_readers/data_reader_hdf5.hpp"
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

class data_coordinator {
 public:
  using data_reader_map_t = std::map<execution_mode, generic_data_reader *>;
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
  data_coordinator(lbann_comm *comm, std::map<execution_mode, generic_data_reader *> data_readers) :
    m_trainer(nullptr),
    m_comm(comm),
    m_data_set_processed(false),
    m_execution_context(nullptr) {}

  virtual ~data_coordinator() {
    // Synchronize the I/O thread pool
    // Note: The thread pool may still be running asynchronously if the
    // trainer is destroyed in the middle of an epoch. The thread pool
    // needs to interact with data readers, etc., so it needs to be
    // synchronized before any of them are destroyed.
    if (m_io_thread_pool != nullptr) {
      m_io_thread_pool->reap_threads();
    }
    // Data coordinator always frees data readers.
    for (auto& dr : m_data_readers) {
      delete dr.second;
    }
  }

  // Data Coordinators copy their data readers.
  data_coordinator(const data_coordinator& other)
    : m_comm(other.m_comm),
      m_training_dataset(other.m_training_dataset),
      m_testing_dataset(other.m_testing_dataset),
      m_validation_dataset(other.m_validation_dataset),
      m_data_readers(other.m_data_readers),
      m_data_set_processed(other.m_data_set_processed),
      m_execution_context(other.m_execution_context) {
    for (auto& dr : m_data_readers) {
      dr.second = dr.second ? dr.second->copy() : nullptr;
    }
  }

  data_coordinator& operator=(const data_coordinator& other) {
    for (auto& dr : m_data_readers) {
      dr.second = dr.second ? dr.second->copy() : nullptr;
    }
    return *this;
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize( Archive & ar ) {
    ar(/*CEREAL_NVP(m_io_buffer),*/
       CEREAL_NVP(m_training_dataset),
       CEREAL_NVP(m_testing_dataset),
       CEREAL_NVP(m_validation_dataset)/*,
       CEREAL_NVP(m_data_readers),
       CEREAL_NVP(m_data_set_processed)*/);
  }

  virtual void setup(thread_pool& io_thread_pool, int max_mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers);

  void set_trainer(trainer &trainer) { m_trainer = &trainer; }

  /** Check to see if there is a valid training context for the data coordinator */
  bool has_valid_execution_context() const {
    return (m_execution_context != nullptr);
  }

  /** Grab the training context of the data coordinator */
  const execution_context& get_execution_context() const {
    if(m_execution_context == nullptr) {
      LBANN_ERROR("execution context is not set");
    }
    return *m_execution_context;
  }

  /** Grab the training context of the data coordinator */
  execution_context& get_execution_context() {
    return const_cast<execution_context&>(static_cast<const data_coordinator&>(*this).get_execution_context());
  }

  /** Return the I/O thread pool */
  thread_pool& get_io_thread_pool() const {
    if (!m_io_thread_pool) { LBANN_ERROR("m_io_thread_pool is null"); }
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
  virtual El::Matrix<El::Int>* get_sample_indices_per_mb(execution_mode mode) = 0;

  virtual int get_num_iterations_per_epoch(execution_mode mode) const;

  virtual int get_current_step_in_epoch(execution_mode mode) const;

  virtual int get_mini_batch_size(execution_mode mode) const;

  virtual int get_last_mini_batch_size(execution_mode mode) const;

  virtual int get_current_mini_batch_size(execution_mode mode) const;

  virtual int get_global_mini_batch_size(execution_mode mode) const;

  virtual int get_current_global_mini_batch_size(execution_mode mode) const;

  virtual int get_global_last_mini_batch_size(execution_mode mode) const;

  virtual int get_world_master_mini_batch_adjustment(execution_mode mode) const;

  virtual int get_current_world_master_mini_batch_adjustment(execution_mode mode, int model_rank) const;

  //************************************************************************
  // Helper functions to access the data readers
  //************************************************************************

  generic_data_reader *get_data_reader(const execution_mode mode) const {
    generic_data_reader *data_reader = nullptr;

    auto it = m_data_readers.find(mode);
    if (it != m_data_readers.end()) data_reader = it->second;

    switch(mode) {
    case execution_mode::training:
      break;
    case execution_mode::validation:
      break;
    case execution_mode::testing:
      break;
    default:
      LBANN_ERROR("generic data distribution: invalid execution phase");
    }
    return data_reader;
  }

  /**
   * Get the dimensions of the underlying data.
   */
  TargetModeDimMap get_data_dims() {
    TargetModeDimMap map;
    generic_data_reader *dr;
    for(execution_mode mode : execution_mode_iterator()) {
      dr = get_data_reader(mode);
      if (dr != nullptr) {
        map[data_reader_target_mode::INPUT] = dr->get_data_dims();
        map[data_reader_target_mode::CLASSIFICATION] = std::vector<int>(1, dr->get_num_labels());
        map[data_reader_target_mode::REGRESSION] = std::vector<int>(1, dr->get_num_responses());
        map[data_reader_target_mode::RECONSTRUCTION] = dr->get_data_dims();
        map[data_reader_target_mode::LABEL_RECONSTRUCTION] = dr->get_data_dims();
        map[data_reader_target_mode::NA] = std::vector<int>(1, 0);
        return map;
      }
    }
    LBANN_ERROR("get_data_dims: no available data readers");
    return {};
  }

  /**
   * Get the dimensions of the underlying data.
   */
  SPModeSlicePoints get_slice_points() {
    SPModeSlicePoints map;
    generic_data_reader *dr;
    for(execution_mode mode : execution_mode_iterator()) {
      dr = get_data_reader(mode);
      if (dr != nullptr) {
        for(slice_points_mode sp_mode : slice_points_mode_iterator()) {
          bool is_supported;
          std::vector<El::Int> tmp = dr->get_slice_points(sp_mode, is_supported);
          if(is_supported) {
            map[sp_mode] = tmp;
          }
        }
        return map;
      }
    }
    LBANN_ERROR("get_data_dims: no available data readers");
    return {};
  }

  DataReaderMetaData get_dr_metadata() {
    DataReaderMetaData drm;
    drm.data_dims = get_data_dims();
    drm.slice_points = get_slice_points();
#ifdef LBANN_HAS_DISTCONV
    const auto training_dr = m_data_readers[execution_mode::training];
    drm.shuffle_required = training_dr->is_tensor_shuffle_required();
#endif // LBANN_HAS_DISTCONV
    return drm;
  }

  /**
   * Get the linearized size of the underlying data.
   */
  long get_linearized_data_size() const {
    long linearized_data_size = -1;

    generic_data_reader *dr;
    dr = get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      linearized_data_size = dr->get_linearized_data_size();
    }

    dr = get_data_reader(execution_mode::validation);
    if (dr != nullptr) {
      long tmp_data_size = dr->get_linearized_data_size();
      if (linearized_data_size != -1 && linearized_data_size != tmp_data_size) {
        LBANN_ERROR("lbann_io_layer: validation data set size does not "
                              "match the currently established data set size");
      }
    }

    dr = get_data_reader(execution_mode::testing);
    if (dr != nullptr) {
      long tmp_data_size = dr->get_linearized_data_size();
      if (linearized_data_size != -1 && linearized_data_size != tmp_data_size) {
        LBANN_ERROR("lbann_io_layer: testing data set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_data_size;
  }

  /**
   * Get the linearized size of the labels for the underlying data.
   */
  long get_linearized_label_size() const {
    // if (this->is_for_regression()) {
    //   return static_cast<long>(1);
    // }
    long linearized_label_size = -1;
    generic_data_reader *dr;
    dr = get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      linearized_label_size = dr->get_linearized_label_size();
    }
    dr = get_data_reader(execution_mode::validation);
    if (dr != nullptr) {
      long tmp_label_size = dr->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        LBANN_ERROR("lbann_io_layer: validation label set size (" + std::to_string(tmp_label_size) + ") does not match the currently established data set size (" + std::to_string(linearized_label_size) + ")");
      }
    }
    dr = get_data_reader(execution_mode::testing);
    if (dr != nullptr) {
      long tmp_label_size = dr->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        LBANN_ERROR("lbann_io_layer: testing label set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_label_size;
  }

  long get_linearized_response_size() const {
    // if (!this->is_for_regression()) {
    //   return static_cast<long>(1);
    // }
    long linearized_response_size = -1;
    generic_data_reader *dr;
    dr = get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      linearized_response_size = dr->get_linearized_response_size();
    }
    dr = get_data_reader(execution_mode::validation);
    if (dr != nullptr) {
      long tmp_response_size = dr->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        LBANN_ERROR("lbann_io_layer: validation response set size does not "
                              "match the currently established data set size");
      }
    }
    dr = get_data_reader(execution_mode::testing);
    if (dr != nullptr) {
      long tmp_response_size = dr->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        LBANN_ERROR("lbann_io_layer: testing response set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_response_size;
  }

  // At the start of the epoch, set the execution mode and make sure
  // that each layer points to this model
  void reset_mode(execution_context& context) {
    m_execution_context = static_cast<observer_ptr<execution_context>>(&context);
  }

  /** @name Helper functions to access the dataset statistics */
///@{
   /** @brief Return the dataset for the given execution mode. */
  dataset& get_dataset(execution_mode m) {
    switch(m) {
    case execution_mode::training:
      return m_training_dataset;
      break;
    case execution_mode::validation:
      return m_validation_dataset;
      break;
    case execution_mode::testing:
      return m_testing_dataset;
      break;
    default:
      LBANN_ERROR("get_dataset: invalid execution mode");
    }
  }

  const dataset& get_dataset(execution_mode m) const {
    switch(m) {
    case execution_mode::training:
      return m_training_dataset;
      break;
    case execution_mode::validation:
      return m_validation_dataset;
      break;
    case execution_mode::testing:
      return m_testing_dataset;
      break;
    default:
       LBANN_ERROR("get_dataset: invalid execution mode");
    }
  }

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  dataset* select_first_valid_dataset() {
    if (m_data_readers[execution_mode::training]) {
      return &m_training_dataset;
    } else if (m_data_readers[execution_mode::validation]) {
      return &m_validation_dataset;
    } else if (m_data_readers[execution_mode::testing]) {
      return &m_testing_dataset;
    } else {
      return nullptr;
    }
  }

  long get_num_samples_trained() const {
    return m_training_dataset.get_num_samples_processed();
  }
  long get_num_samples_tested() const {
    return m_testing_dataset.get_num_samples_processed();
  }
  long get_total_num_training_samples() const {
    return m_training_dataset.get_total_samples();
  }
  long get_total_num_testing_samples() const {
    return m_testing_dataset.get_total_samples();
  }

  /**
   * Update the number of samples processed for the current execution mode.
   */
  long update_num_samples_processed(execution_mode mode, long num_samples) {
    dataset& ds = get_dataset(mode);
    ds.num_samples_processed() += num_samples;
    return ds.get_num_samples_processed();
  }

  /** @brief Check if the execution mode is valid (i.e. has data). */
  bool is_execution_mode_valid(execution_mode mode) const {
    const dataset& ds = get_dataset(mode);
    return (ds.get_total_samples() != static_cast<long>(0));
  }
///@}

  //************************************************************************
  //
  //************************************************************************

  void calculate_num_iterations_per_epoch(int max_mini_batch_size, generic_data_reader *data_reader);
  void calculate_num_iterations_per_epoch(int mini_batch_size);

  int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const;
  static int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers, const lbann_comm* comm);

  virtual int get_num_parallel_readers(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_num_parallel_readers() : 0;
  }

  bool at_new_epoch(execution_mode mode) const {
    const generic_data_reader *dr = get_data_reader(mode);
    return (dr != nullptr && dr->at_new_epoch());
  }

  bool at_new_epoch() const {
    return at_new_epoch(execution_mode::training);
  }

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
  trainer *m_trainer;
  /** Pointer to LBANN communicator. */
  lbann_comm *m_comm;

  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;

  data_reader_map_t m_data_readers;
 //  std::map<execution_mode, dataset_stats> m_dataset_stats;

public:  // @todo BVE FIXME
  bool m_data_set_processed;
  std::mutex dr_mutex;

  /** Pointer to the execution context object used for training or evaluating this model */
  observer_ptr<execution_context> m_execution_context;

  observer_ptr<thread_pool> m_io_thread_pool;
};

} // namespace lbann

#endif // LBANN_DATA_COORDINATOR_HPP
