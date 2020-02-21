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

#ifndef LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/io_layer.hpp"
//#include "lbann/utils/dataset.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/imcomm.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include <cereal/types/utility.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

#include "lbann/utils/profiling.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

#include <future>

namespace lbann {

/** @todo Move functionality to input_layer. */
class generic_input_layer : public io_layer {
 public:
  using data_reader_map_t = std::map<execution_mode, generic_data_reader *>;
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
  generic_input_layer(lbann_comm *comm,
              int num_parallel_readers,
              std::map<execution_mode, generic_data_reader *> data_readers,
              bool data_set_spans_models = true,
              data_reader_target_mode dr_mode = data_reader_target_mode::CLASSIFICATION)
    : io_layer(comm, data_set_spans_models, dr_mode),
      m_io_buffers(),
      m_training_dataset(),
      m_testing_dataset(),
      m_validation_dataset(),
      m_data_readers(data_readers),
      m_data_set_processed(false) {
      //m_data_sets_span_models(data_sets_span_models) {
    // Input layers have no parents
    m_expected_num_parent_layers = 0;
    if(dr_mode == data_reader_target_mode::NA) {
      m_expected_num_child_layers = 1;
    }else {
      // Input layers output a sample and target, which could be the
      // original value, categorical label, or regression value
      m_expected_num_child_layers = 2;
    }

    if(m_data_readers[execution_mode::training] != nullptr) {
      m_training_dataset.total_samples() = m_data_readers[execution_mode::training]->get_num_data();
    }

    if(m_data_readers[execution_mode::validation] != nullptr) {
      m_validation_dataset.total_samples() = m_data_readers[execution_mode::validation]->get_num_data();
    }

    if(m_data_readers[execution_mode::testing] != nullptr) {
      m_testing_dataset.total_samples() = m_data_readers[execution_mode::testing]->get_num_data();
    }

    m_active_buffer[execution_mode::training].store(-1);
    m_active_buffer[execution_mode::validation].store(-1);
    m_active_buffer[execution_mode::testing].store(-1);
  }

  ~generic_input_layer() override {

    // Synchronize the I/O thread pool
    // Note: The thread pool may still be running asynchronously if the
    // trainer is destroyed in the middle of an epoch. The thread pool
    // needs to interact with data readers, etc., so it needs to be
    // synchronized before any of them are destroyed.
    if (this->m_model != nullptr) {
      if (this->m_model->has_valid_execution_context()) {
        this->m_model->get_execution_context().get_io_thread_pool().reap_threads();
      }
    }

    for (auto& io_buffer : m_io_buffers) {
      delete io_buffer;
    }
    // Input layer always frees data readers.
    for (auto& dr : m_data_readers) {
      delete dr.second;
    }
  }

  // Input layers copy their datareaders.
  generic_input_layer(const generic_input_layer& other)
    : io_layer(other),
      m_io_buffers(other.m_io_buffers),
      m_training_dataset(other.m_training_dataset),
      m_testing_dataset(other.m_testing_dataset),
      m_validation_dataset(other.m_validation_dataset),
      m_data_readers(other.m_data_readers) {
    for (auto& io_buffer : m_io_buffers) {
      io_buffer = io_buffer->copy();
    }
    for (auto& dr : m_data_readers) {
      dr.second = dr.second->copy();
    }
  }

  generic_input_layer& operator=(const generic_input_layer& other) {
    io_layer::operator=(other);
    for (auto& io_buffer : m_io_buffers) {
      io_buffer = io_buffer->copy();
    }
    for (auto& dr : m_data_readers) {
      dr.second = dr.second->copy();
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

  template<typename T_io_buffer>
  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);

  std::string get_type() const override { return "generic_input"; }

  description get_description() const override {
    auto desc = io_layer::get_description();
    desc.add("Buffer", m_io_buffers[0]->get_type());
    return desc;
  }

  void setup_dims() override {
    io_layer::setup_dims();
    for (int i = 0; i < get_num_children(); ++i) {
      set_output_dims(get_data_dims(i), i);
    }
  }

  void setup_data() override {
    io_layer::setup_data();

    // Resize output to maximum mini-batch size
    const auto& max_mb_size = this->m_model->get_max_mini_batch_size();
    for (int i = 0; i < get_num_children(); ++i) {
      auto& output = get_activations(i);
      output.Resize(output.Height(), max_mb_size);
    }

    if(io_layer::m_data_set_spans_models) {
      calculate_num_iterations_per_epoch_training_spans_models(max_mb_size);
    } else {
      calculate_num_iterations_per_epoch_training_unique_per_models(max_mb_size);
    }

    for (auto& io_buffer : m_io_buffers) {
      int linearized_target_size;
      switch(m_data_reader_mode) {
      case data_reader_target_mode::REGRESSION:
        linearized_target_size = get_linearized_response_size();
        break;
      case data_reader_target_mode::RECONSTRUCTION:
        linearized_target_size = get_linearized_data_size();
        break;
      case data_reader_target_mode::CLASSIFICATION:
        linearized_target_size = get_linearized_label_size();
        break;
      case data_reader_target_mode::NA:
      default:
        linearized_target_size = 0;
      }
      io_buffer->setup_data(get_output_size(0),
                            linearized_target_size,
                            max_mb_size);
    }
  }

  /** Setup output tensors.
   *  Sets up the effective (global) mini-batch size.
   */
  void fp_setup_outputs(El::Int mini_batch_size) override {
    /// During model setup there is no valid execution context, but
    /// during execution there is a context
    if(this->m_model->has_valid_execution_context()) {
      // Determine model mini-batch size and effective mini-batch size
      // Note: If inter-model communication is activated, the effective
      // mini-batch is equal to the global mini-batch size.
      /// @todo This functionality should probably be moved elsewhere
      mini_batch_size = get_current_mini_batch_size();

      auto effective_mini_batch_size = mini_batch_size;
      for (auto&& cb : this->m_model->get_callbacks()) {
        if (dynamic_cast<callback::imcomm*>(cb) != nullptr) {
          effective_mini_batch_size = get_current_global_mini_batch_size();
          break;
        }
      }

      auto& c = static_cast<sgd_execution_context&>(this->m_model->get_execution_context());
      // Set mini-batch size in model
      c.set_current_mini_batch_size(mini_batch_size);
      c.set_effective_mini_batch_size(effective_mini_batch_size);
    }

    // Initialize matrices
    io_layer::fp_setup_outputs(mini_batch_size);

    for (auto& io_buffer : m_io_buffers) {
      for (int i = 0; i < get_num_children(); ++i) {
        io_buffer->fp_setup_data(mini_batch_size, i);
      }
    }
  }

  void fetch_data_in_background(int future_active_buffer, execution_mode mode) {
    int active_buffer = future_active_buffer % m_io_buffers.size();
    generic_io_buffer* io_buffer = m_io_buffers[active_buffer];
    std::lock_guard<std::mutex> guard(dr_mutex);
    setup_next_io_buffer(io_buffer);
    prof_region_begin("fetch_sample", prof_colors[0], false);
    io_buffer->fetch_to_local_matrix(get_data_reader(mode), mode);
    prof_region_end("fetch_sample", false);
    return;
  }

  /// Check for each buffer if there is an outstanding fetch request
  void collect_background_data_fetch(execution_mode mode) {
    for(auto& io_buffer : m_io_buffers) {
      if(io_buffer->is_data_fetched_in_background(mode)) {
        io_buffer->get_data_fetch_future(mode).get();
        io_buffer->set_fetch_data_in_background(false, mode);
      }
    }
  }

  void fp_compute() override {
    execution_mode mode = this->m_model->get_execution_context().get_execution_mode();

    increment_active_buffer_idx(mode);

    generic_io_buffer* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];

    // If there is no valid data and there is not already a background
    // thread to fetch the data, queue up the background thread
    if(io_buffer->num_samples_ready(mode) == 0 && !io_buffer->is_data_fetched_in_background(mode)) {
      std::future<void> background_fetch_done = this->m_model->get_execution_context().get_io_thread_pool().submit_job(
        std::bind(&generic_input_layer::fetch_data_in_background, this, get_active_buffer_idx(mode), mode));
      io_buffer->set_data_fetch_future(std::move(background_fetch_done), mode);
      io_buffer->set_fetch_data_in_background(true, mode);
    }

    // Wait for the background thread to complete fetching the data
    if(io_buffer->is_data_fetched_in_background(mode)) {
      io_buffer->get_data_fetch_future(mode).get();
      io_buffer->set_fetch_data_in_background(false, mode);
    }

    int num_samples_in_batch = 0;
    if(io_buffer->num_samples_ready(mode) > 0) {
      num_samples_in_batch = io_buffer->num_samples_ready(mode);
    }else {
        if(!get_data_reader()->position_is_overrun()) {
          std::stringstream err;
          err << "I/O buffer does not contain valid samples ("<< num_samples_in_batch << ")";
          // LBANN_ERROR(err.str());
        }
    }

    if(dynamic_cast<partitioned_io_buffer*>(io_buffer) != nullptr) {
      // Use the predetermined size of the mini-batch to set the current
      // batch size for the neural network
      num_samples_in_batch = get_current_mini_batch_size();

      update_num_samples_processed(num_samples_in_batch);
      if(m_expected_num_child_layers == 1) {
        io_buffer->distribute_from_local_matrix(get_data_reader(), mode, get_activations(0));
      }else {
        io_buffer->distribute_from_local_matrix(get_data_reader(), mode, get_activations(0), get_activations(1));
      }
    }else {
          LBANN_ERROR("could not fp_compute for I/O layers : encoutered generic_io_buffer type");
    }

    m_data_set_processed = io_buffer->update_data_set(get_data_reader(mode), mode);

    if(!m_data_set_processed && this->m_model->get_execution_context().background_io_activity_allowed()) {
      int next_active_buffer = get_active_buffer_idx(mode) + 1;
      std::future<void> background_fetch_done = this->m_model->get_execution_context().get_io_thread_pool().submit_job(
        std::bind(&generic_input_layer::fetch_data_in_background, this, next_active_buffer, mode));
      generic_io_buffer* next_io_buffer = m_io_buffers[next_active_buffer % m_io_buffers.size()];
      next_io_buffer->set_data_fetch_future(std::move(background_fetch_done), mode);
      next_io_buffer->set_fetch_data_in_background(true, mode);
    }

#ifdef LBANN_HAS_DISTCONV
    // When enabled, shuffle the input samples and copy them to a device tensor
    if (this->distconv_enabled()) {
      fp_compute_distconv();
    }
#endif
  }

  void setup_next_io_buffer(generic_io_buffer* io_buffer) {
    int mini_batch_size = get_current_mini_batch_size();
    for (int i = 0; i < get_num_children(); ++i) {
      io_buffer->fp_setup_data(mini_batch_size, i);
    }
  }

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() override {
    return m_data_set_processed;
  }

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

  generic_data_reader *get_data_reader() const {
    return get_data_reader(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_num_parallel_readers(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_num_parallel_readers() : 0;
  }

  virtual int get_num_parallel_readers() const {
    return get_num_parallel_readers(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_num_iterations_per_epoch(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_num_iterations_per_epoch() : 0;
  }

  virtual int get_num_iterations_per_epoch() const {
    return get_num_iterations_per_epoch(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_current_step_in_epoch(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_step_in_epoch() : 0;
  }

  virtual int get_current_step_in_epoch() const {
    return get_current_step_in_epoch(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_mini_batch_size() : 0;
  }

  virtual int get_last_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_last_mini_batch_size() : 0;
  }

  virtual int get_last_mini_batch_size() const {
    return get_last_mini_batch_size(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_current_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_mini_batch_size() : 0;
  }

  virtual int get_current_mini_batch_size() const {
    return get_current_mini_batch_size(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_global_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_global_mini_batch_size() : 0;
  }

  virtual int get_global_last_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_global_last_mini_batch_size() : 0;
  }

  virtual int get_current_global_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_global_mini_batch_size() : 0;
  }

  virtual int get_current_global_mini_batch_size() const {
    return get_current_global_mini_batch_size(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_world_master_mini_batch_adjustment(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_world_master_mini_batch_adjustment() : 0;
  }

  virtual int get_world_master_mini_batch_adjustment() const {
    return get_world_master_mini_batch_adjustment(this->m_model->get_execution_context().get_execution_mode());
  }

  virtual int get_current_world_master_mini_batch_adjustment(execution_mode mode, int model_rank) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_world_master_mini_batch_adjustment(model_rank) : 0;
  }

  virtual int get_current_world_master_mini_batch_adjustment(int model_rank) const {
    return get_current_world_master_mini_batch_adjustment(this->m_model->get_execution_context().get_execution_mode(), model_rank);
  }

  /** Calculate how many iterations are required for training, testing,
   *  and validation given a specified mini-batch size and that the
   *  training data set is spanning all of the models.
   */
  void calculate_num_iterations_per_epoch_training_spans_models(int mini_batch_size) {

    generic_data_reader *dr = get_data_reader(execution_mode::training);
    if(dr != nullptr) {
      /// Setup the training data set so that it spans all models
      m_io_buffers[0]->calculate_num_iterations_per_epoch_spanning_models(mini_batch_size, dr);
    }

    dr = get_data_reader(execution_mode::validation);
    if(dr != nullptr) {
      /// Each model uses the entire validation and testing data sets
      m_io_buffers[0]->calculate_num_iterations_per_epoch_single_model(mini_batch_size, dr);
    }

    dr = get_data_reader(execution_mode::testing);
    if(dr != nullptr) {
      m_io_buffers[0]->calculate_num_iterations_per_epoch_single_model(mini_batch_size, dr);
    }

  }

  void calculate_num_iterations_per_epoch_training_unique_per_models(int mini_batch_size) {

    generic_data_reader *dr = get_data_reader(execution_mode::training);
    if(dr != nullptr) {
      /// Setup the training data set so that it spans all models
      m_io_buffers[0]->calculate_num_iterations_per_epoch_single_model(mini_batch_size, dr);
    }

    dr = get_data_reader(execution_mode::validation);
    if(dr != nullptr) {
      /// Each model uses the entire validation and testing data sets
      m_io_buffers[0]->calculate_num_iterations_per_epoch_single_model(mini_batch_size, dr);
    }

    dr = get_data_reader(execution_mode::testing);
    if(dr != nullptr) {
      m_io_buffers[0]->calculate_num_iterations_per_epoch_single_model(mini_batch_size, dr);
    }

  }

  //************************************************************************
  // Helper functions to access the dataset statistics
  //************************************************************************
  dataset& get_dataset(execution_mode m) override {
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

  const dataset& get_dataset(execution_mode m) const override {
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
   * Return the dataset associated with the current execution mode.
   */
  dataset& select_dataset() override { return get_dataset(m_model->get_execution_context().get_execution_mode()); }
  const dataset& select_dataset() const override { return get_dataset(m_model->get_execution_context().get_execution_mode()); }

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  dataset* select_first_valid_dataset() override {
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

  /**
   * Return the data reader associated with the current execution mode.
   */
  generic_data_reader *select_data_reader() const override {
    return get_data_reader();
  }

  /**
   * Update the number of samples processed for the current execution mode.
   */
  long update_num_samples_processed(long num_samples) override {
    dataset& ds = select_dataset();
    ds.num_samples_processed() += num_samples;
    return ds.get_num_samples_processed();
  }

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
  El::Matrix<El::Int>* get_sample_indices_per_mb() override {
    execution_mode mode = this->m_model->get_execution_context().get_execution_mode();
    generic_io_buffer* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];
    return io_buffer->get_sample_indices_fetched_per_mb(this->m_model->get_execution_context().get_execution_mode());
  }

  /**
   * Get the dimensions of the underlying data.
   */
  const std::vector<int> get_data_dims(int child_index = 0) const override {
    // Check the training and testing execution modes for data dimensions
    const generic_data_reader *dr = get_data_reader(execution_mode::training);
    // If there isn't a training data reader, use the testing data reader
    if(dr == nullptr) {
      dr = get_data_reader(execution_mode::testing);
    }
    if(dr == nullptr) { LBANN_ERROR("unable to call get_data_dims -- no valid execution mode"); }
    //    dataset* ds = select_first_valid_dataset();
    if (dr) {
      if(child_index == 0) {
        return dr->get_data_dims();
      }else if(child_index == 1) {
        switch(m_data_reader_mode) {
        case data_reader_target_mode::REGRESSION:
          return std::vector<int>(1, dr->get_num_responses());
        case data_reader_target_mode::RECONSTRUCTION:
          return dr->get_data_dims();
        case data_reader_target_mode::CLASSIFICATION:
        default:
          return std::vector<int>(1, dr->get_num_labels());
        }
        //        the correct value based on initialization
      }else {
        LBANN_ERROR("get_data_dims: Invalid child index");
      }
    }
    return std::vector<int>(1, 0);
  }

  /**
   * Get the linearized size of the underlying data.
   */
  long get_linearized_data_size() const override {
    long linearized_data_size = -1;

    data_reader_map_t::const_iterator it;

    it = m_data_readers.find(execution_mode::training);
    if ((it != m_data_readers.end()) && it->second) {
      linearized_data_size = (it->second)->get_linearized_data_size();
    }

    it = m_data_readers.find(execution_mode::validation);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_data_size = (it->second)->get_linearized_data_size();
      if (linearized_data_size != -1 && linearized_data_size != tmp_data_size) {
        LBANN_ERROR("lbann_io_layer: validation data set size does not "
                              "match the currently established data set size");
      }
    }

    it = m_data_readers.find(execution_mode::testing);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_data_size = (it->second)->get_linearized_data_size();
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
  long get_linearized_label_size() const override {
    if (is_for_regression()) {
      return static_cast<long>(1);
    }
    long linearized_label_size = -1;
    data_reader_map_t::const_iterator it;

    it = m_data_readers.find(execution_mode::training);
    if ((it != m_data_readers.end()) && it->second) {
      linearized_label_size = (it->second)->get_linearized_label_size();
    }
    it = m_data_readers.find(execution_mode::validation);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_label_size = (it->second)->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        LBANN_ERROR("lbann_io_layer: validation label set size (" + std::to_string(tmp_label_size) + ") does not match the currently established data set size (" + std::to_string(linearized_label_size) + ")");
      }
    }
    it = m_data_readers.find(execution_mode::testing);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_label_size = (it->second)->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        LBANN_ERROR("lbann_io_layer: testing label set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_label_size;
  }

  long get_linearized_response_size() const override {
    if (!is_for_regression()) {
      return static_cast<long>(1);
    }
    long linearized_response_size = -1;
    data_reader_map_t::const_iterator it;

    it = m_data_readers.find(execution_mode::training);
    if ((it != m_data_readers.end()) && it->second) {
      linearized_response_size = (it->second)->get_linearized_response_size();
    }
    it = m_data_readers.find(execution_mode::validation);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_response_size = (it->second)->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        LBANN_ERROR("lbann_io_layer: validation response set size does not "
                              "match the currently established data set size");
      }
    }
    it = m_data_readers.find(execution_mode::testing);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_response_size = (it->second)->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        LBANN_ERROR("lbann_io_layer: testing response set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_response_size;
  }

  long get_num_samples_trained() const override {
    return m_training_dataset.get_num_samples_processed();
  }
  long get_num_samples_tested() const override {
    return m_testing_dataset.get_num_samples_processed();
  }
  long get_total_num_training_samples() const override {
    return m_training_dataset.get_total_samples();
  }
  long get_total_num_testing_samples() const override {
    return m_testing_dataset.get_total_samples();
  }

  bool at_new_epoch() const override {
    const data_reader_map_t::const_iterator it = m_data_readers.find(execution_mode::training);
    return ((it != m_data_readers.end()) && it->second && (it->second)->at_new_epoch());
  }

  bool is_execution_mode_valid(execution_mode mode) const override {
    const dataset& ds = get_dataset(mode);
    return (ds.get_total_samples() != static_cast<long>(0));
  }
  //************************************************************************
  //
  //************************************************************************

  // save state of IO to a checkpoint
  bool save_to_checkpoint_shared(persist& p) const override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, execution_mode::training);
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, execution_mode::testing);
      }
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, execution_mode::validation);
      }

      if (get_comm()->am_trainer_master()) {
        write_cereal_archive<const generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
      }

    }
    return true;
  }

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::validation);
    }

    std::string buf;
    if (get_comm()->am_trainer_master()) {
      read_cereal_archive<generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
      buf = create_cereal_archive_binary_string<generic_input_layer>(*this);
   }

    // TODO: this assumes homogeneous processors
    // broadcast state from rank 0
    get_comm()->trainer_broadcast(0, buf);

    if (!get_comm()->am_trainer_master()) {
      unpack_cereal_archive_binary_string<generic_input_layer>(*this, buf);
    }

    return true;
  }

  bool save_to_checkpoint_distributed(persist& p) const override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() == callback_type::execution_context_only || p.get_cb_type() == callback_type::full_checkpoint) {
      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, execution_mode::training);
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, execution_mode::testing);
      }
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, execution_mode::validation);
      }

      write_cereal_archive<const generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
    }
    return true;
  }

  bool load_from_checkpoint_distributed(persist& p) override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, execution_mode::validation);
    }

    read_cereal_archive<generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
    return true;
  }

  int get_active_buffer_idx(execution_mode m) {
    return m_active_buffer[m].load();
  }
  void increment_active_buffer_idx(execution_mode m) {
    m_active_buffer[m]++;
  }

 protected:
  std::vector<generic_io_buffer*> m_io_buffers;
  io_buffer_map_t m_active_buffer;

  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;
  //  bool m_data_sets_span_models;

  data_reader_map_t m_data_readers;
 //  std::map<execution_mode, dataset_stats> m_dataset_stats;
  bool m_data_set_processed;
  std::mutex dr_mutex;

#ifdef LBANN_HAS_DISTCONV
 public:
  void setup_tensors_fwd(
      const std::array<dc::Dist, dc::num_dists> &dists) override {
    using namespace dc;
    Layer::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;

    // copies the label data as well when the second child layer is
    // also enabled for distconv
    if (get_num_children() == 2 && get_child_layers()[1]->using_distconv()) {
      m_copy_labels_dc = true;
      dc::MPIRootPrintStreamInfo() << "Copy label/response data to Distconv as well";
    }

    const auto tensor_shape = get_output_tensor_shape();
    const Dist sample_dist = Layer::get_hydrogen_matrix_distribution();
    auto local_shape = tensor_shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    local_shape[dc::get_sample_dim()] = 0;
    const auto &dist = dists[1];
    auto dist_no_halo = dist;
    dist_no_halo.clear_overlap();

    // Use the same MPI communicator for both IO buffers. This seems
    // to work around MPI errors likely caused with the alltoallv for
    // shuffling.
    const LocaleMPI loc(dc::get_mpi_comm(), false);

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // Assumes the input buffer is already partitioned for
      // Distconv
      m_input_host_view = TensorHost(tensor_shape, loc, dist_no_halo);
      // Create a Distconv tensor at host memory.
      m_input_host_tensor = TensorHost(tensor_shape, loc, dist_no_halo);
    } else {
      // Create a view to the host Elemental matrix
      m_input_host_view = TensorHost(tensor_shape, loc, sample_dist, local_shape);
      // Create a Distconv tensor at host memory.
      m_input_host_tensor = TensorHost(tensor_shape, loc, dist);
    }
    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      // TODO: This is a temporary hack. Should use
      // CUDAHostPooledAllocator, but the shuffler is
      // only specialized for BaseAllocator.
#if 0
      assert0(m_input_host_tensor.allocate());
#else
      size_t buf_size = m_input_host_tensor.get_local_real_size()
          * sizeof(InputType);
      dc::MPIPrintStreamInfo() << "buf size: " << buf_size;
      InputType *buf = nullptr;
      CHECK_CUDA(cudaMallocHost(&buf, buf_size));
      // Note buf should be deallocated.
      dc::tensor::View(m_input_host_tensor, buf);
#endif
    }

    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      setup_shuffler_buffers(m_input_host_view, m_input_host_tensor);
    }

    // Layer::setup_activations_tensor does not work as it assumes
    // prev_activations_tensor is already
    // setup. prev_activations_tensor is not necessary for input.
    //const LocaleMPI loc(dc::get_mpi_comm(), false);
    m_activations_t = TensorDev(tensor_shape, loc, dist);
    assert0(m_activations_t.allocate());
    m_activations_t.zero(dc::get_stream());

    // Keeps the same input type and convert to float on GPU
    m_input_dev = TensorDevInput(tensor_shape, loc, dist);
    assert0(m_input_dev.allocate());
    m_input_dev.zero(dc::get_stream());

    // Allocate pinned memory buffer for copying input
    if (dc::is_cosmoflow_parallel_io_enabled()) {
      auto req_size = m_input_dev.get_local_real_size() * sizeof(InputType);
      CHECK_CUDA(cudaMallocHost(&m_copy_pinned_buffer, req_size));
    }

    if (m_copy_labels_dc) {
      setup_label_tensors();
    }
  }

  void setup_tensors_bwd(
      const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    // Nothing to do as this is an input layer
  }

  void fp_setup_distconv(int mini_batch_size) {
    if (!distconv_enabled()) return;
    // Nothing to do here as everything is done in fp_compute_distconv.
  }

 protected:

  bool m_copy_labels_dc = false;

#ifdef LBANN_DISTCONV_COSMOFLOW_KEEP_INT16
  // Cosmoflow samples are kept stored in int16
  using InputType = short;
#else
  using InputType = DataType;
#endif // LBANN_DISTCONV_COSMOFLOW_KEEP_INT16

  using TensorHost = dc::TensorHost<InputType>;
  using TensorShuffler = dc::TensorHostShuffler<InputType>;
  using TensorDevInput = ::distconv::tensor::Tensor<
    InputType, ::distconv::tensor::LocaleMPI,
    ::distconv::tensor::CUDAAllocator>;

  // 3 last-MB shufflers for training/validation/testing
  dc::TensorHost<InputType> m_input_host_view;
  dc::TensorHost<InputType> m_input_host_tensor;
  TensorDevInput m_input_dev;
  // shufflers for the input data
  std::unique_ptr<TensorShuffler> m_input_shuffler;
  std::array<std::unique_ptr<TensorShuffler>, 3> m_input_shuffler_last_mb;
  std::unique_ptr<InputType> m_shuffler_src_buf;
  size_t m_shuffler_src_buf_size = 0;
  std::unique_ptr<InputType> m_shuffler_dst_buf;
  size_t m_shuffler_dst_buf_size = 0;

  dc::TensorHost<InputType> m_labels_host_view;
  dc::TensorHost<InputType> m_labels_host_tensor;
  dc::TensorDev m_labels_dev;
  TensorDevInput m_labels_input_type;
  // shufflers for the labels
  std::unique_ptr<TensorShuffler> m_label_shuffler;
  std::array<std::unique_ptr<TensorShuffler>, 3> m_label_shuffler_last_mb;

  InputType *m_copy_pinned_buffer = nullptr;

  const dc::TensorDev &get_activations_t(const Layer &child) const override {
    const int child_index = std::find(get_child_layers().begin(),
                                      get_child_layers().end(),
                                      &child) - get_child_layers().begin();
    if (child_index >= get_num_children()) {
      LBANN_ERROR("Invalid child layer");
    }
    if (child_index == 0) {
      return m_activations_t;
    } else {
      assert_eq(child_index, 1);
      return m_labels_dev;
    }
  }

  void setup_shuffler_buffers(const TensorHost &src, const TensorHost &dst) {
    auto shuffler_src_size = TensorShuffler::get_buf_size(src);
    if (m_shuffler_src_buf_size < shuffler_src_size) {
      m_shuffler_src_buf_size = shuffler_src_size;
      m_shuffler_src_buf =
          std::unique_ptr<InputType>(static_cast<InputType*>(
              dc::util::aligned_malloc(m_shuffler_src_buf_size)));
    }
    auto shuffler_dst_size = TensorShuffler::get_buf_size(dst);
    if (m_shuffler_dst_buf_size < shuffler_dst_size) {
      m_shuffler_dst_buf_size = shuffler_dst_size;
      m_shuffler_dst_buf =
          std::unique_ptr<InputType>(static_cast<InputType*>(
              dc::util::aligned_malloc(m_shuffler_dst_buf_size)));
    }
  }

  TensorShuffler &get_shuffler(const TensorHost &src, const TensorHost &dst,
                               bool is_label) {
    size_t cur_mb_size = src.get_shape()[dc::get_sample_dim()];
    auto src_buf = m_shuffler_src_buf.get();
    auto dst_buf = m_shuffler_dst_buf.get();
    if (cur_mb_size == this->get_model()->get_max_mini_batch_size()) {
      auto &shfl = is_label ? m_label_shuffler : m_input_shuffler;
      if (shfl == nullptr) {
        dc::MPIPrintStreamDebug() << "Creating host shuffler: "
                                  << src << " -> " << dst;
        shfl.reset(new TensorShuffler(
            src, dst, src_buf, dst_buf));
      }
      return *shfl;
    } else {
      // The last remaining mini-batches for the train, validation, and
      // testing modes
      auto mode = this->m_model->get_execution_context().get_execution_mode();
      int shfl_idx = static_cast<int>(mode);
      assert_always(shfl_idx >= 0 && shfl_idx < 3);
      auto &shfl = is_label ? m_label_shuffler_last_mb.at(shfl_idx) :
          m_input_shuffler_last_mb.at(shfl_idx);
      if (shfl == nullptr) {
        dc::MPIPrintStreamDebug() << "Creating host last-mb shuffler: "
                                  << src << " -> " << dst;
        shfl.reset(new TensorShuffler(
            src, dst, src_buf, dst_buf));
      }
      return *shfl;
    }
  }

  void fp_compute_distconv() {
    if (!distconv_enabled()) return;

    // Note that the mini-batch size of the data reader is not
    // actually the one for the current mini-batch as the mini-batch
    // index is already updated by fp_compute.
    const int mb_size = static_cast<sgd_execution_context&>(
        m_model->get_execution_context()).get_current_mini_batch_size();
    auto &input_view = m_input_host_view;
    auto &input_tensor = m_input_host_tensor;

    m_activations_t.set_outermost_dimension(mb_size);
    m_input_dev.set_outermost_dimension(mb_size);

    assert_eq(mb_size * dc::get_number_of_io_partitions(), get_activations().Width());
    input_view.set_outermost_dimension(mb_size);
    input_tensor.set_outermost_dimension(mb_size);

    // Setup view
#ifdef LBANN_DISTCONV_COSMOFLOW_KEEP_INT16
    assert0(dc::tensor::View(
        input_view,
        reinterpret_cast<const InputType*>(
            get_activations().LockedBuffer())));
#else
    assert0(dc::tensor::View(
        input_view,
        get_activations().LockedBuffer()));
#endif // LBANN_DISTCONV_COSMOFLOW_KEEP_INT16

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // The input buffer is assumed to be already partitioned
      assert0(dc::tensor::View(
          input_tensor, input_view.get_const_buffer()));
    } else {
      dc::MPIPrintStreamDebug()
          << this->get_name()
          << ": Shuffle the input LBANN tensor to Distconv tensor";
      get_shuffler(input_view, input_tensor, false).shuffle_forward(
          input_view.get_const_base_ptr(),
          input_tensor.get_base_ptr());
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (input_tensor.get_local_size() == 0) {
      copy_label_distconv(mb_size);
      return;
    }

    dc::MPIPrintStreamDebug()
        << this->get_name()
        << ": Copy the host tensor to device tensor";
    // This should not incur communication as the distributions should
    // be the same except for overlapping width. Device copy should be
    // done with cudaMemcpy3D.
    prof_region_begin("copy-to-device", prof_colors[1], false);
    // TODO: Copy doesn't seem to be working correctly, likely because
    // of the additional halo region in the destination buffer. For
    // now, avoid this with the manual copy below. Also, in the
    // Cosmoflow case, "input_tensor" is not a pinned buffer.
    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      assert0(dc::tensor::Copy(m_input_dev, input_tensor, dc::get_stream()));
    } else {
      int chan_dim = input_tensor.get_local_shape()[::distconv::get_channel_dim()];
      size_t block_size = input_tensor.get_local_size() / chan_dim;
      for (int i = 0; i < chan_dim; ++i) {
        auto dev_off =
            m_input_dev.get_local_offset(dc::IndexVector({0,0,0,i,0}));
        auto host_off = block_size * i;
        // First copy to temporary pinned buffer
        std::memcpy(m_copy_pinned_buffer + dev_off,
                    input_tensor.get_const_buffer() + host_off,
                    sizeof(short) * block_size);
      }
      CHECK_CUDA(cudaMemcpyAsync(
          m_input_dev.get_buffer(),  m_copy_pinned_buffer,
          m_input_dev.get_local_real_size() * sizeof(InputType),
          cudaMemcpyHostToDevice, dc::get_stream()));
    }
    prof_region_end("copy-to-device", false);

    {
      const auto norm_alpha_p = std::getenv("COSMOFLOW_NORMALIZE_ALPHA");
      const auto norm_beta_p  = std::getenv("COSMOFLOW_NORMALIZE_BETA");
      if(norm_alpha_p != nullptr) {
        const auto norm_alpha = std::stod(norm_alpha_p);
        const auto norm_beta = std::stod(norm_beta_p);
        prof_region_begin("cast-scale-bias-from-int16", prof_colors[1], false);
        dc::tensor::CastScaleBias(m_activations_t,
                                  m_input_dev,
                                  (DataType) norm_alpha,
                                  (DataType) norm_beta,
                                  dc::get_stream());
        prof_region_end("cast-scale-bias-from-int16", false);
      } else {
        prof_region_begin("cast-from-int16", prof_colors[1], false);
        dc::tensor::Cast(m_activations_t, m_input_dev, dc::get_stream());
        prof_region_end("cast-from-int16", false);
      }
    }
    // Note: no copy out for activation is necessary as the original
    // LBANN tensor is valid.

    // Copy label as well if necessary
    copy_label_distconv(mb_size);
  }

  // TODO: This is a temporary hack. The label tensor shape should
  //be set based on the shape set by the data reader, but the data
  //reader does not provide it. Using the shape shape as the data
  //tensor works fine for the U-Net model.
  dc::Shape get_unet_label_shape() const {
    auto shape = get_output_tensor_shape(0);
    auto label_size = get_output_tensor_shape(1).reduce_prod();
    auto num_channels = label_size / shape.reduce_prod();
    shape[-2] = num_channels;
    return shape;
  }

  void setup_label_tensors() {
    using namespace dc;
    assert_always(m_copy_labels_dc);
    //const auto tensor_shape = get_output_tensor_shape(1);
    const auto tensor_shape = get_unet_label_shape();
    const auto sample_dist = Layer::get_hydrogen_matrix_distribution();
    auto local_shape = tensor_shape;
    // calculated by Distconv
    local_shape[dc::get_sample_dim()] = 0;
    auto dist = m_activations_t.get_distribution();
    // Assumes no halo required.
    dist.clear_overlap();

    const LocaleMPI loc(dc::get_mpi_comm(), false);

    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // Assumes the input buffer is already partitioned for
      // Distconv
      m_labels_host_view = TensorHost(tensor_shape, loc, dist);
      // Create a Distconv tensor at host memory.
      m_labels_host_tensor = TensorHost(tensor_shape, loc, dist);
    } else {
      // Create a view to the host Elemental matrix
      m_labels_host_view = TensorHost(tensor_shape, loc, sample_dist, local_shape);
      // Create a Distconv tensor at host memory.
      m_labels_host_tensor = TensorHost(tensor_shape, loc, dist);
    }

    // When not partitioned yet, setup an intermediate tensor and shuffler
    if (!dc::is_cosmoflow_parallel_io_enabled()) {
      // TODO: This is a temporary hack. Should use
      // CUDAHostPooledAllocator, but the shuffler is
      // only specialized for BaseAllocator.
      size_t buf_size = m_labels_host_tensor.get_local_real_size()
          * sizeof(InputType);
      InputType *buf = nullptr;
      CHECK_CUDA(cudaMallocHost(&buf, buf_size));
      // Note buf should be deallocated.
      dc::tensor::View(m_labels_host_tensor, buf);
      setup_shuffler_buffers(m_labels_host_view, m_labels_host_tensor);
    }

    // Data may be type InputType. Use an intermediate buffer of type
    // InputType, which will be copied to the actual final label
    // tensor with casting to DataType
    m_labels_input_type = TensorDevInput(tensor_shape, loc, dist);
    assert0(m_labels_input_type.allocate());
    m_labels_input_type.zero(dc::get_stream());

    // The final label tensor
    m_labels_dev = TensorDev(tensor_shape, loc, dist);
    assert0(m_labels_dev.allocate());
    m_labels_dev.zero(dc::get_stream());

    dc::MPIRootPrintStreamInfo() << "label tensor: " << m_labels_dev;
  }

  void copy_label_distconv(int mb_size) {
    if (!m_copy_labels_dc) return;
    constexpr int mat_idx = 1;
    assert_eq(mb_size * dc::get_number_of_io_partitions(),
              get_activations(mat_idx).Width());

    // Adjust the sample size
    m_labels_host_view.set_outermost_dimension(mb_size);
    m_labels_host_tensor.set_outermost_dimension(mb_size);
    m_labels_dev.set_outermost_dimension(mb_size);
    m_labels_input_type.set_outermost_dimension(mb_size);

    // Setup view to the LBANN matrix
#ifdef LBANN_DISTCONV_COSMOFLOW_KEEP_INT16
    assert0(dc::tensor::View(
        m_labels_host_view,
        reinterpret_cast<const InputType*>(
            get_activations(mat_idx).LockedBuffer())));
#else
    assert0(dc::tensor::View(
        m_labels_host_view,
        get_activations(mat_idx).LockedBuffer()));
#endif // LBANN_DISTCONV_COSMOFLOW_KEEP_INT16

    // Shuffle if necessary
    if (dc::is_cosmoflow_parallel_io_enabled()) {
      // The input buffer is assumed to be already partitioned
      assert0(dc::tensor::View(
          m_labels_host_tensor, m_labels_host_view.get_const_buffer()));
    } else {
      dc::MPIPrintStreamDebug()
          << this->get_name()
          << ": Shuffle the label LBANN tensor to Distconv tensor";
      get_shuffler(m_labels_host_view, m_labels_host_tensor, true).shuffle_forward(
          m_labels_host_view.get_const_base_ptr(),
          m_labels_host_tensor.get_base_ptr());
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (m_labels_host_tensor.get_local_size() == 0) {
      return;
    }

    // Cpoy the host tensor to device
    dc::MPIPrintStreamDebug() << "Copy the host label to device tensor";
    prof_region_begin("label-copy-to-device", prof_colors[1], false);
    assert0(dc::tensor::Copy(m_labels_input_type, m_labels_host_tensor, dc::get_stream()));
    prof_region_end("label-copy-to-device", false);

    // Cast to DataType. Just a copy if both tensors are in the same type.
    prof_region_begin("label-cast-from-int16", prof_colors[1], false);
    dc::tensor::Cast(m_labels_dev, m_labels_input_type, dc::get_stream());
    prof_region_end("label-cast-from-int16", false);
  }

#endif // LBANN_HAS_DISTCONV
};

template<typename T> inline void generic_input_layer::initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
  m_io_buffers.push_back(new T(comm, num_parallel_readers, data_readers, m_expected_num_child_layers));
}

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
