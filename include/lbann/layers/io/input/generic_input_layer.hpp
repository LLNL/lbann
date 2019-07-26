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
#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/callback_imcomm.hpp"
#include "lbann/utils/omp_diagnostics.hpp"

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

  template<typename T_io_buffer>
  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);

  std::string get_type() const override { return "generic_input"; }

  description get_description() const override {
    auto desc = io_layer::get_description();
    desc.add("Buffer", m_io_buffers[0]->get_type());
    desc.add("Background I/O", this->m_model->background_io_activity_allowed());
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

    auto num_io_threads = this->m_model->get_io_thread_pool()->get_num_threads();
    /// BVE FIXME foreach data reader
    // in case that target_layer gets initialized beforehand
    if(m_data_readers[execution_mode::training] != nullptr) {
      m_data_readers[execution_mode::training]->setup(num_io_threads, this->m_model->get_io_thread_pool());
      m_data_readers[execution_mode::training]->set_rank(Layer::m_comm->get_rank_in_trainer());
    }
    if(m_data_readers[execution_mode::validation] != nullptr) {
      m_data_readers[execution_mode::validation]->setup(num_io_threads, this->m_model->get_io_thread_pool());
      m_data_readers[execution_mode::validation]->set_rank(Layer::m_comm->get_rank_in_trainer());
    }
    if(m_data_readers[execution_mode::testing] != nullptr) {
      m_data_readers[execution_mode::testing]->setup(num_io_threads, this->m_model->get_io_thread_pool());
      m_data_readers[execution_mode::testing]->set_rank(Layer::m_comm->get_rank_in_trainer());
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

    // Determine model mini-batch size and effective mini-batch size
    // Note: If inter-model communication is activated, the effective
    // mini-batch is equal to the global mini-batch size.
    /// @todo This functionality should probably be moved elsewhere
    mini_batch_size = get_current_mini_batch_size();
    int effective_mini_batch_size = mini_batch_size;
    for (auto&& cb : this->m_model->get_callbacks()) {
      if (dynamic_cast<lbann_callback_imcomm*>(cb) != nullptr) {
        effective_mini_batch_size = get_current_global_mini_batch_size();
        break;
      }
    }

    // Set mini-batch size in model
    this->m_model->set_current_mini_batch_size(mini_batch_size);
    this->m_model->set_effective_mini_batch_size(effective_mini_batch_size);

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
    io_buffer->fetch_to_local_matrix(get_data_reader(mode), mode);
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
    execution_mode mode = this->m_model->get_execution_mode();

    increment_active_buffer_idx(mode);

    generic_io_buffer* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];

    // If there is no valid data and there is not already a background
    // thread to fetch the data, queue up the background thread
    if(io_buffer->num_samples_ready(mode) == 0 && !io_buffer->is_data_fetched_in_background(mode)) {
      std::future<void> background_fetch_done = this->m_model->get_io_thread_pool()->submit_job(
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
          LBANN_ERROR(err.str());
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

    if(!m_data_set_processed && this->m_model->background_io_activity_allowed()) {
      int next_active_buffer = get_active_buffer_idx(mode) + 1;
      std::future<void> background_fetch_done = this->m_model->get_io_thread_pool()->submit_job(
        std::bind(&generic_input_layer::fetch_data_in_background, this, next_active_buffer, mode));
      generic_io_buffer* next_io_buffer = m_io_buffers[next_active_buffer % m_io_buffers.size()];
      next_io_buffer->set_data_fetch_future(std::move(background_fetch_done), mode);
      next_io_buffer->set_fetch_data_in_background(true, mode);
    }
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
    return get_data_reader(this->m_model->get_execution_mode());
  }

  virtual int get_num_parallel_readers(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_num_parallel_readers() : 0;
  }

  virtual int get_num_parallel_readers() const {
    return get_num_parallel_readers(this->m_model->get_execution_mode());
  }

  virtual int get_num_iterations_per_epoch(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_num_iterations_per_epoch() : 0;
  }

  virtual int get_num_iterations_per_epoch() const {
    return get_num_iterations_per_epoch(this->m_model->get_execution_mode());
  }

  virtual int get_current_step_in_epoch(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_step_in_epoch() : 0;
  }

  virtual int get_current_step_in_epoch() const {
    return get_current_step_in_epoch(this->m_model->get_execution_mode());
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
    return get_last_mini_batch_size(this->m_model->get_execution_mode());
  }

  virtual int get_current_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_mini_batch_size() : 0;
  }

  virtual int get_current_mini_batch_size() const {
    return get_current_mini_batch_size(this->m_model->get_execution_mode());
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
    return get_current_global_mini_batch_size(this->m_model->get_execution_mode());
  }

  virtual int get_world_master_mini_batch_adjustment(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_world_master_mini_batch_adjustment() : 0;
  }

  virtual int get_world_master_mini_batch_adjustment() const {
    return get_world_master_mini_batch_adjustment(this->m_model->get_execution_mode());
  }

  virtual int get_current_world_master_mini_batch_adjustment(execution_mode mode, int model_rank) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return (data_reader != nullptr) ? data_reader->get_current_world_master_mini_batch_adjustment(model_rank) : 0;
  }

  virtual int get_current_world_master_mini_batch_adjustment(int model_rank) const {
    return get_current_world_master_mini_batch_adjustment(this->m_model->get_execution_mode(), model_rank);
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
  dataset& select_dataset() override { return get_dataset(m_model->get_execution_mode()); }
  const dataset& select_dataset() const override { return get_dataset(m_model->get_execution_mode()); }

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
    execution_mode mode = this->m_model->get_execution_mode();
    generic_io_buffer* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];
    return io_buffer->get_sample_indices_fetched_per_mb(this->m_model->get_execution_mode());
  }

  /**
   * Get the dimensions of the underlying data.
   */
  const std::vector<int> get_data_dims(int child_index = 0) const override {
    const generic_data_reader *dr = get_data_reader();
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
      std::cerr << "XX >>>>>> linearized_data_size: " << linearized_data_size << "\n";
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
    if(p.get_cb_type() != callback_type::validation){
      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, "data_reader_training");
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, "data_reader_testing");
      }
      if (m_comm->am_trainer_master()) {
        p.write_uint64(persist_type::train, "reader_train_processed",
                       (uint64_t) m_training_dataset.get_num_samples_processed());
        p.write_uint64(persist_type::train, "reader_train_total",
                       (uint64_t) m_training_dataset.get_total_samples());

        p.write_uint64(persist_type::train, "reader_test_processed",
                       (uint64_t) m_testing_dataset.get_num_samples_processed());
        p.write_uint64(persist_type::train, "reader_test_total",
                     (uint64_t) m_testing_dataset.get_total_samples());

      }
    }
    if(p.get_cb_type() == callback_type::validation || p.get_cb_type() == callback_type::batch){
      if (m_comm->am_trainer_master()) {
        p.write_uint64(persist_type::validate, "reader_validate_processed",
                       (uint64_t) m_validation_dataset.get_num_samples_processed());
        p.write_uint64(persist_type::validate, "reader_validate_total",
                       (uint64_t) m_validation_dataset.get_total_samples());
      }
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, "data_reader_validation");
      }
    }
    return true;
  }

  struct dataset_header {
    uint64_t train_proc;
    uint64_t train_total;
    uint64_t test_proc;
    uint64_t test_total;
    uint64_t validate_proc;
    uint64_t validate_total;
  };

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, "data_reader_training");
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, "data_reader_testing");
    }

    // save our own state
    // rank 0 reads the file
    dataset_header header;
    // Assume we are loading from a epoch end checkpoint
    if (m_comm->am_trainer_master()) {
      p.read_uint64(persist_type::train, "reader_train_processed",    &header.train_proc);
      p.read_uint64(persist_type::train, "reader_train_total",        &header.train_total);
      p.read_uint64(persist_type::train, "reader_test_processed",     &header.test_proc);
      p.read_uint64(persist_type::train, "reader_test_total",         &header.test_total);
      if(m_data_readers[execution_mode::validation] != nullptr){
        p.read_uint64(persist_type::validate, "reader_validate_processed", &header.validate_proc);
        p.read_uint64(persist_type::validate, "reader_validate_total",     &header.validate_total);
      }
    }

    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, "data_reader_validation");
    }
    // TODO: assumes homogeneous hardware
    // broadcast data from rank 0
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);
    // set our fields
    m_training_dataset.num_samples_processed()   = (long) header.train_proc;
    m_training_dataset.total_samples()           = (long) header.train_total;
    m_testing_dataset.num_samples_processed()    = (long) header.test_proc;
    m_testing_dataset.total_samples()            = (long) header.test_total;
    if(m_data_readers[execution_mode::validation] != nullptr){
      m_validation_dataset.num_samples_processed() = (long) header.validate_proc;
      m_validation_dataset.total_samples()         = (long) header.validate_total;
    }
    return true;
  }

  bool save_to_checkpoint_distributed(persist& p) const override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() != callback_type::validation){
      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, "data_reader_training");
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, "data_reader_testing");
      }
      p.write_uint64(persist_type::train, "reader_train_processed",
                     (uint64_t) m_training_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::train, "reader_train_total",
                     (uint64_t) m_training_dataset.get_total_samples());

      p.write_uint64(persist_type::train, "reader_test_processed",
                     (uint64_t) m_testing_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::train, "reader_test_total",
                   (uint64_t) m_testing_dataset.get_total_samples());

    }
    if(p.get_cb_type() == callback_type::validation || p.get_cb_type() == callback_type::batch){
      p.write_uint64(persist_type::validate, "reader_validate_processed",
                     (uint64_t) m_validation_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::validate, "reader_validate_total",
                     (uint64_t) m_validation_dataset.get_total_samples());
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, "data_reader_validation");
      }

    }
    return true;
  }

  bool load_from_checkpoint_distributed(persist& p) override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, "data_reader_training");
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, "data_reader_testing");
    }
    // save our own state
    // rank 0 reads the file
    dataset_header header;
    p.read_uint64(persist_type::train, "reader_train_processed",    &header.train_proc);
    p.read_uint64(persist_type::train, "reader_train_total",        &header.train_total);
    p.read_uint64(persist_type::train, "reader_test_processed",     &header.test_proc);
    p.read_uint64(persist_type::train, "reader_test_total",         &header.test_total);
    if(m_data_readers[execution_mode::validation] != nullptr){
      p.read_uint64(persist_type::validate, "reader_validate_processed", &header.validate_proc);
      p.read_uint64(persist_type::validate, "reader_validate_total",     &header.validate_total);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, "data_reader_validation");
    }

    // set our fields
    m_training_dataset.num_samples_processed()   = (long) header.train_proc;
    m_training_dataset.total_samples()           = (long) header.train_total;
    m_testing_dataset.num_samples_processed()    = (long) header.test_proc;
    m_testing_dataset.total_samples()            = (long) header.test_total;
    if(m_data_readers[execution_mode::validation] != nullptr){
      m_validation_dataset.num_samples_processed() = (long) header.validate_proc;
      m_validation_dataset.total_samples()         = (long) header.validate_total;
    }
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
};

template<typename T> inline void generic_input_layer::initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
  m_io_buffers.push_back(new T(comm, num_parallel_readers, data_readers, m_expected_num_child_layers));
}

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
