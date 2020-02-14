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

#include <future>

namespace lbann {

/** @todo Move functionality to input_layer. */
template <typename TensorDataType>
class generic_input_layer : public io_layer<TensorDataType> {
 public:
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
  generic_input_layer(lbann_comm *comm,
              int num_parallel_readers,
              data_reader_target_mode dr_mode = data_reader_target_mode::CLASSIFICATION)
    : io_layer<TensorDataType>(comm, dr_mode),
      m_io_buffers() {
      //m_data_sets_span_models(data_sets_span_models) {
    // Input layers have no parents
    this->m_expected_num_parent_layers = 0;
    if(dr_mode == data_reader_target_mode::NA) {
      this->m_expected_num_child_layers = 1;
    }else {
      // Input layers output a sample and target, which could be the
      // original value, categorical label, or regression value
      this->m_expected_num_child_layers = 2;
    }

    this->m_active_buffer[execution_mode::training].store(-1);
    this->m_active_buffer[execution_mode::validation].store(-1);
    this->m_active_buffer[execution_mode::testing].store(-1);
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
  }

  // Input layers copy their datareaders.
  generic_input_layer(const generic_input_layer& other)
    : io_layer<TensorDataType>(other),
      m_io_buffers(other.m_io_buffers) {
    for (auto& io_buffer : m_io_buffers) {
      io_buffer = io_buffer->copy();
    }
  }

  generic_input_layer& operator=(const generic_input_layer& other) {
    io_layer<TensorDataType>::operator=(other);
    for (auto& io_buffer : m_io_buffers) {
      io_buffer = io_buffer->copy();
    }
    return *this;
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize( Archive & ar ) {
    // ar(/*CEREAL_NVP(m_io_buffer),
    //    CEREAL_NVP(m_data_readers),
    //    CEREAL_NVP(m_data_set_processed)*/);
  }

  template<typename T_io_buffer>
  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers) {
    m_io_buffers.push_back(new T_io_buffer(comm, num_parallel_readers, this->m_expected_num_child_layers));
  }

  std::string get_type() const override { return "generic_input"; }

  description get_description() const override {
    auto desc = io_layer<TensorDataType>::get_description();
    desc.add("Buffer", m_io_buffers[0]->get_type());
    return desc;
  }

  void setup_dims() override {
    io_layer<TensorDataType>::setup_dims();
    for (int i = 0; i < this->get_num_children(); ++i) {
      this->set_output_dims(get_data_dims(i), i);
    }
  }

  void setup_data() override {
    io_layer<TensorDataType>::setup_data();

    // Resize output to maximum mini-batch size
    const auto& max_mb_size = this->m_model->get_execution_context().get_trainer().get_max_mini_batch_size();
    for (int i = 0; i < this->get_num_children(); ++i) {
      auto& output = this->get_activations(i);
      output.Resize(output.Height(), max_mb_size);
    }

    /// @todo BVE FIXME
    // if(io_layer<TensorDataType>::m_data_set_spans_models) {
    //calculate_num_iterations_per_epoch_training_spans_models(max_mb_size);
    // } else {
      calculate_num_iterations_per_epoch_training_unique_per_models(max_mb_size);
    // }

    for (auto& io_buffer : m_io_buffers) {
      int linearized_target_size;
      switch(this->m_data_reader_mode) {
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
      io_buffer->setup_data(this->get_output_size(0),
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
    io_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);

    for (auto& io_buffer : m_io_buffers) {
      for (int i = 0; i < this->get_num_children(); ++i) {
        io_buffer->fp_setup_data(mini_batch_size, i);
      }
    }
  }

  void fetch_data_in_background(int future_active_buffer, execution_mode mode) {
    int active_buffer = future_active_buffer % m_io_buffers.size();
    generic_io_buffer<TensorDataType>* io_buffer = m_io_buffers[active_buffer];
    data_coordinator& dc = this->m_model->get_execution_context().get_trainer().get_data_coordinator();
    std::lock_guard<std::mutex> guard(dc.dr_mutex);
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
    execution_mode mode = this->m_model->get_execution_context().get_execution_mode();

    increment_active_buffer_idx(mode);

    generic_io_buffer<TensorDataType>* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];

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
          LBANN_ERROR(err.str());
        }
    }

    if(dynamic_cast<partitioned_io_buffer<TensorDataType>*>(io_buffer) != nullptr) {
      // Use the predetermined size of the mini-batch to set the current
      // batch size for the neural network
      num_samples_in_batch = get_current_mini_batch_size();

      update_num_samples_processed(num_samples_in_batch);
      if(this->m_expected_num_child_layers == 1) {
        io_buffer->distribute_from_local_matrix(get_data_reader(), mode, this->get_activations(0));
      }else {
        io_buffer->distribute_from_local_matrix(get_data_reader(), mode, this->get_activations(0), this->get_activations(1));
      }
    }else {
      LBANN_ERROR("could not fp_compute for I/O layers : encoutered generic_io_buffer type");
    }

    data_coordinator& dc = this->m_model->get_execution_context().get_trainer().get_data_coordinator();
    dc.m_data_set_processed = io_buffer->update_data_set(get_data_reader(mode), mode);

    if(!dc.m_data_set_processed && this->m_model->get_execution_context().background_io_activity_allowed()) {
      int next_active_buffer = get_active_buffer_idx(mode) + 1;
      std::future<void> background_fetch_done = this->m_model->get_execution_context().get_io_thread_pool().submit_job(
        std::bind(&generic_input_layer::fetch_data_in_background, this, next_active_buffer, mode));
      generic_io_buffer<TensorDataType>* next_io_buffer = m_io_buffers[next_active_buffer % m_io_buffers.size()];
      next_io_buffer->set_data_fetch_future(std::move(background_fetch_done), mode);
      next_io_buffer->set_fetch_data_in_background(true, mode);
    }
  }

  void setup_next_io_buffer(generic_io_buffer<TensorDataType>* io_buffer) {
    int mini_batch_size = get_current_mini_batch_size();
    for (int i = 0; i < this->get_num_children(); ++i) {
      io_buffer->fp_setup_data(mini_batch_size, i);
    }
  }

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() override {
    data_coordinator& dc = this->m_model->get_execution_context().get_trainer().get_data_coordinator();
    return dc.m_data_set_processed;
  }

  //************************************************************************
  // Helper functions to access the data readers
  //************************************************************************

  generic_data_reader *get_data_reader(const execution_mode mode) const {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_data_reader(mode);
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
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_dataset(m);
  }

  const dataset& get_dataset(execution_mode m) const override {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_dataset(m);
  }

  /**
   * Return the dataset associated with the current execution mode.
   */
  dataset& select_dataset() override { return get_dataset(this->m_model->get_execution_context().get_execution_mode()); }
  const dataset& select_dataset() const override { return get_dataset(this->m_model->get_execution_context().get_execution_mode()); }

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  dataset* select_first_valid_dataset() override {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().select_first_valid_dataset();
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
    generic_io_buffer<TensorDataType>* io_buffer = m_io_buffers[get_active_buffer_idx(mode) % m_io_buffers.size()];
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
        switch(this->m_data_reader_mode) {
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

    generic_data_reader *dr;

    auto& dc = this->m_model->get_execution_context().get_trainer().get_data_coordinator();
    dr = dc.get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      linearized_data_size = dr->get_linearized_data_size();
    }

    dr = dc.get_data_reader(execution_mode::validation);
    if (dr != nullptr) {
      long tmp_data_size = dr->get_linearized_data_size();
      if (linearized_data_size != -1 && linearized_data_size != tmp_data_size) {
        LBANN_ERROR("lbann_io_layer: validation data set size does not "
                              "match the currently established data set size");
      }
    }

    dr = dc.get_data_reader(execution_mode::testing);
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
  long get_linearized_label_size() const override {
    if (this->is_for_regression()) {
      return static_cast<long>(1);
    }
    long linearized_label_size = -1;
    generic_data_reader *dr;

    auto& dc = this->m_model->get_execution_context().get_trainer().get_data_coordinator();
    dr = dc.get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      linearized_label_size = dr->get_linearized_label_size();
    }
    dr = dc.get_data_reader(execution_mode::validation);
    if (dr != nullptr) {
      long tmp_label_size = dr->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        LBANN_ERROR("lbann_io_layer: validation label set size (" + std::to_string(tmp_label_size) + ") does not match the currently established data set size (" + std::to_string(linearized_label_size) + ")");
      }
    }
    dr = dc.get_data_reader(execution_mode::testing);
    if (dr != nullptr) {
      long tmp_label_size = dr->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        LBANN_ERROR("lbann_io_layer: testing label set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_label_size;
  }

  long get_linearized_response_size() const override {
    if (!this->is_for_regression()) {
      return static_cast<long>(1);
    }
    long linearized_response_size = -1;
    generic_data_reader *dr;

    auto& dc = this->m_model->get_execution_context().get_trainer().get_data_coordinator();
    dr = dc.get_data_reader(execution_mode::training);
    if (dr != nullptr) {
      linearized_response_size = dr->get_linearized_response_size();
    }
    dr = dc.get_data_reader(execution_mode::validation);
    if (dr != nullptr) {
      long tmp_response_size = dr->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        LBANN_ERROR("lbann_io_layer: validation response set size does not "
                              "match the currently established data set size");
      }
    }
    dr = dc.get_data_reader(execution_mode::testing);
    if (dr != nullptr) {
      long tmp_response_size = dr->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        LBANN_ERROR("lbann_io_layer: testing response set size does not "
                              "match the currently established data set size");
      }
    }
    return linearized_response_size;
  }

  long get_num_samples_trained() const override {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_num_samples_trained();
  }
  long get_num_samples_tested() const override {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_num_samples_tested();
  }
  long get_total_num_training_samples() const override {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_total_num_training_samples();
  }
  long get_total_num_testing_samples() const override {
    return this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_total_num_testing_samples();
  }

  bool at_new_epoch() const override {
    const generic_data_reader *dr = this->m_model->get_execution_context().get_trainer().get_data_coordinator().get_data_reader(execution_mode::training);
    return (dr != nullptr && dr->at_new_epoch());
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
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      this->m_model->get_execution_context().get_trainer().get_data_coordinator().save_to_checkpoint_shared(p);

      if (this->get_comm()->am_trainer_master()) {
        write_cereal_archive<const generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
      }

    }
    return true;
  }

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) override {
    // save state of the input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      std::string buf;
      if (this->get_comm()->am_trainer_master()) {
        read_cereal_archive<generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
        buf = create_cereal_archive_binary_string<generic_input_layer>(*this);
      }

      this->m_model->get_execution_context().get_trainer().get_data_coordinator().load_from_checkpoint_shared(p);

      // TODO: this assumes homogeneous processors
      // broadcast state from rank 0
      this->get_comm()->trainer_broadcast(0, buf);

      if (!this->get_comm()->am_trainer_master()) {
        unpack_cereal_archive_binary_string<generic_input_layer>(*this, buf);
      }

    }
    return true;
  }

  bool save_to_checkpoint_distributed(persist& p) const override {
    // save state of data readers from input layer
    if(p.get_cb_type() == callback_type::execution_context_only || p.get_cb_type() == callback_type::full_checkpoint) {
      this->m_model->get_execution_context().get_trainer().get_data_coordinator().save_to_checkpoint_distributed(p);

      write_cereal_archive<const generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
    }
    return true;
  }

  bool load_from_checkpoint_distributed(persist& p) override {
    // load state of data readers for input layer

    this->m_model->get_execution_context().get_trainer().get_data_coordinator().load_from_checkpoint_distributed(p);

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
  std::vector<generic_io_buffer<TensorDataType>*> m_io_buffers;
  io_buffer_map_t m_active_buffer;
};

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
