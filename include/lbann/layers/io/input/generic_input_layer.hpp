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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/io_layer.hpp"
//#include "lbann/utils/dataset.hpp"
#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/io/data_buffers/distributed_io_buffer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {
class generic_input_layer : public io_layer {
 public:
  using data_reader_map_t = std::map<execution_mode, generic_data_reader *>;
  generic_io_buffer *io_buffer;

 public:
  generic_input_layer(lbann_comm *comm,
              int num_parallel_readers,
              std::map<execution_mode, generic_data_reader *> data_readers,
              bool data_set_spans_models = true, bool for_regression = false)
    : io_layer(comm, data_set_spans_models, for_regression),
      io_buffer(nullptr),
      m_training_dataset(),
      m_testing_dataset(),
      m_validation_dataset(),
      m_data_readers(data_readers) {
      //m_data_sets_span_models(data_sets_span_models) {
    // Input layers have no parents
    m_expected_num_parent_layers = 0;

    if(m_data_readers[execution_mode::training] != nullptr) {
      m_training_dataset.total_samples() = m_data_readers[execution_mode::training]->get_num_data();
    }

    if(m_data_readers[execution_mode::validation] != nullptr) {
      m_validation_dataset.total_samples() = m_data_readers[execution_mode::validation]->get_num_data();
    }

    if(m_data_readers[execution_mode::testing] != nullptr) {
      m_testing_dataset.total_samples() = m_data_readers[execution_mode::testing]->get_num_data();
    }
  }

  ~generic_input_layer() override {
    if(io_buffer != nullptr) {
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
      io_buffer(other.io_buffer),
      m_training_dataset(other.m_training_dataset),
      m_testing_dataset(other.m_testing_dataset),
      m_validation_dataset(other.m_validation_dataset),
      m_data_readers(other.m_data_readers) {
    for (auto& dr : m_data_readers) {
      dr.second = dr.second->copy();
    }
  }

  generic_input_layer& operator=(const generic_input_layer& other) {
    io_layer::operator=(other);
    io_buffer = other.io_buffer->copy();
    for (auto& dr : m_data_readers) {
      dr.second = dr.second->copy();
    }
    return *this;
  }

  template<typename T_io_buffer>
  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);

  std::string get_type() const override { return "generic_input"; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::string s = get_topo_description();
    return std::string {} + " input_layer " + io_buffer->get_type()
           + " dataLayout: " + this->get_data_layout_string(get_data_layout())
           + " (" + s + ")";
  }

  //data_layout get_data_layout() const override { return T_layout; }
  // std::string get_description() const {
  //   std::stringstream s;
  //   for (size_t i = 0; i < this->m_neuron_dims.size(); i++) {
  //     s << this->m_neuron_dims[i];
  //     if ( i != this->m_neuron_dims.size()-1) {
  //       s << " x ";
  //     }
  //   }
  //   return s.str();;
  // }

  void setup_dims() override {
    io_layer::setup_dims();
    this->m_neuron_dims = get_data_dims();
    this->m_num_neuron_dims = this->m_neuron_dims.size();
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  void setup_data() override {
    io_layer::setup_data();

    /// BVE FIXME foreach data reader
    // in case that target_layer gets initialized beforehand
    if(m_data_readers[execution_mode::training] != nullptr) {
      m_data_readers[execution_mode::training]->setup();
      m_data_readers[execution_mode::training]->set_rank(Layer::m_comm->get_rank_in_model());
    }
    if(m_data_readers[execution_mode::validation] != nullptr) {
      m_data_readers[execution_mode::validation]->setup();
      m_data_readers[execution_mode::validation]->set_rank(Layer::m_comm->get_rank_in_model());
    }
    if(m_data_readers[execution_mode::testing] != nullptr) {
      m_data_readers[execution_mode::testing]->setup();
      m_data_readers[execution_mode::testing]->set_rank(Layer::m_comm->get_rank_in_model());
    }

    int max_mb_size = this->m_model->get_max_mini_batch_size();
    if(io_layer::m_data_set_spans_models) {
      calculate_num_iterations_per_epoch_training_spans_models(max_mb_size);
    } else {
      calculate_num_iterations_per_epoch_training_unique_per_models(max_mb_size);
    }

    io_buffer->setup_data(this->m_num_neurons, max_mb_size);
  }

  /** Define the standard view of the matrix -- and set it for the model
   * Setup the effective (global) mini-batch size so that gradients are properly
   * averaged across models. */
  void fp_setup_data(int mini_batch_size) override {

    // Use the predetermined size of the mini-batch to set the current
    // batch size for the neural network
    mini_batch_size = get_current_mini_batch_size();
    this->m_model->set_current_mini_batch_size(mini_batch_size);

    // Use the precomputed size of the global mini-batch to set the
    // current effective batch size across all models
    int total_mini_batch_size = get_current_global_mini_batch_size();
    this->m_model->set_effective_mini_batch_size(total_mini_batch_size);

    // Initialize matrices
    io_layer::fp_setup_data(mini_batch_size);

    // Once the current mini-batch size is defined, set the standard view for activations only
    io_buffer->set_local_matrix_bypass(static_cast<CPUMat*>(&get_local_activations()));
    io_buffer->set_std_matrix_view(mini_batch_size);

  }

  void fp_compute() override {
    execution_mode mode = this->m_model->get_execution_mode();
    int num_samples_in_batch = io_buffer->fetch_to_local_matrix(get_data_reader(), mode);

    if(dynamic_cast<partitioned_io_buffer*>(io_buffer) != nullptr) {
      // Use the predetermined size of the mini-batch to set the current
      // batch size for the neural network
      num_samples_in_batch = get_current_mini_batch_size();

      update_num_samples_processed(num_samples_in_batch);
    }else if(dynamic_cast<distributed_io_buffer*>(io_buffer) != nullptr) {
      if(((distributed_io_buffer*) io_buffer)->is_current_root(mode)) {
        /// Only update the number of samples processed by this parallel reader, when it is the current root
        update_num_samples_processed(num_samples_in_batch);
      }

      int expected_num_samples_in_batch = this->m_model->get_current_mini_batch_size();

      /// Let each rank know this size of the current mini-batch
      /// Note that this field has to be updated before distributing the data
      num_samples_in_batch = Layer::m_comm->model_broadcast(((distributed_io_buffer*) io_buffer)->current_root_rank(mode), num_samples_in_batch);
      this->m_model->set_current_mini_batch_size(num_samples_in_batch
                                                 + get_current_world_master_mini_batch_adjustment(m_comm->get_model_rank()));

      io_buffer->distribute_from_local_matrix(get_activations(), get_data_reader(), mode);

      if(num_samples_in_batch !=
         (expected_num_samples_in_batch - get_current_world_master_mini_batch_adjustment(m_comm->get_model_rank()))) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "I/O layers number of samples processed ("<< num_samples_in_batch
            <<") does not match the mini-batch size ("
            << (expected_num_samples_in_batch -
                get_current_world_master_mini_batch_adjustment(m_comm->get_model_rank()))
            << ")";
        throw lbann_exception(err.str());
      }
    }else {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "could not fp_compute for I/O layers : encoutered generic_io_buffer type";
      throw lbann_exception(err.str());
    }
  }

  void bp_compute() override {}

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() override {
    return io_buffer->is_data_set_processed(get_data_reader(), this->m_model->get_execution_mode());
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
      throw lbann_exception(
                            std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                            " :: generic data distribution: invalid execution phase");
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

    /// Setup the training data set so that it spans all models
    io_buffer->calculate_num_iterations_per_epoch_spanning_models(mini_batch_size,
                                                                  get_data_reader(execution_mode::training));

    /// Each model uses the entire validation and testing data sets
    io_buffer->calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                               get_data_reader(execution_mode::validation));
    io_buffer->calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                               get_data_reader(execution_mode::testing));

  }

  void calculate_num_iterations_per_epoch_training_unique_per_models(int mini_batch_size) {

    /// Setup the training data set so that it spans all models
    io_buffer->calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                               get_data_reader(execution_mode::training));

    /// Each model uses the entire validation and testing data sets
    io_buffer->calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                               get_data_reader(execution_mode::validation));
    io_buffer->calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                               get_data_reader(execution_mode::testing));

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
      throw lbann_exception("get_dataset: invalid execution mode");
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
      throw lbann_exception("get_dataset: invalid execution mode");
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
    generic_data_reader *dr = get_data_reader();
    return dr->get_indices_fetched_per_mb();
  }

  /**
   * Get the dimensions of the underlying data.
   */
  const std::vector<int> get_data_dims() const override {
    const generic_data_reader *dr = get_data_reader();
    //    dataset* ds = select_first_valid_dataset();
    if (dr) {
      return dr->get_data_dims();
    }
    return std::vector<int>(1, 0);
  }

  std::string get_topo_description() const override {
    std::stringstream s;
    for (size_t i = 0; i < this->m_neuron_dims.size(); i++) {
      s << this->m_neuron_dims[i];
      if ( i != this->m_neuron_dims.size()-1) {
        s << " x ";
      }
    }
    return s.str();;
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
        throw lbann_exception("lbann_io_layer: validation data set size does not "
                              "match the currently established data set size");
      }
    }

    it = m_data_readers.find(execution_mode::testing);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_data_size = (it->second)->get_linearized_data_size();
      if (linearized_data_size != -1 && linearized_data_size != tmp_data_size) {
        throw lbann_exception("lbann_io_layer: testing data set size does not "
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
        throw lbann_exception("lbann_io_layer: validation label set size does not "
                              "match the currently established data set size");
      }
    }
    it = m_data_readers.find(execution_mode::testing);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_label_size = (it->second)->get_linearized_label_size();
      if (linearized_label_size != -1 && linearized_label_size != tmp_label_size) {
        throw lbann_exception("lbann_io_layer: testing label set size does not "
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
        throw lbann_exception("lbann_io_layer: validation response set size does not "
                              "match the currently established data set size");
      }
    }
    it = m_data_readers.find(execution_mode::testing);
    if ((it != m_data_readers.end()) && it->second) {
      long tmp_response_size = (it->second)->get_linearized_response_size();
      if (linearized_response_size != -1 && linearized_response_size != tmp_response_size) {
        throw lbann_exception("lbann_io_layer: testing response set size does not "
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
  bool save_to_checkpoint_shared(persist& p, bool val_end) const override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(!val_end){
      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, "data_reader_training");
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, "data_reader_testing");
      }
      if (p.get_rank() == 0) {
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
    else if(val_end){
      p.write_uint64(persist_type::validate, "reader_validate_processed",
                     (uint64_t) m_validation_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::validate, "reader_validate_total",
                     (uint64_t) m_validation_dataset.get_total_samples());

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
    if (p.get_rank() == 0) {
      p.read_uint64(persist_type::train, "reader_train_processed",    &header.train_proc);
      p.read_uint64(persist_type::train, "reader_train_total",        &header.train_total);
      p.read_uint64(persist_type::train, "reader_test_processed",     &header.test_proc);
      p.read_uint64(persist_type::train, "reader_test_total",         &header.test_total);
      p.read_uint64(persist_type::validate, "reader_validate_processed", &header.validate_proc);
      p.read_uint64(persist_type::validate, "reader_validate_total",     &header.validate_total);
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
    m_validation_dataset.num_samples_processed() = (long) header.validate_proc;
    m_validation_dataset.total_samples()         = (long) header.validate_total;
    return true;
  }

 protected:
  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;
  //  bool m_data_sets_span_models;

  data_reader_map_t m_data_readers;
 //  std::map<execution_mode, dataset_stats> m_dataset_stats;
};

template<> inline void generic_input_layer::initialize_io_buffer<partitioned_io_buffer>(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
  io_buffer = new partitioned_io_buffer(comm, num_parallel_readers, data_readers);
}

template<> inline void generic_input_layer::initialize_io_buffer<distributed_io_buffer>(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
  io_buffer = new distributed_io_buffer(comm, num_parallel_readers, data_readers);
}

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
