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

#ifndef LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/io_layer.hpp"
//#include "lbann/utils/dataset.hpp"
#include "lbann/data_distributions/data_distribution.hpp"
#include "lbann/models/model.hpp"

namespace lbann {
class input_layer : public io_layer, public virtual generic_data_distribution {
 public:
  typedef std::map<execution_mode, generic_data_reader *> data_reader_map_t;

 public:
  input_layer(lbann_comm *comm,
              int num_parallel_readers,
              std::map<execution_mode, generic_data_reader *> data_readers,
              bool data_set_spans_models = true)
    : generic_data_distribution(comm, num_parallel_readers, data_readers),
      io_layer(comm, data_set_spans_models),
      m_training_dataset(),
      m_testing_dataset(),
      m_validation_dataset(),
      m_data_readers(data_readers) {
      //m_data_sets_span_models(data_sets_span_models) {
    // Input layers have no parents
    m_max_num_parent_layers = 0;

    generic_data_distribution::fetch_data_fn = new fetch_data_functor(true, false);
    generic_data_distribution::update_data_reader_fn = new update_data_reader_functor(true);

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

  ~input_layer() override {
    // Input layer always frees data readers.
    for (auto& dr : m_data_readers) {
      delete dr.second;
    }
  }

  // Input layers copy their datareaders.
  input_layer(const input_layer& other)
    : generic_data_distribution(other), io_layer(other),
      m_training_dataset(other.m_training_dataset),
      m_testing_dataset(other.m_testing_dataset),
      m_validation_dataset(other.m_validation_dataset),
      m_data_readers(other.m_data_readers) {
    for (auto& dr : m_data_readers) {
      dr.second = dr.second->copy();
    }
  }

  input_layer& operator=(const input_layer& other) {
    generic_data_distribution::operator=(other);
    io_layer::operator=(other);
    for (auto& dr : m_data_readers) {
      dr.second = dr.second->copy();
    }
    return *this;
  }
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
  }

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    io_layer::initialize_distributed_matrices<T_layout>();
  }

  /** Define the standard view of the matrix -- and set it for the model
   * Setup the effective (global) mini-batch size so that gradients are properly
   * averaged across models. */
  void fp_set_std_matrix_view() override {
    // Use the predetermined size of the mini-batch to set the current
    // batch size for the neural network
    El::Int cur_mini_batch_size = get_current_mini_batch_size();
    this->m_model->set_current_mini_batch_size(cur_mini_batch_size);

    // Use the precomputed size of the global mini-batch to set the
    // current effective batch size across all models
    int total_mini_batch_size = get_current_global_mini_batch_size();
    this->m_model->set_effective_mini_batch_size(total_mini_batch_size);

    // Once the current mini-batch size is defined, set the standard view for activations only
    El::View(*m_activations_v, *m_activations, El::ALL, El::IR(0, cur_mini_batch_size));
  }

  /** No setting the standard view of the matrix -- it defines the standard view */
  void bp_set_std_matrix_view() override {}

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
    return data_reader->get_num_parallel_readers();
  }

  virtual int get_num_parallel_readers() const {
    return get_num_parallel_readers(this->m_model->get_execution_mode());
  }

  virtual int get_num_iterations_per_epoch(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_num_iterations_per_epoch();
  }

  virtual int get_num_iterations_per_epoch() const {
    return get_num_iterations_per_epoch(this->m_model->get_execution_mode());
  }

  virtual int get_current_step_in_epoch(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_current_step_in_epoch();
  }

  virtual int get_current_step_in_epoch() const {
    return get_current_step_in_epoch(this->m_model->get_execution_mode());
  }

  virtual int get_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_mini_batch_size();
  }

  virtual int get_last_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_last_mini_batch_size();
  }

  virtual int get_last_mini_batch_size() const {
    return get_last_mini_batch_size(this->m_model->get_execution_mode());
  }

  virtual int get_current_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_current_mini_batch_size();
  }

  virtual int get_current_mini_batch_size() const {
    return get_current_mini_batch_size(this->m_model->get_execution_mode());
  }

  virtual int get_global_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_global_mini_batch_size();
  }

  virtual int get_global_last_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_global_last_mini_batch_size();
  }

  virtual int get_current_global_mini_batch_size(execution_mode mode) const {
    const generic_data_reader *data_reader = get_data_reader(mode);
    return data_reader->get_current_global_mini_batch_size();
  }

  virtual int get_current_global_mini_batch_size() const {
    return get_current_global_mini_batch_size(this->m_model->get_execution_mode());
  }

  /** Calculate how many iterations are required for training, testing,
   *  and validation given a specified mini-batch size and that the
   *  training data set is spanning all of the models.
   */
  void calculate_num_iterations_per_epoch_training_spans_models(int mini_batch_size) {

    /// Setup the training data set so that it spans all models
    calculate_num_iterations_per_epoch_spanning_models(mini_batch_size,
                                                       get_data_reader(execution_mode::training));

    /// Each model uses the entire validation and testing data sets
    calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                    get_data_reader(execution_mode::validation));
    calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                    get_data_reader(execution_mode::testing));

  }

  void calculate_num_iterations_per_epoch_training_unique_per_models(int mini_batch_size) {

    /// Setup the training data set so that it spans all models
    calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                    get_data_reader(execution_mode::training));

    /// Each model uses the entire validation and testing data sets
    calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                    get_data_reader(execution_mode::validation));
    calculate_num_iterations_per_epoch_single_model(mini_batch_size,
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
    return static_cast<long>(1);
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
  bool saveToCheckpointShared(persist& p) const override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->saveToCheckpointShared(p, "data_reader_training");
    }

    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->saveToCheckpointShared(p, "data_reader_validation");
    }

    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->saveToCheckpointShared(p, "data_reader_testing");
    }

    // save our own state
    // rank 0 writes the file
    if (p.get_rank() == 0) {
      p.write_uint64(persist_type::train, "reader_train_processed",
                     (uint64_t) m_training_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::train, "reader_train_total",
                     (uint64_t) m_training_dataset.get_total_samples());

      p.write_uint64(persist_type::train, "reader_test_processed",
                     (uint64_t) m_testing_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::train, "reader_test_total",
                     (uint64_t) m_testing_dataset.get_total_samples());

      p.write_uint64(persist_type::train, "reader_validate_processed",
                     (uint64_t) m_validation_dataset.get_num_samples_processed());
      p.write_uint64(persist_type::train, "reader_validate_total",
                     (uint64_t) m_validation_dataset.get_total_samples());
    }
    io_layer::saveToCheckpointShared(p);

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
  bool loadFromCheckpointShared(persist& p) override {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->loadFromCheckpointShared(p, "data_reader_training");
    }

    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->loadFromCheckpointShared(p, "data_reader_validation");
    }

    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->loadFromCheckpointShared(p, "data_reader_testing");
    }

    // save our own state
    // rank 0 reads the file
    dataset_header header;
    if (p.get_rank() == 0) {
      p.read_uint64(persist_type::train, "reader_train_processed",    &header.train_proc);
      p.read_uint64(persist_type::train, "reader_train_total",        &header.train_total);
      p.read_uint64(persist_type::train, "reader_test_processed",     &header.test_proc);
      p.read_uint64(persist_type::train, "reader_test_total",         &header.test_total);
      p.read_uint64(persist_type::train, "reader_validate_processed", &header.validate_proc);
      p.read_uint64(persist_type::train, "reader_validate_total",     &header.validate_total);
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

    io_layer::loadFromCheckpointShared(p);

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

}  // namespace lbann

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
