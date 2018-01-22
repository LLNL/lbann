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

#ifndef LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED

#include "lbann/layers/io/io_layer.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/io/data_buffers/distributed_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
class generic_target_layer : public io_layer {
 protected:
  generic_input_layer *m_paired_input_layer;
  generic_io_buffer *io_buffer;

 public:
  generic_target_layer(lbann_comm *comm, generic_input_layer* input_layer, std::map<execution_mode, generic_data_reader *> data_readers, bool for_regression = false)
    : io_layer(comm, true, for_regression), m_paired_input_layer(input_layer), io_buffer(nullptr)  {
    // Target layers have no children
    m_max_num_child_layers = 0;
  }

  ~generic_target_layer() override {
    if(io_buffer != nullptr) {
      delete io_buffer;
    }
  };

  generic_target_layer(const generic_target_layer& other) = default;

  generic_target_layer& operator=(const generic_target_layer& other) = default;

  template<typename T_io_buffer>
  inline void initialize_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    io_layer::initialize_distributed_matrices<T_layout>();
  }

  generic_input_layer* get_paired_input_layer() const {
    return m_paired_input_layer;
  }

  void set_paired_input_layer(generic_input_layer *input_layer) {
    m_paired_input_layer = input_layer;
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::string s = get_topo_description();
    return std::string {} + " target_layer " + io_buffer->get_type()
           + " dataLayout: " + this->get_data_layout_string(get_data_layout())
           + " (" + s + ")";
  }

  void setup_dims() override {
    io_layer::setup_dims();
    if (this->is_for_regression()) {
      this->m_neuron_dims = get_data_dims();
      this->m_num_neuron_dims = this->m_neuron_dims.size();
      this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                            this->m_neuron_dims.end(),
                                            1,
                                            std::multiplies<int>());
    } else {
      this->m_num_neurons = get_linearized_label_size();
      this->m_num_neuron_dims = 1;
      this->m_neuron_dims.assign(1, this->m_num_neurons);
    }
  }

  void setup_data() override {
    io_layer::setup_data();
    int max_mb_size = this->m_model->get_max_mini_batch_size();
    if(io_buffer != nullptr) {  /// Note that reconstruction layers do not have io_buffers
      io_buffer->setup_data(this->m_num_neurons, max_mb_size);
    }
  }

  void check_setup() override {
    io_layer::check_setup();
    if(this->m_num_prev_neurons != this->m_num_neurons) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__
          << "input and output dimensions do not match "
          << "(" << this->m_num_prev_neurons << " input neurons, "
          << this->m_num_neurons << " output neurons)";
      throw lbann_exception(err.str());
    }
  }

  void fp_set_std_matrix_view() override {
    io_layer::fp_set_std_matrix_view();
    if(io_buffer != nullptr) {  /// Note that reconstruction layers do not have io_buffers
      El::Int cur_mini_batch_size = m_model->get_current_mini_batch_size();
      io_buffer->set_local_matrix_bypass(&this->m_activations_v->Matrix());
      io_buffer->set_std_matrix_view(cur_mini_batch_size);
    }
  }

  void fp_compute() override {
    execution_mode mode = this->m_model->get_execution_mode();
    int num_samples_in_batch = io_buffer->fetch_to_local_matrix(m_paired_input_layer->get_data_reader(), mode);

    if(dynamic_cast<partitioned_io_buffer*>(io_buffer) != nullptr) {
      update_num_samples_processed(num_samples_in_batch);
    }else if(dynamic_cast<distributed_io_buffer*>(io_buffer) != nullptr) {
      if(((distributed_io_buffer*) io_buffer)->is_current_root(mode)) {
        /// Only update the number of samples processed by this parallel reader, when it is the current root
        update_num_samples_processed(num_samples_in_batch);
      }

      int curr_mini_batch_size = this->m_model->get_current_mini_batch_size();
      if(((distributed_io_buffer*) io_buffer)->is_current_root(mode) && num_samples_in_batch != curr_mini_batch_size) {
        throw lbann_exception("lbann_target_layer_distributed_minibatch: number of labels ("
                              + std::to_string(num_samples_in_batch) + ") does not match the current mini-batch size ("
                              + std::to_string(curr_mini_batch_size) + ")."
                              );
      }
      /// @todo should this distribute the entire matrix even if there is only a partial mini-batch
      io_buffer->distribute_from_local_matrix(*this->m_activations, m_paired_input_layer->get_data_reader(), mode);
    }else {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "could not fp_compute for I/O layers : encoutered generic_io_buffer type";
      throw lbann_exception(err.str());
    }
  }

  void bp_compute() override {}

  bool update_compute() override {
    return io_buffer->is_data_set_processed(m_paired_input_layer->get_data_reader(), this->m_model->get_execution_mode());
  }
  // lbann::generic_data_reader *set_training_data_reader(generic_data_reader *data_reader, bool shared_data_reader) {
  //   return io_layer::set_training_data_reader(data_reader);
  // }

  // lbann::generic_data_reader *set_testing_data_reader(generic_data_reader *data_reader, bool shared_data_reader) {
  //   return io_layer::set_testing_data_reader(data_reader);
  // }

  //************************************************************************
  // Helper functions to access the data readers
  //************************************************************************
  dataset& get_dataset(execution_mode m) override {
    return m_paired_input_layer->get_dataset(m);
  }

  const dataset& get_dataset(execution_mode m) const override {
    return m_paired_input_layer->get_dataset(m);
  }

  /**
   * Return the dataset associated with the current execution mode.
   */
  dataset& select_dataset() override { return m_paired_input_layer->select_dataset(); }
  const dataset& select_dataset() const override { return m_paired_input_layer->select_dataset(); }

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  dataset* select_first_valid_dataset() override {
    return m_paired_input_layer->select_first_valid_dataset();
  }

  /**
   * Return the data reader associated with the current execution mode.
   */
  generic_data_reader *select_data_reader() const override {
    return m_paired_input_layer->select_data_reader();
  }

  /**
   * Update the number of samples processed for the current execution mode.
   */
  long update_num_samples_processed(long num_samples) override {
    return m_paired_input_layer->update_num_samples_processed(num_samples);
  }

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
  El::Matrix<El::Int>* get_sample_indices_per_mb() override {
    return m_paired_input_layer->get_sample_indices_per_mb();
  }

  /**
   * Get the dimensions of the underlying data.
   */
  const std::vector<int> get_data_dims() const override {
    return m_paired_input_layer->get_data_dims();
  }

  std::string get_topo_description() const override {
    return m_paired_input_layer->get_topo_description();
  }

  /**
   * Get the linearized size of the underlying data.
   */
  long get_linearized_data_size() const override {
    return m_paired_input_layer->get_linearized_data_size();
  }

  /**
   * Get the linearized size of the labels for the underlying data.
   */
  long get_linearized_label_size() const override {
    return m_paired_input_layer->get_linearized_label_size();
  }

  long get_linearized_response_size() const override {
    return m_paired_input_layer->get_linearized_response_size();
  }

  long get_num_samples_trained() const override {
    return m_paired_input_layer->get_num_samples_trained();
  }
  long get_num_samples_tested() const override {
    return m_paired_input_layer->get_num_samples_tested();
  }
  long get_total_num_training_samples() const override {
    return m_paired_input_layer->get_total_num_training_samples();
  }
  long get_total_num_testing_samples() const override {
    return m_paired_input_layer->get_total_num_testing_samples();
  }

  bool at_new_epoch() const override {
    return m_paired_input_layer->at_new_epoch();
  }

  bool is_execution_mode_valid(execution_mode mode) const override {
    return m_paired_input_layer->is_execution_mode_valid(mode);
  }

  AbsDistMat& get_prediction() { return *this->m_prev_activations_v; }
  AbsDistMat& get_ground_truth() { return *this->m_activations_v; }
  const AbsDistMat& get_prediction() const { return *this->m_prev_activations_v; }
  const AbsDistMat& get_ground_truth() const { return *this->m_activations_v; }

  std::vector<Layer*> get_layer_pointers() override {
    std::vector<Layer*> layers = io_layer::get_layer_pointers();
    layers.push_back((Layer*) m_paired_input_layer);
    return layers;
  }

  void set_layer_pointers(std::vector<Layer*> layers) override {
    m_paired_input_layer = dynamic_cast<generic_input_layer*>(layers.back());
    if (m_paired_input_layer == nullptr) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: lbann_target_layer: invalid layer pointer used to set paired input layer";
      throw lbann_exception(err.str());
    }
    layers.pop_back();
    io_layer::set_layer_pointers(layers);
  }

  //************************************************************************
  //
  //************************************************************************

  bool saveToCheckpoint(int fd, const char *filename, size_t *bytes) const override {
    /// @todo should probably save m_shared_data_reader
    return Layer::saveToCheckpoint(fd, filename, bytes);
  }

  bool loadFromCheckpoint(int fd, const char *filename, size_t *bytes) override {
    /// @todo should probably save m_shared_data_reader
    return Layer::loadFromCheckpoint(fd, filename, bytes);
  }

  bool save_to_checkpoint_shared(persist& p) const override {
    // rank 0 writes softmax cost to file
    if (p.get_rank() == 0) {
      // p.write_double(persist_type::train, "aggregate cost", (double) aggregate_cost);
      // p.write_uint64(persist_type::train, "num backprop steps", (uint64_t) num_backprop_steps);
    }

    return Layer::save_to_checkpoint_shared(p);
  }

  bool load_from_checkpoint_shared(persist& p) override {
    // rank 0 writes softmax cost to file
    // if (p.get_rank() == 0) {
    //     double dval;
    //     p.read_double(persist_type::train, "aggregate cost", &dval);
    //     aggregate_cost = (DataType) dval;

    //     uint64_t val;
    //     p.read_uint64(persist_type::train, "num backprop steps", &val);
    //     num_backprop_steps = (long) val;
    // }

    // // get values from rank 0
    // MPI_Bcast(&aggregate_cost, 1, DataTypeMPI, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&num_backprop_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    return Layer::load_from_checkpoint_shared(p);
    //return true;
  }
};

template<> inline void generic_target_layer::initialize_io_buffer<partitioned_io_buffer>(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
  io_buffer = new partitioned_io_buffer(comm, num_parallel_readers, data_readers);
}

template<> inline void generic_target_layer::initialize_io_buffer<distributed_io_buffer>(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers) {
  io_buffer = new distributed_io_buffer(comm, num_parallel_readers, data_readers);
}

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED
