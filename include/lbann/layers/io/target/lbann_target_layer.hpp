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

#ifndef LBANN_LAYERS_TARGET_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_TARGET_LAYER_HPP_INCLUDED

#include "lbann/layers/io/lbann_io_layer.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
template <class T_layout>
class target_layer : public io_layer<T_layout> {
 protected:
  bool m_shared_data_reader;

 public:
  target_layer(data_layout data_dist, lbann_comm *comm, uint mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers, bool shared_data_reader, bool for_regression = false)
    : io_layer<T_layout>(data_dist, comm, mini_batch_size, data_readers, true, for_regression) {
    if (this->is_for_regression()) {
      this->m_num_neurons = io_layer<T_layout>::get_linearized_response_size();
    } else {
      this->m_num_neurons = io_layer<T_layout>::get_linearized_label_size();
    }
    m_shared_data_reader = shared_data_reader;
  }

  virtual void initialize_model_parallel_distribution() {
    target_layer<T_layout>::initialize_model_parallel_distribution();
  }
  virtual void initialize_data_parallel_distribution() {
    target_layer<T_layout>::initialize_data_parallel_distribution();
  }

  void setup(int num_prev_neurons) {
    if(this->m_neural_network_model->m_obj_fn == NULL) {
      throw lbann_exception("target layer has invalid objective function pointer");
    }
    this->m_neural_network_model->m_obj_fn->setup(this->m_num_neurons, this->m_mini_batch_size);
    for (auto&& m : this->m_neural_network_model->m_metrics) {
      m->setup(this->m_num_neurons, this->m_mini_batch_size);
      m->m_neural_network_model = this->m_neural_network_model;
    }
    Zeros(*this->m_activations, this->m_num_neurons, this->m_mini_batch_size);
    Zeros(*this->m_weighted_sum, this->m_num_neurons, this->m_mini_batch_size);
  }

  /**
   * Target layers are not able to return target matrices for forward propagation
   */
  DistMat *fp_output() {
    return NULL;
  }

  lbann::generic_data_reader *set_training_data_reader(generic_data_reader *data_reader, bool shared_data_reader) {
    m_shared_data_reader = shared_data_reader;
    return io_layer<T_layout>::set_training_data_reader(data_reader);
  }

  lbann::generic_data_reader *set_testing_data_reader(generic_data_reader *data_reader, bool shared_data_reader) {
    m_shared_data_reader = shared_data_reader;
    return io_layer<T_layout>::set_testing_data_reader(data_reader);
  }

  void fp_set_std_matrix_view() {
    int64_t cur_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();
    Layer::fp_set_std_matrix_view();
    this->m_neural_network_model->m_obj_fn->fp_set_std_matrix_view(cur_mini_batch_size);
    for (auto&& m : this->m_neural_network_model->m_metrics) {
      m->fp_set_std_matrix_view(cur_mini_batch_size);
    }
  }

  void summarize(lbann_summary& summarizer, int64_t step) {
    Layer::summarize(summarizer, step);
    std::string tag = "layer" + std::to_string(static_cast<long long>(this->m_index))
      + "/CrossEntropyCost";
    summarizer.reduce_scalar(tag, this->m_neural_network_model->m_obj_fn->report_aggregate_avg_obj_fn(execution_mode::training), step);
  }

  void epoch_print() const {
    double obj_cost = this->m_neural_network_model->m_obj_fn->report_aggregate_avg_obj_fn(execution_mode::training);
    if (this->m_comm->am_world_master()) {
      std::vector<double> avg_obj_fn_costs(this->m_comm->get_num_models());
      this->m_comm->intermodel_gather(obj_cost, avg_obj_fn_costs);
      for (size_t i = 0; i < avg_obj_fn_costs.size(); ++i) {
        std::cout << "Model " << i << " average " << _to_string(this->m_neural_network_model->m_obj_fn->type) << ": " << avg_obj_fn_costs[i] <<
          std::endl;
      }
    } else {
      this->m_comm->intermodel_gather(obj_cost, this->m_comm->get_world_master());
    }
  }

  void epoch_reset() {
    Layer::epoch_reset();
    resetCost();
  }

  void resetCost() {
    this->m_neural_network_model->m_obj_fn->reset_obj_fn();
  }

  bool saveToCheckpoint(int fd, const char *filename, uint64_t *bytes) {
    /// @todo should probably save m_shared_data_reader
    return Layer::saveToCheckpoint(fd, filename, bytes);
  }

  bool loadFromCheckpoint(int fd, const char *filename, uint64_t *bytes) {
    /// @todo should probably save m_shared_data_reader
    return Layer::loadFromCheckpoint(fd, filename, bytes);
  }

  bool saveToCheckpointShared(persist& p) {
    // rank 0 writes softmax cost to file
    if (p.get_rank() == 0) {
      // p.write_double(persist_type::train, "aggregate cost", (double) aggregate_cost);
      // p.write_uint64(persist_type::train, "num backprop steps", (uint64_t) num_backprop_steps);
    }

    return true;
  }

  bool loadFromCheckpointShared(persist& p) {
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

    //return Layer::loadFromCheckpointShared(dir, bytes);
    return true;
  }
};
}

#endif  // LBANN_LAYERS_TARGET_LAYER_HPP_INCLUDED
