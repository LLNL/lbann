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

#include "lbann/layers/lbann_target_layer.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::target_layer::target_layer(data_layout data_dist, lbann_comm* comm, uint mini_batch_size, std::map<execution_mode, generic_data_reader*> data_readers, bool shared_data_reader, bool for_regression)
  : io_layer(data_dist, comm, mini_batch_size, data_readers, std::vector<lbann::regularizer*>(), true, for_regression)
{
  if (is_for_regression())
    NumNeurons = io_layer::get_linearized_response_size();
  else
    NumNeurons = io_layer::get_linearized_label_size();
  m_shared_data_reader = shared_data_reader;
}

void lbann::target_layer::setup(int num_prev_neurons) {
  if(neural_network_model->obj_fn == NULL) {
    throw lbann_exception("target layer has invalid objective function pointer");
  }
  neural_network_model->obj_fn->setup(NumNeurons, m_mini_batch_size);
  for (auto&& m : neural_network_model->metrics) {
    m->setup(NumNeurons, m_mini_batch_size);
    m->neural_network_model = neural_network_model;
  }
  Zeros(*m_activations, NumNeurons, m_mini_batch_size);
  Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
}

/**
 * Target layers are not able to return target matrices for forward propagation
 */
DistMat *lbann::target_layer::fp_output() {
  return NULL;
}

lbann::generic_data_reader *lbann::target_layer::set_training_data_reader(generic_data_reader *data_reader, bool shared_data_reader) {
  m_shared_data_reader = shared_data_reader;
  return io_layer::set_training_data_reader(data_reader);
}

lbann::generic_data_reader *lbann::target_layer::set_testing_data_reader(generic_data_reader *data_reader, bool shared_data_reader) {
  m_shared_data_reader = shared_data_reader;
  return io_layer::set_testing_data_reader(data_reader);
}

void lbann::target_layer::fp_set_std_matrix_view() {
  int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  Layer::fp_set_std_matrix_view();
  neural_network_model->obj_fn->fp_set_std_matrix_view(cur_mini_batch_size);
  for (auto&& m : neural_network_model->metrics) {
    m->fp_set_std_matrix_view(cur_mini_batch_size);
  }
}

void lbann::target_layer::summarize(lbann_summary& summarizer, int64_t step) {
  Layer::summarize(summarizer, step);
  std::string tag = "layer" + std::to_string(static_cast<long long>(Index))
    + "/CrossEntropyCost";
  summarizer.reduce_scalar(tag, neural_network_model->obj_fn->report_aggregate_avg_obj_fn(execution_mode::training), step);
}

void lbann::target_layer::epoch_print() const {
  double obj_cost = neural_network_model->obj_fn->report_aggregate_avg_obj_fn(execution_mode::training);
  if (comm->am_world_master()) {
    std::vector<double> avg_obj_fn_costs(comm->get_num_models());
    comm->intermodel_gather(obj_cost, avg_obj_fn_costs);
    for (size_t i = 0; i < avg_obj_fn_costs.size(); ++i) {
      std::cout << "Model " << i << " average " << _to_string(neural_network_model->obj_fn->type) << ": " << avg_obj_fn_costs[i] <<
        std::endl;
    }
  } else {
    comm->intermodel_gather(obj_cost, comm->get_world_master());
  }
}

void lbann::target_layer::epoch_reset() {
  Layer::epoch_reset();
  resetCost();
}

void lbann::target_layer::resetCost() {
  neural_network_model->obj_fn->reset_obj_fn();
}

bool lbann::target_layer::saveToCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  /// @todo should probably save m_shared_data_reader
  return Layer::saveToCheckpoint(fd, filename, bytes);
}

bool lbann::target_layer::loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  /// @todo should probably save m_shared_data_reader
  return Layer::loadFromCheckpoint(fd, filename, bytes);
}

bool lbann::target_layer::saveToCheckpointShared(persist& p)
{
    // rank 0 writes softmax cost to file
    if (p.m_rank == 0) {
        // p.write_double(persist_type::train, "aggregate cost", (double) aggregate_cost);
        // p.write_uint64(persist_type::train, "num backprop steps", (uint64_t) num_backprop_steps);
    }
  
    return true;
}

bool lbann::target_layer::loadFromCheckpointShared(persist& p)
{
    // rank 0 writes softmax cost to file
    // if (p.m_rank == 0) {
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
