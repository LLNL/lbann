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

lbann::target_layer::target_layer(lbann_comm* comm, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, bool shared_data_reader, bool for_regression)
  : io_layer(comm, mini_batch_size, data_readers, std::vector<lbann::regularizer*>(), true, for_regression)
{
  if (is_for_regression())
    NumNeurons = io_layer::get_linearized_response_size();
  else
    NumNeurons = io_layer::get_linearized_label_size();
  m_shared_data_reader = shared_data_reader;
  aggregate_cost = 0.0;
  num_backprop_steps = 0;
}

void lbann::target_layer::setup(int num_prev_neurons) {
  if(neural_network_model->obj_fn == NULL) {
    throw lbann_exception("target layer has invalid objective function pointer");
  }
  neural_network_model->obj_fn->setup(NumNeurons, m_mini_batch_size);
  Zeros(*m_activations, NumNeurons, m_mini_batch_size);
  Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
}


/**
 * Target layers are not able to return target matrices for forward propagation
 */
DistMat *lbann::target_layer::fp_output() {
  return NULL;
}

lbann::DataReader *lbann::target_layer::set_training_data_reader(DataReader *data_reader, bool shared_data_reader) {
  m_shared_data_reader = shared_data_reader;
  return io_layer::set_training_data_reader(data_reader);
}

lbann::DataReader *lbann::target_layer::set_testing_data_reader(DataReader *data_reader, bool shared_data_reader) {
  m_shared_data_reader = shared_data_reader;
  return io_layer::set_testing_data_reader(data_reader);
}

void lbann::target_layer::fp_set_std_matrix_view() {
  int64_t cur_mini_batch_size = neural_network_model->get_current_mini_batch_size();
  Layer::fp_set_std_matrix_view();
  neural_network_model->obj_fn->fp_set_std_matrix_view(cur_mini_batch_size);
}

void lbann::target_layer::summarize(lbann_summary& summarizer, int64_t step) {
  Layer::summarize(summarizer, step);
  std::string tag = "layer" + std::to_string(static_cast<long long>(Index))
    + "/CrossEntropyCost";
  summarizer.reduce_scalar(tag, avgCost(), step);
}

void lbann::target_layer::epoch_print() const {
  double avg_cost = avgCost();
  if (comm->am_world_master()) {
    std::vector<double> avg_costs(comm->get_num_models());
    comm->intermodel_gather(avg_cost, avg_costs);
    for (size_t i = 0; i < avg_costs.size(); ++i) {
      std::cout << "Model " << i << " average cross entropy cost: " << avg_costs[i] <<
        std::endl;
    }
  } else {
    comm->intermodel_gather(avg_cost, comm->get_world_master());
  }
}

void lbann::target_layer::epoch_reset() {
  Layer::epoch_reset();
  resetCost();
}

void lbann::target_layer::resetCost() {
  aggregate_cost = 0.0;
  num_backprop_steps = 0;
}

DataType lbann::target_layer::avgCost() const {
  return aggregate_cost / num_backprop_steps;
}

bool lbann::target_layer::saveToCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  ssize_t write_rc = write(fd, &aggregate_cost, sizeof(aggregate_cost));
  if (write_rc != sizeof(aggregate_cost)) {
    // error!
  }
  *bytes += write_rc;

  write_rc = write(fd, &num_backprop_steps, sizeof(num_backprop_steps));
  if (write_rc != sizeof(num_backprop_steps)) {
    // error!
  }
  *bytes += write_rc;

  return Layer::saveToCheckpoint(fd, filename, bytes);
}

bool lbann::target_layer::loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes)
{
  ssize_t read_rc = read(fd, &aggregate_cost, sizeof(aggregate_cost));
  if (read_rc != sizeof(aggregate_cost)) {
    // error!
  }
  *bytes += read_rc;

  read_rc = read(fd, &num_backprop_steps, sizeof(num_backprop_steps));
  if (read_rc != sizeof(num_backprop_steps)) {
    // error!
  }
  *bytes += read_rc;

  return Layer::loadFromCheckpoint(fd, filename, bytes);
}

bool lbann::target_layer::saveToCheckpointShared(const char* dir, uint64_t* bytes)
{
  // get our rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // rank 0 writes softmax cost to file
  if (rank == 0) {
      // define the filename
      char file[1024];
      sprintf(file, "%s/target_L%d", dir, Index);

      // open the file
      int fd = lbann::openwrite(file);
      if (fd != -1 ) {
          ssize_t write_rc = write(fd, &aggregate_cost, sizeof(aggregate_cost));
          if (write_rc != sizeof(aggregate_cost)) {
            // error!
          }
          *bytes += write_rc;

          write_rc = write(fd, &num_backprop_steps, sizeof(num_backprop_steps));
          if (write_rc != sizeof(num_backprop_steps)) {
            // error!
          }
          *bytes += write_rc;

          // close the file
          lbann::closewrite(fd, file);
      }
  }

  //return Layer::saveToCheckpointShared(dir, bytes);
  return true;
}

bool lbann::target_layer::loadFromCheckpointShared(const char* dir, uint64_t* bytes)
{
    // get our rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // rank 0 writes softmax cost to file
    if (rank == 0) {
        // define the filename
        char file[1024];
        sprintf(file, "%s/target_L%d", dir, Index);

        // open the file
        int fd = lbann::openread(file);
        if (fd != -1 ) {
            ssize_t read_rc = read(fd, &aggregate_cost, sizeof(aggregate_cost));
            if (read_rc != sizeof(aggregate_cost)) {
              // error!
            }
            *bytes += read_rc;

            read_rc = read(fd, &num_backprop_steps, sizeof(num_backprop_steps));
            if (read_rc != sizeof(num_backprop_steps)) {
              // error!
            }
            *bytes += read_rc;

            // close the file
            lbann::closeread(fd, file);
        }
    }

    // get values from rank 0
    MPI_Bcast(&aggregate_cost, 1, DataTypeMPI, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_backprop_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    //return Layer::loadFromCheckpointShared(dir, bytes);
    return true;
}
