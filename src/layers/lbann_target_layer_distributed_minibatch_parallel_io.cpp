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

#include "lbann/layers/lbann_target_layer_distributed_minibatch_parallel_io.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::target_layer_distributed_minibatch_parallel_io::target_layer_distributed_minibatch_parallel_io(lbann_comm* comm, int num_parallel_readers, uint mini_batch_size, DataReader *training_data_reader, DataReader *testing_data_reader, bool shared_data_reader)
  : target_layer(comm, mini_batch_size, training_data_reader, testing_data_reader, shared_data_reader), 
    distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, training_data_reader->getNumData(), testing_data_reader->getNumData()),
    Ys(comm->get_model_grid()), YsColMax(comm->get_model_grid()), YsColMaxStar(comm->get_model_grid())
{
  //  NumNeurons = m_training_data_reader->get_linearized_label_size(); /// @todo NumNeurons should be hidden inside of an accessor function
}

lbann::target_layer_distributed_minibatch_parallel_io::target_layer_distributed_minibatch_parallel_io(lbann_comm* comm, int num_parallel_readers, uint mini_batch_size, DataReader *training_data_reader, bool shared_data_reader)
  : target_layer_distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, training_data_reader, NULL, shared_data_reader)
{
}

void lbann::target_layer_distributed_minibatch_parallel_io::setup(int num_prev_neurons) {
  if(!m_shared_data_reader) { /// If the target layer shares a data reader with an input layer, do not setup the data reader a second time
    io_layer::setup_data_readers(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                 m_num_parallel_readers_training * Layer::m_mini_batch_size);
  }

  /// @todo put in warning about bad target size
  if(num_prev_neurons != NumNeurons) {
    throw lbann_exception("lbann_target_layer_distributed_minibatch_parallel_io: number of neurons in previous layer does not match the number of neurons in the target layer.");
  }

  Zeros(*Ds_Temp, NumNeurons, Layer::m_mini_batch_size);
  Zeros(Y_local, NumNeurons, Layer::m_mini_batch_size);
  Zeros(Ys, NumNeurons, Layer::m_mini_batch_size);
  Zeros(YsColMax, Layer::m_mini_batch_size, 1); /// Note that the column max matrix has the number of mini-batches on the rows instead of columns
  Zeros(YsColMaxStar, Layer::m_mini_batch_size, 1);
  Zeros(m_max_index, Layer::m_mini_batch_size, 1);
  Zeros(m_reduced_max_indicies, Layer::m_mini_batch_size, 1);

  m_local_data_valid = false;
  m_local_reader_done = false;
  m_num_data_per_epoch = 0;
}

DataType lbann::target_layer_distributed_minibatch_parallel_io::forwardProp(DataType prev_WBL2NormSum) {
  int num_samples_in_batch = fetch_to_local_matrix(Y_local);
  target_layer::update_num_samples_processed(num_samples_in_batch);

  /// Compute the error between the previous layers activations and the ground truth
  ColumnMax((DistMat) *fp_input, YsColMax); /// For each minibatch (column) find the maximimum value
  Copy(YsColMax, YsColMaxStar); /// Give every rank a copy so that they can find the max index locally

  Zeros(m_max_index, Layer::m_mini_batch_size, 1);

  /// Find which rank holds the index for the maxmimum value
  for (int mb_index= 0; mb_index < fp_input->LocalWidth(); mb_index++) { /// For each sample in mini-batch that this rank has
    int mb_global_index = fp_input->GlobalCol(mb_index);
    DataType sample_max = YsColMaxStar.GetLocal(mb_global_index, 0);
    for (int f_index = 0; f_index < fp_input->LocalHeight(); f_index++) { /// For each feature
      if(fp_input->GetLocal(f_index, mb_index) == sample_max) {
        m_max_index.Set(mb_global_index, 0, fp_input->GlobalRow(f_index));
      }
    }
  }

  Zeros(m_reduced_max_indicies, Layer::m_mini_batch_size, 1);
  /// Merge all of the local index sets into a common buffer, if there are two potential maximum values, highest index wins
  Layer::comm->model_allreduce(m_max_index.Buffer(), m_max_index.Height() * m_max_index.Width(), m_reduced_max_indicies.Buffer(), mpi::MAX);

  /// Check to see if the predicted results match the target results
  int num_errors = 0;

  /// Allow the current root to compute the errors, since it has the data locally
  if(is_current_root()) {
    for (int mb_index= 0; mb_index < Y_local.Width(); mb_index++) { /// For each sample in mini-batch
      int targetidx = -1;
      float targetmax = 0;
      for (int f_index= 0; f_index < Y_local.Height(); f_index++) {
        if (targetmax < Y_local.Get(f_index, mb_index)) {
          targetmax = Y_local.Get(f_index, mb_index);
          targetidx = f_index;
        }
      }
      if(m_reduced_max_indicies.Get(mb_index, 0) != targetidx) {
        num_errors++;
      }
    }
  }
  num_errors = Layer::comm->model_broadcast(m_root, num_errors);

  distribute_from_local_matrix(Y_local, Ys);

  return num_errors;
}

void lbann::target_layer_distributed_minibatch_parallel_io::backProp() {
  /// Copy the results to the Ds_Temp variable for access by the next lower layer
  Copy(Ys, *Ds_Temp);
}

/**
 * Once a mini-batch is processed, resuffle the data for the next batch if necessary
 */
bool lbann::target_layer_distributed_minibatch_parallel_io::update() {
  return is_data_set_processed();
}

int lbann::target_layer_distributed_minibatch_parallel_io::fetch_from_data_reader(Mat &M_local) {
  DataReader *data_reader = target_layer::select_data_reader();
  return data_reader->fetch_label(M_local);
}

void lbann::target_layer_distributed_minibatch_parallel_io::preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {
  return;
}

bool lbann::target_layer_distributed_minibatch_parallel_io::update_data_reader() {
  DataReader *data_reader = target_layer::select_data_reader();
  if(m_shared_data_reader) { 
    /// If the data reader is shared with an input layer, don't update the reader just check to see if the epoch is done
    /// or will be done on the next update of the input layer (which includes adding the stride)
    return (data_reader->get_next_position() < data_reader->getNumData());
  }else {
    return data_reader->update();
  }
}
  
execution_mode lbann::target_layer_distributed_minibatch_parallel_io::get_execution_mode() {
  return m_execution_mode;
}
