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
#include "lbann/models/lbann_model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::target_layer_distributed_minibatch_parallel_io::target_layer_distributed_minibatch_parallel_io(lbann_comm* comm, int num_parallel_readers, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, bool shared_data_reader, bool for_regression)
  : target_layer(comm, mini_batch_size, data_readers, shared_data_reader, for_regression), 
    distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, data_readers),
    Ys(comm->get_model_grid()), YsColMax(comm->get_model_grid()), YsColMaxStar(comm->get_model_grid())
{
  //  NumNeurons = m_training_data_reader->get_linearized_label_size(); /// @todo NumNeurons should be hidden inside of an accessor function
}

void lbann::target_layer_distributed_minibatch_parallel_io::setup(int num_prev_neurons) {
  target_layer::setup(num_prev_neurons);
  if(!m_shared_data_reader) { /// If the target layer shares a data reader with an input layer, do not setup the data reader a second time
    if(io_layer::m_data_sets_span_models) {
      int stride = Layer::comm->get_num_models() * m_num_parallel_readers_training * Layer::m_mini_batch_size;
      int base_offset = Layer::comm->get_rank_in_model() * Layer::comm->get_num_models() * Layer::m_mini_batch_size;
      int model_offset = Layer::comm->get_model_rank() * Layer::m_mini_batch_size;
      //cout << "Setting up input layer, with " << Layer::comm->get_num_models() << " models and " << m_num_parallel_readers_training << " parallel readers and " << Layer::m_mini_batch_size << " mb size, which gives a stride of " << stride << endl;
      io_layer::setup_data_readers_for_training(base_offset,
                                                stride,
                                                model_offset);
      /// Note that the data readers for evaluation should not be partitioned over multiple models (otherwise each model will be scored on a different set of data)
      io_layer::setup_data_readers_for_evaluation(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                  m_num_parallel_readers_training * Layer::m_mini_batch_size);
    }else {
      io_layer::setup_data_readers_for_training(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                m_num_parallel_readers_training * Layer::m_mini_batch_size);
      io_layer::setup_data_readers_for_evaluation(Layer::comm->get_rank_in_model() * Layer::m_mini_batch_size,
                                                  m_num_parallel_readers_training * Layer::m_mini_batch_size);
    }
  }

  /// @todo put in warning about bad target size
  if(num_prev_neurons != NumNeurons) {
    throw lbann_exception("lbann_target_layer_distributed_minibatch_parallel_io: number of neurons in previous layer does not match the number of neurons in the target layer.");
  }

  Zeros(*m_error_signal, NumNeurons, Layer::m_mini_batch_size);
  Zeros(Y_local, NumNeurons, Layer::m_mini_batch_size);
  Zeros(Ys, NumNeurons, Layer::m_mini_batch_size);
  if (!is_for_regression()) {
    Zeros(YsColMax, Layer::m_mini_batch_size, 1); /// Note that the column max matrix has the number of mini-batches on the rows instead of columns
    Zeros(YsColMaxStar, Layer::m_mini_batch_size, 1);
    Zeros(m_max_index, Layer::m_mini_batch_size, 1);
    Zeros(m_reduced_max_indicies, Layer::m_mini_batch_size, 1);
  }
  Zeros(*m_prev_activations, num_prev_neurons, m_mini_batch_size);
  Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
  Zeros(*m_activations, NumNeurons, m_mini_batch_size);

  m_local_data_valid = false;
  m_local_reader_done = false;
  m_num_data_per_epoch = 0;
}

DataType lbann::target_layer_distributed_minibatch_parallel_io::forwardProp(DataType prev_WBL2NormSum) {
  if (is_for_regression())
      return forwardProp_regression(prev_WBL2NormSum);
  else
      return forwardProp_classification(prev_WBL2NormSum);
}

///@todo update this to use the new fp_linearity framework
DataType lbann::target_layer_distributed_minibatch_parallel_io::forwardProp_classification(DataType prev_WBL2NormSum) {
  int num_samples_in_batch = fetch_to_local_matrix(Y_local);
  if(is_current_root()) {
    /// Only update the number of samples processed by this parallel reader, when it is the current root
    target_layer::update_num_samples_processed(num_samples_in_batch);
  }

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();

  if(is_current_root() && num_samples_in_batch != curr_mini_batch_size) {
    throw lbann_exception("lbann_target_layer_distributed_minibatch_parallel_io: number of labels does not match the current mini-batch size.");
  }

  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input);
  DistMat& X = XProxy.Get();
  DistMat X_v;
  View(X_v, X, IR(0, X.Height()), IR(0, curr_mini_batch_size));
  Mat Y_local_v;
  View(Y_local_v, Y_local, IR(0, Y_local.Height()), IR(0, curr_mini_batch_size));

  *m_prev_activations = *fp_input; // BVE this should be handled in the new fp framework
  View(*m_prev_activations_v, *m_prev_activations, IR(0, m_prev_activations->Height()), IR(0, curr_mini_batch_size));
  target_layer::fp_set_std_matrix_view();

  /// This is computing the categorical accuracy

  // Clear the contents of the intermediate matrices
  Zeros(YsColMax, Layer::m_mini_batch_size, 1);
  Zeros(YsColMaxStar, Layer::m_mini_batch_size, 1);

  /// Compute the error between the previous layers activations and the ground truth
  ColumnMax((DistMat)*m_prev_activations_v, YsColMax); /// For each minibatch (column) find the maximimum value
  Copy(YsColMax, YsColMaxStar); /// Give every rank a copy so that they can find the max index locally

  Zeros(m_max_index, Layer::m_mini_batch_size, 1); // Clear the entire matrix

  /// Find which rank holds the index for the maxmimum value
  for (int mb_index = 0; mb_index < m_prev_activations_v->LocalWidth(); mb_index++) { /// For each sample in mini-batch that this rank has
    int mb_global_index = m_prev_activations_v->GlobalCol(mb_index);
    DataType sample_max = YsColMaxStar.GetLocal(mb_global_index, 0);
    for (int f_index = 0; f_index < m_prev_activations_v->LocalHeight(); f_index++) { /// For each feature
      if(m_prev_activations_v->GetLocal(f_index, mb_index) == sample_max) {
        m_max_index.Set(mb_global_index, 0, m_prev_activations_v->GlobalRow(f_index));
      }
    }
  }

  Zeros(m_reduced_max_indicies, Layer::m_mini_batch_size, 1); // Clear the entire matrix
  /// Merge all of the local index sets into a common buffer, if there are two potential maximum values, highest index wins
  Layer::comm->model_allreduce(m_max_index.Buffer(), m_max_index.Height() * m_max_index.Width(), m_reduced_max_indicies.Buffer(), mpi::MAX);

  /// Check to see if the predicted results match the target results
  int num_errors = 0;

  /// Allow the current root to compute the errors, since it has the data locally
  if(is_current_root()) {
    for (int mb_index= 0; mb_index < Y_local_v.Width(); mb_index++) { /// For each sample in mini-batch
      int targetidx = -1;
      float targetmax = 0;
      for (int f_index= 0; f_index < Y_local_v.Height(); f_index++) {
        if (targetmax < Y_local_v.Get(f_index, mb_index)) {
          targetmax = Y_local_v.Get(f_index, mb_index);
          targetidx = f_index;
        }
      }
      if(m_reduced_max_indicies.Get(mb_index, 0) != targetidx) {
        num_errors++;
      }
    }
  }
  num_errors = Layer::comm->model_broadcast(m_root, num_errors);

  /// @todo should this distribute the entire matrix even if there is only a partial mini-batch
  distribute_from_local_matrix(Y_local, Ys);
  Copy(Ys, *m_activations);

  int tmp_num_errors = 0;
  for (auto&& m : neural_network_model->metrics) {
    tmp_num_errors = (int) m->compute_metric(*m_prev_activations_v, *m_activations_v);
    m->record_error(tmp_num_errors, curr_mini_batch_size);
  }

  if(num_errors != (int) tmp_num_errors) {
    cout << "The new metric function calculated " << tmp_num_errors << " errors and the old function computed " << num_errors << endl;
  }

  return num_errors;
}

DataType lbann::target_layer_distributed_minibatch_parallel_io::forwardProp_regression(DataType prev_WBL2NormSum) {
  int num_samples_in_batch = fetch_to_local_matrix(Y_local);
  if(is_current_root()) {
    /// Only update the number of samples processed by this parallel reader, when it is the current root
    target_layer::update_num_samples_processed(num_samples_in_batch);
  }

  int64_t curr_mini_batch_size = neural_network_model->get_current_mini_batch_size();

  if(is_current_root() && num_samples_in_batch != curr_mini_batch_size) {
    throw lbann_exception("lbann_target_layer_distributed_minibatch_parallel_io: number of responses does not match the current mini-batch size.");
  }

  DistMatrixReadProxy<DataType,DataType,MC,MR> XProxy(*fp_input);
  DistMat& X = XProxy.Get();
  DistMat X_v;
  View(X_v, X, IR(0, X.Height()), IR(0, curr_mini_batch_size));
  Mat Y_local_v;
  View(Y_local_v, Y_local, IR(0, Y_local.Height()), IR(0, curr_mini_batch_size));

  *m_prev_activations = *fp_input; // BVE this should be handled in the new fp framework
  View(*m_prev_activations_v, *m_prev_activations, IR(0, m_prev_activations->Height()), IR(0, curr_mini_batch_size));
  target_layer::fp_set_std_matrix_view();


  DataType sumSqErr = static_cast<DataType>(0);
  DataType SSE = static_cast<DataType>(0); // error sum of squares: sum of (response-predicted)^2

  /// Allow the current root to compute the errors, since it has the data locally
  if(is_current_root()) {
    for (int mb_index= 0; mb_index < Y_local_v.Width(); mb_index++) { /// For each sample in mini-batch
      DataType response = Y_local_v.Get(0, mb_index); // truth
      DataType predicted = X_v.Get(0, mb_index); // prediction
      SSE += (response - predicted)*(response - predicted);
      //cout << '#' << mb_index << ' ' << response << ' ' << predicted << endl;
    }
  }
  SSE = Layer::comm->model_broadcast(m_root, SSE);
  /// @todo should this distribute the entire matrix even if there is only a partial mini-batch
  distribute_from_local_matrix(Y_local, Ys);
  Copy(Ys, *m_activations);

  return SSE;
}

void lbann::target_layer_distributed_minibatch_parallel_io::backProp() {
  /// Compute the error between the target values and the previous layer's activations
  /// Copy the results to the m_error_signal variable for access by the next lower layer
  Copy(*m_prev_activations, *m_error_signal); // delta = (activation - y)
  Axpy(-1., *m_activations, *m_error_signal); // Per-neuron error
  /// @todo - BVE should we be using views here.

  if (m_execution_mode == execution_mode::training) {
    DataType avg_error = neural_network_model->obj_fn->compute_obj_fn(*m_prev_activations_v, *m_activations_v);
    aggregate_cost += avg_error;
    num_backprop_steps++;
  }
}

/**
 * Once a mini-batch is processed, resuffle the data for the next batch if necessary
 */
bool lbann::target_layer_distributed_minibatch_parallel_io::update() {
  return is_data_set_processed();
}

int lbann::target_layer_distributed_minibatch_parallel_io::fetch_from_data_reader(Mat &M_local) {
  DataReader *data_reader = target_layer::select_data_reader();
  if (is_for_regression())
    return data_reader->fetch_response(M_local);
  else
    return data_reader->fetch_label(M_local);
}

void lbann::target_layer_distributed_minibatch_parallel_io::preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {
  return;
}

bool lbann::target_layer_distributed_minibatch_parallel_io::update_data_reader() {
  DataReader *data_reader = target_layer::select_data_reader();
  if(m_shared_data_reader) { 
    /// If the data reader is shared with an input layer, don't update the reader just check to see if the epoch is done
    /// or will be done on the next update of the input layer (which includes adding the stride).
    /// Note that target layers are always update before input layers, which is why the position
    /// is not up to date yet.
    return (data_reader->get_next_position() < data_reader->getNumData());
  }else {
    return data_reader->update();
  }
}
  
execution_mode lbann::target_layer_distributed_minibatch_parallel_io::get_execution_mode() {
  return m_execution_mode;
}
