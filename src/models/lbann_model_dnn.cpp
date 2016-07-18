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
//
// lbann_model_dnn .hpp .cpp - Deep Neural Networks models
////////////////////////////////////////////////////////////////////////////////

#include "lbann/models/lbann_model_dnn.hpp"
#include "lbann/layers/lbann_layer_fully_connected.hpp"
#include "lbann/layers/lbann_layer_softmax.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"

#include <string>
#include <chrono>
#include <random>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include "mpi.h"

using namespace std;
using namespace El;


////////////////////////////////////////////////////////////////////////////////
// Dnn : main deep neural network class
////////////////////////////////////////////////////////////////////////////////

lbann::Dnn::Dnn(Optimizer_factory *optimizer_factory, const uint MiniBatchSize,
                lbann_comm* comm, layer_factory* layer_fac)
  : Sequential(optimizer_factory, MiniBatchSize, comm, layer_fac), training_accuracy(0.0),
    validation_accuracy(0.0), test_accuracy(0.0)
{
}

lbann::Dnn::Dnn(const uint MiniBatchSize,
                const double Lambda, Optimizer_factory *optimizer_factory,
                lbann_comm* comm, layer_factory* layer_fac)
  : Sequential(optimizer_factory, MiniBatchSize, comm, layer_fac), training_accuracy(0.0),
    validation_accuracy(0.0), test_accuracy(0.0)
{

}

lbann::Dnn::~Dnn()
{
    // // free neural network (layers)
    // for (size_t l = 0; l < Layers.size(); l++) {
    //     delete Layers[l];
    // }
}

void lbann::Dnn::checkGradient(CircMat& X, CircMat& Y, double* GradientErrors)
{
    // setup input (last/additional row should always be 1)
  Copy(X, *(Layers[0]->Acts));

    // forward propagation (mini-batch)
    DataType L2NormSum = 0;
    for (size_t l = 1; l < Layers.size(); l++)
      L2NormSum = Layers[l]->forwardProp(L2NormSum);

    // backward propagation (mini-batch)
    for (size_t l = Layers.size() - 1; l >= 1; l--) {
      Layers[l]->backProp();
    }

    // check gradient
    GradientErrors[0] = 0;
    for (size_t l = 1; l < Layers.size(); l++)
        GradientErrors[l] = Layers[l]->checkGradientMB(*Layers[l-1]);
}

void lbann::Dnn::train(int NumEpoch, bool EvaluateEveryEpoch)
{
  do_train_begin_cbs();
  // do main loop for epoch
  for (int epoch = 0; epoch < NumEpoch; epoch++) {
    if (get_terminate_training()) {
      // Training has been terminated for this model.
      break;
    }
    ++cur_epoch;
    do_epoch_begin_cbs();
    /// Set the correct execution mode so that the proper data reader
    /// is used
    for (size_t l = 0; l < Layers.size(); l++) {
      Layers[l]->m_execution_mode = training;
    }

    long num_samples = 0;
    long num_errors = 0;

    // Trigger a shuffle of the input data
    bool epoch_done;
    do {
      epoch_done = trainBatch(&num_samples, &num_errors);
    }while(!epoch_done); /// The data reader will automatically
                         /// shuffle the data after each epoch

    training_accuracy = (float)(num_samples - num_errors) / num_samples * 100.0f;

    if(EvaluateEveryEpoch) {
      do_validation_begin_cbs();
      validation_accuracy = evaluate();
      do_validation_end_cbs();
    }

    do_epoch_end_cbs();
  }
  do_train_end_cbs();
}

void lbann::Dnn::summarize(lbann_summary& summarizer) {
  for (size_t l = 1; l < Layers.size(); ++l) {
    Layers[l]->summarize(summarizer, get_cur_step());
  }
}

bool lbann::Dnn::trainBatch(long *num_samples, long *num_errors)
{
  do_batch_begin_cbs();
  bool data_set_processed = false;

  do_model_forward_prop_begin_cbs();
  // forward propagation (mini-batch)
  DataType L2NormSum = 0;
  for (size_t l = 0; l < Layers.size(); l++) {
    do_layer_forward_prop_begin_cbs(Layers[l]);
    L2NormSum = Layers[l]->forwardProp(L2NormSum);
    do_layer_forward_prop_end_cbs(Layers[l]);
  }
  *num_errors += (long) L2NormSum;
  *num_samples += MiniBatchSize;
  do_model_forward_prop_end_cbs();

  // Update training accuracy.
  training_accuracy = ((DataType) *num_errors) / ((DataType) *num_samples) * 100.0;
  ++cur_step;

  do_model_backward_prop_begin_cbs();
  // backward propagation (mini-batch)
  for (size_t l = Layers.size(); l-- > 0;) {
    do_layer_backward_prop_begin_cbs(Layers[l]);
    Layers[l]->backProp();
    do_layer_backward_prop_end_cbs(Layers[l]);
  }
  do_model_backward_prop_end_cbs();

  // Make sure that the output error is periodically reported in backpropagation

  /// update weights, biases -- starting with the deepest layer and
  /// finishing with the input layer
  //  for (size_t l = 0; l < Layers.size(); l++) {
  for (size_t l = Layers.size(); l-- > 0;) {
    data_set_processed = Layers[l]->update();
  }

  do_batch_end_cbs();

  return data_set_processed;
}

DataType lbann::Dnn::evaluate()
{
  // test

  //  DataType error = 0;
  long num_samples = 0;
  long num_errors = 0;

  do_test_begin_cbs();

  /// Set the mode for each layer so that validation data is used
  for (size_t l = 0; l < Layers.size(); l++) {
    Layers[l]->m_execution_mode = testing;
  }

  // Trigger a shuffle of the input data
  bool epoch_done;
  do {
    epoch_done = evaluateBatch(&num_samples, &num_errors);
  }while(!epoch_done); /// The data reader will automatically
  /// shuffle the data after each epoch

  test_accuracy = (DataType)(num_samples - num_errors) / num_samples * 100.0f;

  do_test_end_cbs();
  return test_accuracy;
}

bool lbann::Dnn::evaluateBatch(long *num_samples, long *num_errors)
{
  bool data_set_processed = false;

  // forward propagation (mini-batch)
  DataType L2NormSum = 0;
  for (size_t l = 0; l < Layers.size(); l++) {
    L2NormSum = Layers[l]->forwardProp(L2NormSum);
  }
  *num_errors += (long) L2NormSum;
  *num_samples += MiniBatchSize;

  /// @todo change this so that every layer is called to update, but
  /// FC and other will not apply gradients in testing mode
  Layers[Layers.size()-1]->update();
  data_set_processed = Layers[0]->update();
  return data_set_processed;
}
