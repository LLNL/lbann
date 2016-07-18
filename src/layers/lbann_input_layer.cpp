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

#include "lbann/layers/lbann_input_layer.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::input_layer::input_layer(lbann_comm* comm, uint mini_batch_size, DataReader *training_data_reader, DataReader *testing_data_reader, std::vector<regularizer*> regs)
  : Layer(0, comm, NULL, mini_batch_size, regs)
{
  m_training_data_reader = training_data_reader;
  m_testing_data_reader = testing_data_reader;
  m_num_training_samples_processed = 0;
  m_total_training_samples = 0;
  m_num_testing_samples_processed = 0;
  m_total_testing_samples = 0;

  /// @todo NumNeurons should be hidden inside of an accessor function
  if(m_training_data_reader != NULL && m_testing_data_reader != NULL 
     && m_training_data_reader->get_linearized_data_size() != m_testing_data_reader->get_linearized_data_size()) {
    throw lbann_exception("lbann_input_layer: testing data set size does not match training data set size");
  }

  if(m_training_data_reader != NULL) {
    NumNeurons = m_training_data_reader->get_linearized_data_size();
    m_total_training_samples = m_training_data_reader->getNumData();
  }

  if(m_testing_data_reader != NULL) {
    NumNeurons = m_testing_data_reader->get_linearized_data_size();
    m_total_testing_samples = m_testing_data_reader->getNumData();
  }
}

lbann::input_layer::input_layer(lbann_comm* comm, uint mini_batch_size, DataReader *training_data_reader)
  : input_layer(comm, mini_batch_size, training_data_reader, NULL, {}) {}

/**
 * Input layers are not able to return output matrices for backward propagation
 */
DistMat *lbann::input_layer::bp_output() {
  return NULL;
}

lbann::DataReader *lbann::input_layer::set_training_data_reader(DataReader *data_reader) {
  /// @todo put in a check to make sure that this is a data reader
  /// that matches what was already there
  DataReader *old_data_reader = m_training_data_reader;
  m_training_data_reader = data_reader;
  m_num_training_samples_processed = 0;
  m_total_training_samples = data_reader->getNumData();
  return old_data_reader;
}

lbann::DataReader *lbann::input_layer::set_testing_data_reader(DataReader *data_reader) {
  /// @todo put in a check to make sure that this is a data reader
  /// that matches what was already there
  DataReader *old_data_reader = m_testing_data_reader;
  m_testing_data_reader = data_reader;
  m_num_testing_samples_processed = 0;
  m_total_testing_samples = data_reader->getNumData();
  return old_data_reader;
}

lbann::DataReader *lbann::input_layer::select_data_reader() {
  switch(m_execution_mode) {
  case training:
    return m_training_data_reader;
    break;
  case validation:
    throw lbann_exception("lbann_input_layer: validation phase is not properly setup");
    break;
  case testing:
    return m_testing_data_reader;
    break;
  // case prediction:
  //   return m_prediction_data_reader;
  //   break;
  default:
    throw -1;
  }
}

long lbann::input_layer::update_num_samples_processed(long num_samples) {
  switch(m_execution_mode) {
  case training:
    m_num_training_samples_processed += num_samples;
    return m_num_training_samples_processed;
    break;
  case validation:
    throw lbann_exception("lbann_input_layer: validation phase is not properly setup");
    break;
  case testing:
    m_num_testing_samples_processed += num_samples;
    return m_num_testing_samples_processed;
    break;
  // case prediction:
  //   return m_prediction_data_reader;
  //   break;
  default:
    throw lbann_exception("lbann_input_layer: invalid execution phase");
  }
}
