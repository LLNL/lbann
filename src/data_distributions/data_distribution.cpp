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

#include "lbann/data_distributions/data_distribution.hpp"
#include "lbann/utils/exception.hpp"

lbann::generic_data_distribution::generic_data_distribution(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
  : m_comm(comm), m_requested_max_num_parallel_readers(num_parallel_readers), m_data_readers(data_readers) {
  m_root = 0;
  m_num_samples_in_batch = 0;
}

lbann::generic_data_reader *lbann::generic_data_distribution::get_data_reader(execution_mode mode) {
  generic_data_reader *data_reader;
  switch(mode) {
  case execution_mode::training:
    data_reader = m_data_readers[execution_mode::training];
    break;
  case execution_mode::validation:
    data_reader = m_data_readers[execution_mode::validation];
    break;
  case execution_mode::testing:
    data_reader = m_data_readers[execution_mode::testing];
    break;
  default:
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic data distribution: invalid execution phase");
  }
  return data_reader;
}

lbann::generic_data_reader *lbann::generic_data_distribution::get_data_reader() {
  return get_data_reader(get_execution_mode());
}

int lbann::generic_data_distribution::get_num_parallel_readers(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_num_parallel_readers();
}

int lbann::generic_data_distribution::get_num_parallel_readers() {
  return get_num_parallel_readers(get_execution_mode());
}

int lbann::generic_data_distribution::get_num_iterations_per_epoch(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_num_iterations_per_epoch();
}

int lbann::generic_data_distribution::get_num_iterations_per_epoch() {
  return get_num_iterations_per_epoch(get_execution_mode());
}

int lbann::generic_data_distribution::get_current_step_in_epoch(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_current_step_in_epoch();
}

int lbann::generic_data_distribution::get_current_step_in_epoch() {
  return get_current_step_in_epoch(get_execution_mode());
}

int lbann::generic_data_distribution::get_mini_batch_size(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_mini_batch_size();
}

int lbann::generic_data_distribution::get_last_mini_batch_size(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_last_mini_batch_size();
}

int lbann::generic_data_distribution::get_last_mini_batch_size() {
  return get_last_mini_batch_size(get_execution_mode());
}

int lbann::generic_data_distribution::get_current_mini_batch_size(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_current_mini_batch_size();
}

int lbann::generic_data_distribution::get_current_mini_batch_size() {
  return get_current_mini_batch_size(get_execution_mode());
}

int lbann::generic_data_distribution::get_global_mini_batch_size(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_global_mini_batch_size();
}

int lbann::generic_data_distribution::get_global_last_mini_batch_size(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_global_last_mini_batch_size();
}

int lbann::generic_data_distribution::get_current_global_mini_batch_size(execution_mode mode) {
  generic_data_reader *data_reader = get_data_reader(mode);
  return data_reader->get_current_global_mini_batch_size();
}

int lbann::generic_data_distribution::get_current_global_mini_batch_size() {
  return get_current_global_mini_batch_size(get_execution_mode());
}

/** Calculate how many iterations are required for training, testing,
 *  and validation given a specified mini-batch size and that the
 *  training data set is spanning all of the models.
 */
void lbann::generic_data_distribution::calculate_num_iterations_per_epoch_training_spans_models(int mini_batch_size) {

  /// Setup the training data set so that it spans all models
  calculate_num_iterations_per_epoch_spanning_models(mini_batch_size,
                                                     m_data_readers[execution_mode::training]);

  /// Each model uses the entire validation and testing data sets
  calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                  m_data_readers[execution_mode::validation]);
  calculate_num_iterations_per_epoch_single_model(mini_batch_size, 
                                                  m_data_readers[execution_mode::testing]);

}

void lbann::generic_data_distribution::calculate_num_iterations_per_epoch_training_unique_per_models(int mini_batch_size) {

  /// Setup the training data set so that it spans all models
  calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                   m_data_readers[execution_mode::training]);

  /// Each model uses the entire validation and testing data sets
  calculate_num_iterations_per_epoch_single_model(mini_batch_size,
                                                  m_data_readers[execution_mode::validation]);
  calculate_num_iterations_per_epoch_single_model(mini_batch_size, 
                                                  m_data_readers[execution_mode::testing]);

}
