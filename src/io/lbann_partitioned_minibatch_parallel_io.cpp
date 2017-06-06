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
// lbann_partitioned_minibatch_parallel_io .hpp .cpp - parallel I/O routines for distriubted minibatches
////////////////////////////////////////////////////////////////////////////////

#include "lbann/io/lbann_partitioned_minibatch_parallel_io.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;

lbann::partitioned_minibatch_parallel_io::partitioned_minibatch_parallel_io(lbann_comm *comm, int num_parallel_readers, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers)
  : comm(comm), m_num_parallel_readers_training(num_parallel_readers), m_num_parallel_readers_validating(num_parallel_readers), m_num_parallel_readers_testing(num_parallel_readers), m_max_mini_batch_size(mini_batch_size)
{
  m_root = 0;
  m_num_samples_in_batch = 0;
  m_num_valid_readers = 0;

  m_cur_step_in_epoch = 0;

  m_num_iterations_per_epoch_training = 0;
  m_num_iterations_per_epoch_validation = 0;
  m_num_iterations_per_epoch_testing = 0;

  int training_data_set_size = 0;
  int validation_data_set_size = 0;
  int testing_data_set_size = 0;

  if(data_readers[execution_mode::training] != NULL) {
    training_data_set_size = data_readers[execution_mode::training]->getNumData();
    m_num_iterations_per_epoch_training = data_readers[execution_mode::training]->get_num_iterations_per_epoch();
    //    std::cout << "Here is the number of iterations " << m_num_iterations_per_epoch_training << std::endl;
  }

  if(data_readers[execution_mode::validation] != NULL) {
    validation_data_set_size = data_readers[execution_mode::validation]->getNumData();
    m_num_iterations_per_epoch_validation = data_readers[execution_mode::validation]->get_num_iterations_per_epoch();
  }

  if(data_readers[execution_mode::testing] != NULL) {
    testing_data_set_size = data_readers[execution_mode::testing]->getNumData();
    m_num_iterations_per_epoch_testing = data_readers[execution_mode::testing]->get_num_iterations_per_epoch();
  }

  if(comm->get_model_grid().Size() != num_parallel_readers) {
    cout << "Warning the requested number of parallel readers " 
         << num_parallel_readers 
         << " does not match the grid size " << comm->get_model_grid().Size()
         << " OVERRIDING requested number of parallel readers."
         << endl;
    m_num_parallel_readers_training = comm->get_model_grid().Size();
    m_num_parallel_readers_validating = comm->get_model_grid().Size();
    m_num_parallel_readers_testing = comm->get_model_grid().Size();
    num_parallel_readers = comm->get_model_grid().Size();
  }

  if(mini_batch_size < num_parallel_readers) {
    cout << "Warning the requested number of parallel readers " 
         << num_parallel_readers 
         << " is larger than the requested mini-batch size " << mini_batch_size
         << " OVERRIDING requested number of parallel readers."
         << endl;
    m_num_parallel_readers_training = mini_batch_size;
    m_num_parallel_readers_validating = mini_batch_size;
    m_num_parallel_readers_testing = mini_batch_size;
  }

  /// Check to make sure that there is enough training data for all of the parallel readers
  if(training_data_set_size != 0) {
    int max_num_parallel_readers = m_num_parallel_readers_training;
    while(ceil((float)training_data_set_size / (float)(mini_batch_size * comm->get_num_models())) < max_num_parallel_readers) {
      max_num_parallel_readers--;
    }
    if(max_num_parallel_readers != m_num_parallel_readers_training) {
      cout << "Warning the training data set size "<<training_data_set_size
           <<" is too small for the number of requested parallel readers "
           <<m_num_parallel_readers_training<<", using "<< max_num_parallel_readers<<"." << endl;
      m_num_parallel_readers_training = max_num_parallel_readers;
    }
  }else {
    m_num_parallel_readers_training = 0;
  }

  /// Check to make sure that there is enough training data for all of the parallel readers
  if(validation_data_set_size != 0) {
    int max_num_parallel_readers = m_num_parallel_readers_validating;
    while(ceil((float)validation_data_set_size / (float)(mini_batch_size * comm->get_num_models())) < max_num_parallel_readers) {
      max_num_parallel_readers--;
    }
    if(max_num_parallel_readers != m_num_parallel_readers_validating) {
      cout << "Warning the validation data set size "<<validation_data_set_size
           <<" is too small for the number of requested parallel readers "
           <<m_num_parallel_readers_validating<<", using "<< max_num_parallel_readers<<"." << endl;
      m_num_parallel_readers_validating = max_num_parallel_readers;
    }
  }else {
    m_num_parallel_readers_validating = 0;
  }

  /// Check to make sure that there is enough testing data for all of the parallel readers
  if(testing_data_set_size != 0) {
    int max_num_parallel_readers = m_num_parallel_readers_testing;
    while(ceil((float)testing_data_set_size / (float)(mini_batch_size * comm->get_num_models())) < max_num_parallel_readers) {
      max_num_parallel_readers--;
    }
    if(max_num_parallel_readers != m_num_parallel_readers_testing) {
      cout << "Warning the testing data set size "<<testing_data_set_size
           <<" is too small for the number of requested parallel readers "
           <<m_num_parallel_readers_testing<<", using "<< max_num_parallel_readers<<"." << endl;
      m_num_parallel_readers_testing = max_num_parallel_readers;
    }
  }else {
    m_num_parallel_readers_testing = 0;
  }
}

int lbann::partitioned_minibatch_parallel_io::fetch_to_local_matrix(Mat& M_local) {
  int num_parallel_readers = get_num_parallel_readers();
  
  m_num_samples_in_batch = 0;

  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  if (comm->get_rank_in_model() < num_parallel_readers && (M_local.Height() != 0 && M_local.Width() != 0) && !m_local_reader_done) {
    // std::cout << "[" << comm->get_rank_in_model() << "] fetch to local matrix " << M_local.Height() << " x " << M_local.Width() << std::endl;
      Zero(M_local);

      /// Each data reader needs to either have independent / split
      /// data, or take an offset / stride
      m_num_samples_in_batch = fetch_from_data_reader(M_local);
      // if(m_num_samples_in_batch != 150) {
      // std::cout << "[" << comm->get_rank_in_model() << "] fetch to local matrix " << M_local.Height() << " x " << M_local.Width() << " only got samples = " << m_num_samples_in_batch << std::endl;
      // }
      bool data_valid = (m_num_samples_in_batch > 0);
      if(data_valid) {
        m_num_data_per_epoch+=m_num_samples_in_batch; /// BVE FIXME need to change how this is shared
        preprocess_data_samples(M_local, m_num_samples_in_batch);
      }
      m_local_data_valid = data_valid;
      //  }else {
    //    DISPLAY_MATRIX(M_local);
  }
  m_num_valid_readers = comm->model_allreduce((int) m_local_data_valid, mpi::SUM); /// BVE FIXME I don't think that we need this any more
  m_num_samples_in_batch = comm->model_allreduce((int) m_num_samples_in_batch, mpi::SUM); /// @todo compute this by dead reckoning to avoid allreduce 

  //    std::cout << "[" << comm->get_rank_in_model() << "] fetch to local matrix " << M_local.Height() << " x " << M_local.Width() << " and there are a total of samples = " << m_num_samples_in_batch << std::endl;
  return m_num_samples_in_batch;
}

void lbann::partitioned_minibatch_parallel_io::distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {

  /// Nothing to do here, it is already done
  return;
}

bool lbann::partitioned_minibatch_parallel_io::is_data_set_processed() {
  int num_readers_done = 0;
  int max_active_parallel_readers = get_num_parallel_readers();  // When calculating if all parallel readers are done, include the maximum number,
                                                                 // not just the ones in the last round.  This will ensure that all readers, that had data
                                                                 // will have partitioned it.
  int num_parallel_readers = m_num_valid_readers;

  int num_iterations_per_epoch = get_num_iterations_per_epoch();

  //  if(comm->get_rank_in_model() < num_parallel_readers) {
    // if((comm->get_rank_in_model()+1)%num_parallel_readers == m_root) {
    //   if(m_local_data_valid) { /// Make sure that all local data has been processed
    //     throw lbann_exception("lbann_input_layer_partitioned_minibatch_parallel_io: all valid data was not processed.");
    //   }
      m_local_reader_done = !update_data_reader();
  //   }
  // }

  /// Set the reduction variable
  if(m_local_reader_done) {
    num_readers_done = 1;
  }

  /// Once all of the readers have finished their part of the mini-batch indicate that the epoch is finished
  num_readers_done = comm->model_allreduce(num_readers_done);
  //  std::cout << "[" << comm->get_rank_in_model() << " of " << comm->get_model_rank() << "]"  << " IO layer is checking if the models are done " << num_readers_done << " and locally " << m_local_reader_done << " max active readers " << max_active_parallel_readers << " num iterations " << num_iterations_per_epoch << " and the current iteration is " << m_cur_step_in_epoch << std::endl;
  if(m_cur_step_in_epoch == (num_iterations_per_epoch - 1) /*num_readers_done >= max_active_parallel_readers*/) {
    m_local_reader_done = false;
    m_root = 0; /// When the epoch is finished, make sure that the root node for distributing data is reset because
                /// if the number of parallel readers does not evenly divide the data set size, the epoch will finish
                /// without all of the parallel readers participating in the last round.
    m_num_data_per_epoch = 0;
    m_cur_step_in_epoch = 0;
    return true;
  }else {
    m_cur_step_in_epoch++;
    return false;
  }
}

int lbann::partitioned_minibatch_parallel_io::get_num_parallel_readers() {
  int num_parallel_readers = 0;
  switch(get_execution_mode()) {
  case execution_mode::training:
    num_parallel_readers = m_num_parallel_readers_training;
    break;
  case execution_mode::validation:
    num_parallel_readers = m_num_parallel_readers_validating;
    break;
  case execution_mode::testing:
    num_parallel_readers = m_num_parallel_readers_testing;
    break;
  default:
    throw lbann_exception("lbann_partitioned_minibatch_parallel_io: invalid execution phase");
  }
  return num_parallel_readers;
}

int lbann::partitioned_minibatch_parallel_io::get_num_iterations_per_epoch() {
  int num_iterations_per_epoch = 0;
  switch(get_execution_mode()) {
  case execution_mode::training:
    num_iterations_per_epoch = m_num_iterations_per_epoch_training;
    break;
  case execution_mode::validation:
    num_iterations_per_epoch = m_num_iterations_per_epoch_validation;
    break;
  case execution_mode::testing:
    num_iterations_per_epoch = m_num_iterations_per_epoch_testing;
    break;
  default:
    throw lbann_exception("lbann_partitioned_minibatch_parallel_io: invalid execution phase");
  }
  return num_iterations_per_epoch;
}
