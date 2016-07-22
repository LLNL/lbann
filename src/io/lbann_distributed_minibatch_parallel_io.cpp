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
// lbann_distributed_minibatch_parallel_io .hpp .cpp - parallel I/O routines for distriubted minibatches
////////////////////////////////////////////////////////////////////////////////

#include "lbann/io/lbann_distributed_minibatch_parallel_io.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;

lbann::distributed_minibatch_parallel_io::distributed_minibatch_parallel_io(lbann_comm *comm, int num_parallel_readers, uint mini_batch_size, int training_data_set_size, int testing_data_set_size)
  : comm(comm), m_num_parallel_readers_training(num_parallel_readers), m_num_parallel_readers_testing(num_parallel_readers), m_mini_batch_size(mini_batch_size)
{
  m_root = 0;
  
  if(comm->get_model_grid().Size() < num_parallel_readers) {
    cout << "Warning the grid size "<<comm->get_model_grid().Size()
         <<"is smaller than the number of requested parallel readers "
         <<num_parallel_readers<<"." << endl;
    m_num_parallel_readers_training = comm->get_model_grid().Size();
    m_num_parallel_readers_testing = comm->get_model_grid().Size();
  }

  /// Check to make sure that there is enough training data for all of the parallel readers
  if(training_data_set_size != 0) {
    int max_num_parallel_readers = m_num_parallel_readers_training;
    while(ceil((float)training_data_set_size/(float)mini_batch_size) < max_num_parallel_readers) {
      max_num_parallel_readers--;
    }
    if(max_num_parallel_readers != m_num_parallel_readers_training) {
      cout << "Warning the training data set size "<<training_data_set_size
           <<" is too small for the number of requested parallel readers "
           <<m_num_parallel_readers_training<<", using "<< max_num_parallel_readers<<"." << endl;
      m_num_parallel_readers_training = max_num_parallel_readers;
    }
  }

  /// Check to make sure that there is enough testing data for all of the parallel readers
  if(testing_data_set_size != 0) {
    int max_num_parallel_readers = m_num_parallel_readers_testing;
    while(ceil((float)testing_data_set_size/(float)mini_batch_size) < max_num_parallel_readers) {
      max_num_parallel_readers--;
    }
    if(max_num_parallel_readers != m_num_parallel_readers_testing) {
      cout << "Warning the testing data set size "<<testing_data_set_size
           <<" is too small for the number of requested parallel readers "
           <<m_num_parallel_readers_testing<<", using "<< max_num_parallel_readers<<"." << endl;
      m_num_parallel_readers_testing = max_num_parallel_readers;
    }
  }
}

int lbann::distributed_minibatch_parallel_io::fetch_to_local_matrix(Mat& M_local) {
  int num_parallel_readers = get_num_parallel_readers();
  int num_samples_in_batch = 0;

  /// Check to see if this rank has valid data -- if not read in the next batch
  /// Coordinate all available readers so that the perform I/O in the same step
  if (m_root == 0) {
    if (comm->get_rank_in_model() < num_parallel_readers && !m_local_reader_done) {
      Zero(M_local);

      /// Each data reader needs to either have independent / split
      /// data, or take an offset / stride
      num_samples_in_batch = fetch_from_data_reader(M_local);
      bool data_valid = (num_samples_in_batch > 0);
      if(data_valid) {
        m_num_data_per_epoch+=num_samples_in_batch;
      }
      preprocess_data_samples(M_local, num_samples_in_batch);
      m_local_data_valid = data_valid;
    }
  }
  return num_samples_in_batch;
}

void lbann::distributed_minibatch_parallel_io::distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {
  int num_parallel_readers = get_num_parallel_readers();
  Ms.SetRoot(m_root);

  comm->model_barrier();

  if (comm->get_rank_in_model() == m_root) {
    if(!m_local_data_valid) {
      throw lbann_exception("lbann_distributed_minibatch_parallel_io: No valid data for this step -- local data was invalid");
    }
    CopyFromRoot(M_local, Ms);
    m_local_data_valid = false;
  }else {
    CopyFromNonRoot(Ms);
  }

  comm->model_barrier();
   
  m_root = (m_root + 1) % num_parallel_readers;
  return;
}

bool lbann::distributed_minibatch_parallel_io::is_data_set_processed() {
  int num_readers_done = 0;
  int num_parallel_readers = get_num_parallel_readers();

  if(comm->get_rank_in_model() < num_parallel_readers) {
    if((comm->get_rank_in_model()+1)%num_parallel_readers == m_root) {
      if(m_local_data_valid) { /// Make sure that all local data has been processed
        throw lbann_exception("lbann_input_layer_distributed_minibatch_parallel_io: all valid data was not processed.");
      }
      m_local_reader_done = !update_data_reader();
    }
  }

  /// Set the reduction variable
  if(m_local_reader_done) {
    num_readers_done = 1;
  }

  /// Once all of the readers have finished their part of the mini-batch indicate that the epoch is finished
  num_readers_done = comm->model_allreduce(num_readers_done);
  if(num_readers_done == num_parallel_readers) {
    m_local_reader_done = false;
    m_root = 0; /// When the epoch is finished, make sure that the root node for distributing data is reset because
                /// if the number of parallel readers does not evenly divide the data set size, the epoch will finish
                /// without all of the parallel readers participating in the last round.
    m_num_data_per_epoch = 0;
    return true;
  }else {
    return false;
  }
}

int lbann::distributed_minibatch_parallel_io::get_num_parallel_readers() {
  int num_parallel_readers = 0;
  switch(get_execution_mode()) {
  case execution_mode::training:
    num_parallel_readers = m_num_parallel_readers_training;
    break;
  case execution_mode::testing:
    num_parallel_readers = m_num_parallel_readers_testing;
    break;
  default:
    throw lbann_exception("lbann_distributed_minibatch_parallel_io: invalid execution phase");
  }
  return num_parallel_readers;
}
