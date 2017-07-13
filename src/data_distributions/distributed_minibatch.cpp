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

#include "lbann/data_distributions/distributed_minibatch.hpp"
#include "lbann/utils/exception.hpp"

using namespace std;

lbann::distributed_minibatch::distributed_minibatch(lbann_comm *comm, int num_parallel_readers, int mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers)
  : generic_data_distribution(comm, num_parallel_readers, mini_batch_size, data_readers) { 

  int training_data_set_size = 0;
  int validation_data_set_size = 0;
  int testing_data_set_size = 0;

  if(data_readers[execution_mode::training] != NULL) {
    training_data_set_size = data_readers[execution_mode::training]->getNumData();
  }

  if(data_readers[execution_mode::validation] != NULL) {
    validation_data_set_size = data_readers[execution_mode::validation]->getNumData();
  }

  if(data_readers[execution_mode::testing] != NULL) {
    testing_data_set_size = data_readers[execution_mode::testing]->getNumData();
  }

  if(m_comm->get_model_grid().Size() < num_parallel_readers) {
    if (m_comm->am_model_master()) {
      cout << "Warning the grid size "<<m_comm->get_model_grid().Size()
           <<"is smaller than the number of requested parallel readers "
           <<num_parallel_readers<<"." << endl;
    }
    m_num_parallel_readers_training = m_comm->get_model_grid().Size();
    m_num_parallel_readers_validating = m_comm->get_model_grid().Size();
    m_num_parallel_readers_testing = m_comm->get_model_grid().Size();
  }

  /// Check to make sure that there is enough data for all of the parallel readers
  m_num_parallel_readers_training = compute_max_num_parallel_readers(training_data_set_size, mini_batch_size, m_num_parallel_readers_training);

  m_num_parallel_readers_validating = compute_max_num_parallel_readers(validation_data_set_size, mini_batch_size, m_num_parallel_readers_validating);

  m_num_parallel_readers_testing = compute_max_num_parallel_readers(testing_data_set_size, mini_batch_size, m_num_parallel_readers_testing);
}

int lbann::distributed_minibatch::fetch_to_local_matrix(Mat& M_local) {
  int num_parallel_readers = get_num_parallel_readers();

  /// Check to see if this rank has valid data -- if not read in the next batch
  /// Coordinate all available readers so that the perform I/O in the same step
  if (m_root == 0) {
    if (m_comm->get_rank_in_model() < num_parallel_readers && !m_local_reader_done) {
      Zero(M_local);

      /// Each data reader needs to either have independent / split
      /// data, or take an offset / stride
      m_num_samples_in_batch = fetch_from_data_reader(M_local);
      bool data_valid = (m_num_samples_in_batch > 0);
      if(data_valid) {
        m_num_data_per_epoch+=m_num_samples_in_batch;
        preprocess_data_samples(M_local, m_num_samples_in_batch);
      }
      m_local_data_valid = data_valid;
    }
    m_num_valid_readers = m_comm->model_allreduce((int) m_local_data_valid, mpi::SUM);
  }
  return m_num_samples_in_batch;
}

void lbann::distributed_minibatch::distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {
  int num_parallel_readers = m_num_valid_readers;
  Ms.SetRoot(m_root);

  m_comm->model_barrier();

  if (m_comm->get_rank_in_model() == m_root) {
    if(!m_local_data_valid) {
      stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: lbann_distributed_minibatch: No valid data for this step -- local data was invalid";
      lbann_exception(err.str());
    }
    CopyFromRoot(M_local, Ms);
    m_local_data_valid = false;
    m_num_samples_in_batch = 0;
  } else {
    CopyFromNonRoot(Ms);
  }

  m_comm->model_barrier();

  m_root = (m_root + 1) % num_parallel_readers;
  return;
}

bool lbann::distributed_minibatch::is_data_set_processed() {
  int num_readers_done = 0;
  int max_active_parallel_readers = get_num_parallel_readers();  // When calculating if all parallel readers are done, include the maximum number,
  // not just the ones in the last round.  This will ensure that all readers, that had data
  // will have distributed it.
  int num_parallel_readers = m_num_valid_readers;

  if(m_comm->get_rank_in_model() < num_parallel_readers) {
    if((m_comm->get_rank_in_model()+1)%num_parallel_readers == m_root) {
      if(m_local_data_valid) { /// Make sure that all local data has been processed
        stringstream err;
        err << __FILE__ << " "<<  __LINE__
            << " :: lbann_input_layer_distributed_minibatch: all valid data was not processed.";
        throw lbann_exception(err.str());
      }
      m_local_reader_done = !update_data_reader();
    }
  }

  /// Set the reduction variable
  if(m_local_reader_done) {
    num_readers_done = 1;
  }

  /// Once all of the readers have finished their part of the mini-batch indicate that the epoch is finished
  num_readers_done = m_comm->model_allreduce(num_readers_done);
  if(num_readers_done == max_active_parallel_readers) {
    m_local_reader_done = false;
    m_root = 0; /// When the epoch is finished, make sure that the root node for distributing data is reset because
    /// if the number of parallel readers does not evenly divide the data set size, the epoch will finish
    /// without all of the parallel readers participating in the last round.
    m_num_data_per_epoch = 0;
    return true;
  } else {
    return false;
  }
}

int lbann::distributed_minibatch::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int num_parallel_readers) {
  /// Check to make sure that there is enough data for all of the parallel readers
  if(data_set_size != 0) {
    int max_num_parallel_readers = num_parallel_readers;
    while(ceil((float)data_set_size / (float)(mini_batch_size * m_comm->get_num_models())) < max_num_parallel_readers) {
      max_num_parallel_readers--;
    }
    if(max_num_parallel_readers != num_parallel_readers) {
      std::cout << "Warning the training data set size " << data_set_size
                << " is too small for the number of requested parallel readers "
                << num_parallel_readers << ", using " << max_num_parallel_readers << "."
                << std::endl;
    }
    return max_num_parallel_readers;
  } else {
    return 0;
  }
}

void lbann::distributed_minibatch::calculate_num_iterations_per_epoch(generic_data_reader *data_reader) {
  int max_mini_batch_size = data_reader->getm_batch_max();
  int num_parallel_readers_per_model = max(1, (data_reader->get_batch_stride() / m_comm->get_num_models()) / max_mini_batch_size);
  int min_stride_across_models = max_mini_batch_size * m_comm->get_num_models();  /// Given that each model has to have at least one reader, what is the minimum stride

  data_reader->set_last_mini_batch_size(max_mini_batch_size); /// By default the last mini-batch is a full one

  int num_whole_mini_batches_per_model = floor(data_reader->getNumData() / min_stride_across_models);
  int num_whole_mini_batches_per_reader = floor(num_whole_mini_batches_per_model / num_parallel_readers_per_model);
  int parallel_readers_with_extra_mini_batch = num_whole_mini_batches_per_model % num_parallel_readers_per_model;
  int per_model_partial_mini_batch_size = (data_reader->getNumData() - (num_whole_mini_batches_per_model * min_stride_across_models))/(m_comm->get_num_models());
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  data_reader->set_last_mini_batch_threshold(num_whole_mini_batches_per_model * min_stride_across_models);

  if(m_comm->get_rank_in_model() < parallel_readers_with_extra_mini_batch) {
    num_whole_mini_batches_per_reader += 1;
    data_reader->set_last_mini_batch_size(max_mini_batch_size);
  }

  data_reader->set_num_mini_batches_per_reader(num_whole_mini_batches_per_reader);

  int world_master_remainder_adjustment = data_reader->getNumData()
                                          - (num_whole_mini_batches_per_model * min_stride_across_models)
                                          - (per_model_partial_mini_batch_size * m_comm->get_num_models());
  if(m_comm->am_world_master()) {
    world_master_remainder_data = world_master_remainder_adjustment;
    world_master_remainder_adjustment = 0;
  }
  per_model_partial_mini_batch_size += world_master_remainder_data;

  /// The first reader that doesn't have an extra mini batch gets the partial batch
  if(m_comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch && per_model_partial_mini_batch_size > 0) {
    data_reader->set_num_mini_batches_per_reader(data_reader->get_num_mini_batches_per_reader()+1);
    data_reader->set_last_mini_batch_size(per_model_partial_mini_batch_size);
  }

  if(data_reader->get_last_mini_batch_size() > max_mini_batch_size) {
    throw new lbann_exception("Error in calculating the partial mini-batch size, exceeds the max mini-batch size");
  }

  /// Note that m_comm->get_model_rank() + m_comm->get_rank_in_model() is not equivalent to m_comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers model rank, how many models have a higher rank

  /// By default the last stride of each reader is part of a regular (full) round
  data_reader->set_last_mini_batch_stride(data_reader->get_batch_stride());

  int last_mini_batch_offset = max(0, num_whole_mini_batches_per_reader - 1) * data_reader->get_batch_stride();

  ///  The last mini-batch may be partial and thus may have a smaller stride
  if(m_comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch && per_model_partial_mini_batch_size > 0) {
    data_reader->set_last_mini_batch_stride((data_reader->get_last_mini_batch_threshold() - data_reader->get_base_offset() - data_reader->get_model_offset() - last_mini_batch_offset)
                                            + m_comm->get_model_rank() * per_model_partial_mini_batch_size + world_master_remainder_adjustment); /// BVE 10/18/16
  }

  //cout << "[" << m_comm->get_rank_in_world() << "] " << m_comm->get_model_rank() << " model rank, "<< m_comm->get_rank_in_model() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << "(m_num_mini_batches_per_reader=" << data_reader->get_num_mini_batches_per_reader() << ") parallel_readers_with_extra_mini_batch " << parallel_readers_with_extra_mini_batch << " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << data_reader->get_last_mini_batch_threshold() << " with a last stride of " << data_reader->get_last_mini_batch_stride() << " and stride of " << data_reader->get_batch_stride() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << parallel_readers_with_extra_mini_batch << " model bracket = " << (parallel_readers_with_extra_mini_batch * max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;

  return;
}
