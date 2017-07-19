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

#include "lbann/data_distributions/partitioned_minibatch.hpp"
#include "lbann/utils/exception.hpp"

using namespace std;

lbann::partitioned_minibatch::partitioned_minibatch(lbann_comm *comm, int num_parallel_readers, int mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers)
  : generic_data_distribution(comm, num_parallel_readers, mini_batch_size, data_readers) { 

  if(m_comm->get_model_grid().Size() != num_parallel_readers) {
    cout << "Warning the requested number of parallel readers "
         << num_parallel_readers
         << " does not match the grid size " << m_comm->get_model_grid().Size()
         << " OVERRIDING requested number of parallel readers."
         << endl;
    m_num_parallel_readers_training = m_comm->get_model_grid().Size();
    m_num_parallel_readers_validating = m_comm->get_model_grid().Size();
    m_num_parallel_readers_testing = m_comm->get_model_grid().Size();
    num_parallel_readers = m_comm->get_model_grid().Size();
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
}

int lbann::partitioned_minibatch::fetch_to_local_matrix(Mat& M_local) {
  int num_parallel_readers = get_num_parallel_readers();

  m_num_samples_in_batch = 0;

  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  if (m_comm->get_rank_in_model() < num_parallel_readers && (M_local.Height() != 0 && M_local.Width() != 0) && !m_local_reader_done) {
    Zero(M_local);

    /// Each data reader needs to either have independent / split
    /// data, or take an offset / stride
    m_num_samples_in_batch = fetch_from_data_reader(M_local);
    bool data_valid = (m_num_samples_in_batch > 0);
    if(data_valid) {
      m_num_data_per_epoch+=m_num_samples_in_batch; /// BVE FIXME need to change how this is shared
      preprocess_data_samples(M_local, m_num_samples_in_batch);
    }
    m_local_data_valid = data_valid;
  }
  m_num_valid_readers = m_comm->model_allreduce((int) m_local_data_valid, mpi::SUM); /// BVE FIXME I don't think that we need this any more
  m_num_samples_in_batch = m_comm->model_allreduce((int) m_num_samples_in_batch, mpi::SUM); /// @todo compute this by dead reckoning to avoid allreduce
  return m_num_samples_in_batch;
}

void lbann::partitioned_minibatch::distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {

  /// Nothing to do here, it is already done
  return;
}

bool lbann::partitioned_minibatch::is_data_set_processed() {
  int num_readers_done = 0;
  int num_iterations_per_epoch = get_num_iterations_per_epoch();

  m_local_reader_done = !update_data_reader();

  /// Set the reduction variable
  if(m_local_reader_done) {
    num_readers_done = 1;
  }

  if(m_cur_step_in_epoch == (num_iterations_per_epoch - 1)) {
    m_local_reader_done = false;
    m_root = 0; /// When the epoch is finished, make sure that the root node for distributing data is reset because
    /// if the number of parallel readers does not evenly divide the data set size, the epoch will finish
    /// without all of the parallel readers participating in the last round.
    m_num_data_per_epoch = 0;
    m_cur_step_in_epoch = 0;
    return true;
  } else {
    m_cur_step_in_epoch++;
    return false;
  }
}

void lbann::partitioned_minibatch::calculate_num_iterations_per_epoch(generic_data_reader *data_reader) {
  int max_mini_batch_size = data_reader->getm_batch_max();
  int num_parallel_readers_per_model = max(1, (data_reader->get_batch_stride() / m_comm->get_num_models()) / max_mini_batch_size);
  int min_stride_across_models = max_mini_batch_size * m_comm->get_num_models();  /// Given that each model has to have at least one reader, what is the minimum stride

  data_reader->set_last_mini_batch_size(max_mini_batch_size); /// By default the last mini-batch is a full one

  int num_whole_mini_batches_per_model = floor(data_reader->getNumData() / min_stride_across_models);
  int num_whole_mini_batches_per_reader = floor(num_whole_mini_batches_per_model / num_parallel_readers_per_model);
  //  int parallel_readers_with_extra_mini_batch = num_whole_mini_batches_per_model % num_parallel_readers_per_model;
  int per_model_partial_mini_batch_size = (data_reader->getNumData() - (num_whole_mini_batches_per_model * min_stride_across_models))/(m_comm->get_num_models());
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  data_reader->set_last_mini_batch_threshold(num_whole_mini_batches_per_model * min_stride_across_models);

  data_reader->set_num_mini_batches_per_reader(num_whole_mini_batches_per_reader);

  int world_master_remainder_adjustment = data_reader->getNumData()
                                          - (num_whole_mini_batches_per_model * min_stride_across_models)
                                          - (per_model_partial_mini_batch_size * m_comm->get_num_models());
  if(m_comm->am_world_master()) {
    world_master_remainder_data = world_master_remainder_adjustment;
    world_master_remainder_adjustment = 0;
  }
  per_model_partial_mini_batch_size += world_master_remainder_data;

  if(per_model_partial_mini_batch_size > 0 || world_master_remainder_adjustment > 0) {
    data_reader->set_num_mini_batches_per_reader(data_reader->get_num_mini_batches_per_reader()+1);
    data_reader->set_last_mini_batch_size(per_model_partial_mini_batch_size);
  }

  data_reader->set_num_iterations_per_epoch(data_reader->get_num_mini_batches_per_reader());

  if(data_reader->get_last_mini_batch_size() > max_mini_batch_size) {
    throw new lbann_exception("Error in calculating the partial mini-batch size, exceeds the max mini-batch size");
  }

  /// Note that m_comm->get_model_rank() + m_comm->get_rank_in_model() is not equivalent to m_comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers model rank, how many models have a higher rank

  /// By default the last stride of each reader is part of a regular (full) round
  data_reader->set_last_mini_batch_stride(data_reader->get_batch_stride());

  int last_mini_batch_offset = max(0, num_whole_mini_batches_per_reader - 1) * data_reader->get_batch_stride();

  ///  The last mini-batch may be partial and thus may have a smaller stride
  if(per_model_partial_mini_batch_size > 0 || world_master_remainder_adjustment > 0) {
    data_reader->set_last_mini_batch_stride((data_reader->get_last_mini_batch_threshold() - data_reader->get_base_offset() - data_reader->get_model_offset() - last_mini_batch_offset) + m_comm->get_model_rank() * per_model_partial_mini_batch_size + m_comm->get_rank_in_model());
  }

  //cout << "[" << m_comm->get_rank_in_world() << "] " << m_comm->get_model_rank() << " model rank, "<< m_comm->get_rank_in_model() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << "(m_num_mini_batches_per_reader=" << data_reader->get_num_mini_batches_per_reader() << ") parallel_readers_with_extra_mini_batch " << /*parallel_readers_with_extra_mini_batch <<*/ " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << data_reader->get_last_mini_batch_threshold() << " with a last stride of " << data_reader->get_last_mini_batch_stride() << " and stride of " << data_reader->get_batch_stride() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << /*parallel_readers_with_extra_mini_batch << */" model bracket = " << (/*parallel_readers_with_extra_mini_batch **/ max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;
  return;
}
