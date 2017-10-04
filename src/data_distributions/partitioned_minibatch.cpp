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

lbann::partitioned_minibatch::partitioned_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
  : generic_data_distribution(comm, num_parallel_readers, data_readers) {}

int lbann::partitioned_minibatch::fetch_to_local_matrix(Mat& M_local) {
  int num_parallel_readers = get_num_parallel_readers();

  int num_samples_fetched = 0;

  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  if (m_comm->get_rank_in_model() < num_parallel_readers && (M_local.Height() != 0 && M_local.Width() != 0) && !m_local_reader_done) {
    Zero(M_local);

    /// Each data reader needs to either have independent / split
    /// data, or take an offset / stride
    num_samples_fetched = fetch_from_data_reader(M_local);
    bool data_valid = (num_samples_fetched > 0);
    if(data_valid) {
      m_num_data_per_epoch+=num_samples_fetched; /// BVE FIXME need to change how this is shared
      preprocess_data_samples(M_local, num_samples_fetched);
    }
    m_local_data_valid = data_valid;
  }
  return num_samples_fetched;
}

void lbann::partitioned_minibatch::distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {

  /// Nothing to do here, it is already done
  return;
}

bool lbann::partitioned_minibatch::is_data_set_processed() {
  int num_iterations_per_epoch = get_num_iterations_per_epoch();

  m_local_reader_done = !update_data_reader(true);

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

int lbann::partitioned_minibatch::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) {
  int num_parallel_readers = requested_num_parallel_readers;

  if(m_comm->get_procs_per_model() != num_parallel_readers) {
    if (m_comm->am_model_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " does not match the grid size "
                << m_comm->get_procs_per_model()
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = m_comm->get_procs_per_model();
  }

  if(mini_batch_size < num_parallel_readers) {
    if (m_comm->am_model_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " is larger than the requested mini-batch size "
                << mini_batch_size
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = mini_batch_size;
  }

  return num_parallel_readers;
}

void lbann::partitioned_minibatch::calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) {
  if(data_reader == NULL) { return; }
  // If the data reader does not have any data bail out (e.g. unused validation reader)
  if(data_reader->get_use_percent() == double(0.0)) { return; }

  /// Make sure that the mini-batch size is not larger than the data set
  if(max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel readers
  int num_parallel_readers_per_model = compute_max_num_parallel_readers(data_reader->get_num_data(), max_mini_batch_size, m_requested_max_num_parallel_readers);
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if(num_parallel_readers_per_model == 0 
     || (num_parallel_readers_per_model != m_comm->get_procs_per_model() && num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: partitioned_minibatch: number of parallel readers is " + std::to_string(num_parallel_readers_per_model)
      + " and there are " + std::to_string(m_comm->get_procs_per_model()) + " processes in the model");
  }

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = m_comm->get_num_models() * max_mini_batch_size;
  int base_offset  = m_comm->get_rank_in_model();
  int model_offset = m_comm->get_model_rank() * max_mini_batch_size;
  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
  data_reader->set_sample_stride(num_parallel_readers_per_model);
  data_reader->set_iteration_stride(1);
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(model_offset);
  data_reader->set_initial_position();

  int min_stride_across_models = max_mini_batch_size * m_comm->get_num_models();  /// Given that each model has to have at least one reader, what is the minimum stride

  data_reader->set_global_mini_batch_size(min_stride_across_models); /// The global mini-batch is a full mini-batch per model

  data_reader->set_last_mini_batch_size(max_mini_batch_size); /// By default the last mini-batch is a full one
  data_reader->set_global_last_mini_batch_size(min_stride_across_models); /// By default the last mini-batch is a full one per model

  int num_whole_mini_batches_per_model = floor(data_reader->get_num_data() / min_stride_across_models);
  int global_partial_mini_batch_size = data_reader->get_num_data() - (num_whole_mini_batches_per_model * min_stride_across_models);
  int per_model_partial_mini_batch_size = global_partial_mini_batch_size / m_comm->get_num_models();
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  int last_mini_batch_threshold = num_whole_mini_batches_per_model * min_stride_across_models;

  int world_master_remainder_adjustment = data_reader->get_num_data()
                                          - (num_whole_mini_batches_per_model * min_stride_across_models)
                                          - (per_model_partial_mini_batch_size * m_comm->get_num_models());
  if(m_comm->get_model_rank() == 0) {
    world_master_remainder_data = world_master_remainder_adjustment;
    world_master_remainder_adjustment = 0;
  }
  per_model_partial_mini_batch_size += world_master_remainder_data;

  if(per_model_partial_mini_batch_size > 0 || world_master_remainder_adjustment > 0) {
    data_reader->set_last_mini_batch_size(per_model_partial_mini_batch_size);
    data_reader->set_global_last_mini_batch_size(global_partial_mini_batch_size);
  }

  if(global_partial_mini_batch_size != 0) {
    data_reader->set_num_iterations_per_epoch(num_whole_mini_batches_per_model+1);
  }else {
    data_reader->set_num_iterations_per_epoch(num_whole_mini_batches_per_model);
  }

  if(data_reader->get_last_mini_batch_size() > max_mini_batch_size) {
    throw new lbann_exception("Error in calculating the partial mini-batch size, exceeds the max mini-batch size");
  }

  /// Note that m_comm->get_model_rank() + m_comm->get_rank_in_model() is not equivalent to m_comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers model rank, how many models have a higher rank

  /// By default the last stride of each reader is part of a regular (full) round
  data_reader->set_stride_to_last_mini_batch(data_reader->get_stride_to_next_mini_batch());

  /// BVE FIXME - I feel like this is wrong  I don't think that the -1
  /// should be there
  int last_mini_batch_offset = max(0, num_whole_mini_batches_per_model - 1) * data_reader->get_stride_to_next_mini_batch();

  ///  The last mini-batch may be partial and thus may have a smaller stride
  if(per_model_partial_mini_batch_size > 0 || world_master_remainder_adjustment > 0) {
    data_reader->set_stride_to_last_mini_batch((last_mini_batch_threshold - data_reader->get_base_offset() - data_reader->get_model_offset() - last_mini_batch_offset) + m_comm->get_model_rank() * per_model_partial_mini_batch_size + m_comm->get_rank_in_model());
  }

  //  cout << "[" << m_comm->get_rank_in_world() << "] " << m_comm->get_model_rank() << " model rank, "<< m_comm->get_rank_in_model() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " parallel_readers_with_extra_mini_batch " << /*parallel_readers_with_extra_mini_batch <<*/ " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " with a last stride of " << data_reader->get_stride_to_last_mini_batch() << " and stride of " << data_reader->get_stride_to_next_mini_batch() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << /*parallel_readers_with_extra_mini_batch << */" model bracket = " << (/*parallel_readers_with_extra_mini_batch **/ max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;
//cout << "[" << m_comm->get_rank_in_world() << "] " << m_comm->get_model_rank() << " model rank, "<< m_comm->get_rank_in_model() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << "(m_num_mini_batches_per_reader=" << data_reader->get_num_mini_batches_per_reader() << ") parallel_readers_with_extra_mini_batch " << /*parallel_readers_with_extra_mini_batch <<*/ " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << data_reader->get_last_mini_batch_threshold() << " with a last stride of " << data_reader->get_stride_to_last_mini_batch() << " and stride of " << data_reader->get_batch_stride() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << /*parallel_readers_with_extra_mini_batch << */" model bracket = " << (/*parallel_readers_with_extra_mini_batch **/ max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;
  return;
}

void lbann::partitioned_minibatch::calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) {
  if(data_reader == NULL) { return; }
  // If the data reader does not have any data bail out (e.g. unused validation reader)
  if(data_reader->get_use_percent() == double(0.0)) { return; }

  if(max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel readers
  int num_parallel_readers_per_model = compute_max_num_parallel_readers(data_reader->get_num_data(), max_mini_batch_size, m_requested_max_num_parallel_readers);
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if(num_parallel_readers_per_model == 0
     || (num_parallel_readers_per_model != m_comm->get_procs_per_model() && num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_distribution: number of parallel readers is zero");
  }

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = max_mini_batch_size;
  int base_offset  = m_comm->get_rank_in_model();
  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
  data_reader->set_sample_stride(num_parallel_readers_per_model);
  data_reader->set_iteration_stride(1);
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(0);
  data_reader->set_initial_position();

  /// By default each data reader will plan to process the entire data set
  int num_iterations_per_epoch = ceil((float) data_reader->get_num_data() / (float) max_mini_batch_size);
  int last_mini_batch_size = data_reader->get_num_data() % max_mini_batch_size;
  if(last_mini_batch_size == 0) {
    last_mini_batch_size = max_mini_batch_size;
  }
  data_reader->set_num_iterations_per_epoch(num_iterations_per_epoch);
  data_reader->set_last_mini_batch_size(last_mini_batch_size);
  data_reader->set_stride_to_last_mini_batch(data_reader->get_stride_to_next_mini_batch());

  data_reader->set_global_mini_batch_size(max_mini_batch_size);
  data_reader->set_global_last_mini_batch_size(last_mini_batch_size);
  return;
}
