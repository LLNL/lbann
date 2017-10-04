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

lbann::distributed_minibatch::distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
  : generic_data_distribution(comm, num_parallel_readers, data_readers) {}

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
  }
  return m_num_samples_in_batch;
}

void lbann::distributed_minibatch::distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {
  int num_parallel_readers = get_num_parallel_readers();
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
  // not just the ones in the last round.  This will ensure that all readers, that had data
  // will have distributed it.
  int num_parallel_readers = get_num_parallel_readers();
  int num_iterations_per_epoch = get_num_iterations_per_epoch();

  bool is_active_reader = (m_comm->get_rank_in_model() < num_parallel_readers) 
    && ((m_comm->get_rank_in_model()+1)%num_parallel_readers == m_root);

  if(is_active_reader) {
      if(m_local_data_valid) { /// Make sure that all local data has been processed
        stringstream err;
        err << __FILE__ << " "<<  __LINE__
            << " :: lbann_input_layer_distributed_minibatch: all valid data was not processed.";
        throw lbann_exception(err.str());
      }
  }
  m_local_reader_done = !update_data_reader(is_active_reader);

  /// Once all of the readers have finished their part of the mini-batch indicate that the epoch is finished
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

/** Make sure that there are enough ranks and data for all of the
 *  parallel readers requested.
 */
int lbann::distributed_minibatch::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) {
  int num_parallel_readers = requested_num_parallel_readers;

  /// Are there enough ranks in the model to support the requested
  /// number of parallel readers
  if(m_comm->get_model_grid().Size() < num_parallel_readers) {
    if (m_comm->am_model_master()) {
      cout << "Warning the grid size "<<m_comm->get_model_grid().Size()
           <<"is smaller than the number of requested parallel readers "
           <<num_parallel_readers<<"." << endl;
    }
    num_parallel_readers = m_comm->get_model_grid().Size();
  }

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

void lbann::distributed_minibatch::calculate_num_iterations_per_epoch(int num_models, int model_rank, int max_mini_batch_size, generic_data_reader *data_reader) {
  if(data_reader == NULL) { return; }
  // If the data reader does not have any data bail out (e.g. unused validation reader)
  if(data_reader->get_use_percent() == double(0.0)) { return; }

  if(max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel readers
  int num_parallel_readers_per_model = compute_max_num_parallel_readers(data_reader->get_num_data(), max_mini_batch_size, m_requested_max_num_parallel_readers);
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if(num_parallel_readers_per_model == 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: distributed_minibatch: number of parallel readers is zero");
  }

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = num_models * num_parallel_readers_per_model * max_mini_batch_size;
  int base_offset = m_comm->get_rank_in_model() * num_models * max_mini_batch_size;
  int model_offset = model_rank * max_mini_batch_size;
  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
  data_reader->set_sample_stride(1);
  data_reader->set_iteration_stride(num_parallel_readers_per_model);
  data_reader->set_reset_mini_batch_index(m_comm->get_rank_in_model());
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(model_offset);
  data_reader->set_initial_position();

  int min_stride_across_models = max_mini_batch_size * num_models;  /// Given that each model has to have at least one reader, what is the minimum stride

  data_reader->set_global_mini_batch_size(min_stride_across_models); /// The global mini-batch is a full mini-batch per model
  data_reader->set_last_mini_batch_size(max_mini_batch_size); /// By default the last mini-batch is a full one
  data_reader->set_global_last_mini_batch_size(min_stride_across_models); /// By default the last mini-batch is a full one per model

  int num_whole_mini_batches_per_model = floor(data_reader->get_num_data() / min_stride_across_models);
  int num_whole_mini_batches_per_reader = floor(num_whole_mini_batches_per_model / num_parallel_readers_per_model);
  int parallel_readers_with_extra_mini_batch = num_whole_mini_batches_per_model % num_parallel_readers_per_model;
  int global_partial_mini_batch_size = data_reader->get_num_data() - (num_whole_mini_batches_per_model * min_stride_across_models);
  int per_model_partial_mini_batch_size = global_partial_mini_batch_size / num_models;
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  int last_mini_batch_threshold = num_whole_mini_batches_per_model * min_stride_across_models;

  // BVE FIXME revisit this piece of code
  if(m_comm->get_rank_in_model() < parallel_readers_with_extra_mini_batch) {
    num_whole_mini_batches_per_reader += 1;
    data_reader->set_last_mini_batch_size(max_mini_batch_size);
  }

  int world_master_remainder_adjustment = data_reader->get_num_data()
                                          - (num_whole_mini_batches_per_model * min_stride_across_models)
                                          - (per_model_partial_mini_batch_size * num_models);
  if(m_comm->am_world_master()) {
    world_master_remainder_data = world_master_remainder_adjustment;
    world_master_remainder_adjustment = 0;
  }
  per_model_partial_mini_batch_size += world_master_remainder_data;

  /// If there is a partial mini-batch all readers need to know about it
  if(per_model_partial_mini_batch_size > 0) {
    data_reader->set_last_mini_batch_size(per_model_partial_mini_batch_size);
    data_reader->set_global_last_mini_batch_size(global_partial_mini_batch_size);
  }

  // BVE FIXME this is wonky
  if(global_partial_mini_batch_size != 0) {
    data_reader->set_num_iterations_per_epoch(num_whole_mini_batches_per_model+1);
  }else {
    data_reader->set_num_iterations_per_epoch(num_whole_mini_batches_per_model);
  }

  if(data_reader->get_last_mini_batch_size() > max_mini_batch_size) {
    throw new lbann_exception("Error in calculating the partial mini-batch size, exceeds the max mini-batch size");
  }

  /// Note that model_rank + m_comm->get_rank_in_model() is not equivalent to m_comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers model rank, how many models have a higher rank

  /// By default the last stride of each reader is part of a regular (full) round
  data_reader->set_stride_to_last_mini_batch(data_reader->get_stride_to_next_mini_batch());

  int last_mini_batch_offset = max(0, num_whole_mini_batches_per_reader - 1) * data_reader->get_stride_to_next_mini_batch();

  /// BVE FIXME I don't think that the last mb stride is correct
  ///  The last mini-batch may be partial and thus may have a smaller stride
  if(m_comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch && per_model_partial_mini_batch_size > 0) {
    data_reader->set_stride_to_last_mini_batch((last_mini_batch_threshold - data_reader->get_base_offset() - data_reader->get_model_offset() - last_mini_batch_offset)
                                            + model_rank * per_model_partial_mini_batch_size + world_master_remainder_adjustment); /// BVE 10/18/16
  }

  //cout << "[" << m_comm->get_rank_in_world() << "] " << model_rank << " model rank, "<< m_comm->get_rank_in_model() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << "(m_num_mini_batches_per_reader=" << data_reader->get_num_mini_batches_per_reader() << ") parallel_readers_with_extra_mini_batch " << parallel_readers_with_extra_mini_batch << " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << data_reader->get_last_mini_batch_threshold() << " with a last stride of " << data_reader->get_stride_to_last_mini_batch() << " and stride of " << data_reader->get_stride_to_next_mini_batch() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << parallel_readers_with_extra_mini_batch << " model bracket = " << (parallel_readers_with_extra_mini_batch * max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;

  return;
}

void lbann::distributed_minibatch::calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) {
  calculate_num_iterations_per_epoch(m_comm->get_num_models(), m_comm->get_model_rank(), max_mini_batch_size, data_reader);
}

void lbann::distributed_minibatch::calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) {
  calculate_num_iterations_per_epoch(1, 0, max_mini_batch_size, data_reader);
}
