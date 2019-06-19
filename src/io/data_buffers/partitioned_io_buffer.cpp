////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/utils/exception.hpp"

lbann::partitioned_io_buffer::partitioned_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers, int num_child_layers)
  : generic_io_buffer(comm, num_parallel_readers, data_readers) {
  m_data_buffers[execution_mode::training] = new data_buffer(comm, num_child_layers);
  m_data_buffers[execution_mode::validation] = new data_buffer(comm, num_child_layers);
  m_data_buffers[execution_mode::testing] = new data_buffer(comm, num_child_layers);
}

lbann::partitioned_io_buffer::~partitioned_io_buffer() {
  for (auto& buf : m_data_buffers) {
    delete buf.second;
  }
}

lbann::partitioned_io_buffer::partitioned_io_buffer(const lbann::partitioned_io_buffer& other)
  : generic_io_buffer(other) {
  for (const auto& buf : other.m_data_buffers) {
    m_data_buffers[buf.first] = buf.second->copy();
  }
}

lbann::partitioned_io_buffer* lbann::partitioned_io_buffer::copy() const {
  return new partitioned_io_buffer(*this);
}

lbann::partitioned_io_buffer& lbann::partitioned_io_buffer::operator=(const lbann::partitioned_io_buffer& other) {
  generic_io_buffer::operator=(other);
  for (auto& buf : m_data_buffers) {
    if (buf.second) delete buf.second;
    buf.second = buf.second->copy();
  }
  return *this;
}

void lbann::partitioned_io_buffer::fp_setup_data(El::Int cur_mini_batch_size, int idx) {
  for (auto& buf : m_data_buffers) {
    buf.second->m_input_buffers[idx]->Resize(buf.second->m_input_buffers[idx]->Height(), cur_mini_batch_size);
  }
}

void lbann::partitioned_io_buffer::setup_data(El::Int num_neurons, El::Int num_targets, El::Int max_mini_batch_size) {
  El::Int local_mini_batch_size = max_mini_batch_size / m_comm->get_procs_per_trainer();
  El::Int partial_mini_batch_size = max_mini_batch_size % m_comm->get_procs_per_trainer();
  if(partial_mini_batch_size > 0 && m_comm->get_rank_in_trainer() < partial_mini_batch_size) {
    local_mini_batch_size++;
  }
  for (const auto& it : m_data_buffers) {
    data_buffer *data_buffer = it.second;
    int i = 0;
    for (const auto& buf : data_buffer->m_input_buffers) {
      if(i == 0) {
        buf->Resize(num_neurons, max_mini_batch_size);
      }else if(i == 1) {
        buf->Resize(num_targets, max_mini_batch_size);
      }else {
        LBANN_ERROR("Unsupported number of input channels");
      }
      i++;
    }
    /// The amount of space needed will vary based on input layer type,
    /// but the batch size is the maximum space necessary
    El::Zeros_seq(data_buffer->m_indices_fetched_per_mb, local_mini_batch_size, 1);
  }
}

int lbann::partitioned_io_buffer::fetch_to_local_matrix(generic_data_reader *data_reader, execution_mode mode) {
  int num_parallel_readers = data_reader->get_num_parallel_readers();

  /// Coordinate all available readers so that the perform I/O in the same step
  /// Check to make sure that the local matrix has space for data
  data_buffer *buf = get_data_buffer(mode);
  buf->m_num_samples_fetched = 0;
  if (m_comm->get_rank_in_trainer() < num_parallel_readers && (buf->m_input_buffers[0]->Height() != 0 && buf->m_input_buffers[0]->Width() != 0)) {
    for(auto& m : buf->m_input_buffers) {
      El::Zeros_seq(*m, m->Height(), m->Width());
    }

    /// Each data reader needs to either have independent / split
    /// data, or take an offset / stride
    if(buf->m_input_buffers.size() == 2) {
      buf->m_num_samples_fetched = (*fetch_data_fn)(buf->m_input_buffers[0]->Matrix(), buf->m_input_buffers[1]->Matrix(), buf->m_indices_fetched_per_mb, data_reader);
    }else {
      buf->m_num_samples_fetched = (*fetch_data_fn)(buf->m_input_buffers[0]->Matrix(), buf->m_indices_fetched_per_mb, data_reader);
    }
    bool data_valid = (buf->m_num_samples_fetched > 0);
    if(data_valid) {
      //      m_num_data_per_epoch+=num_samples_fetched; /// BVE FIXME need to change how this is shared
    }
  }
  return buf->m_num_samples_fetched;
}

void lbann::partitioned_io_buffer::distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample, AbsDistMat& response) {
  data_buffer *buf = get_data_buffer(mode);
  Copy(*buf->m_input_buffers[0], sample);
  Copy(*buf->m_input_buffers[1], response);
  buf->m_num_samples_fetched = 0;
  return;
}

void lbann::partitioned_io_buffer::distribute_from_local_matrix(generic_data_reader *data_reader, execution_mode mode, AbsDistMat& sample) {
  data_buffer *buf = get_data_buffer(mode);
  Copy(*buf->m_input_buffers[0], sample);
  buf->m_num_samples_fetched = 0;
  return;
}

bool lbann::partitioned_io_buffer::update_data_set(generic_data_reader *data_reader, execution_mode mode) {
  int num_iterations_per_epoch = data_reader->get_num_iterations_per_epoch();
  int current_step_in_epoch = data_reader->get_current_step_in_epoch(); // Get the current step before the update function increments it

  (*update_data_reader_fn)(true, data_reader);

  if(current_step_in_epoch == (num_iterations_per_epoch - 1)) {
    return true;
  } else {
    return false;
  }
}

void lbann::partitioned_io_buffer::set_fetch_data_in_background(bool flag, execution_mode mode) {
  data_buffer *buf = get_data_buffer(mode);
  buf->m_fetch_data_in_background = flag;
}

bool lbann::partitioned_io_buffer::is_data_fetched_in_background(execution_mode mode) {
  data_buffer *buf = get_data_buffer(mode);
  return buf->m_fetch_data_in_background;
}

/**
 * Return the sample indices fetched in the current mini-batch.
 */
El::Matrix<El::Int>* lbann::partitioned_io_buffer::get_sample_indices_fetched_per_mb(execution_mode mode) {
  data_buffer *buf = get_data_buffer(mode);
  return &(buf->m_indices_fetched_per_mb);
}

int lbann::partitioned_io_buffer::num_samples_ready(execution_mode mode) {
  data_buffer *buf = get_data_buffer(mode);
  return buf->m_num_samples_fetched;
}

void lbann::partitioned_io_buffer::set_data_fetch_future(std::future<void> future, execution_mode mode) {
  data_buffer *buf = get_data_buffer(mode);
  buf->m_data_fetch_future = std::move(future);
}

std::future<void> lbann::partitioned_io_buffer::get_data_fetch_future(execution_mode mode) {
  data_buffer *buf = get_data_buffer(mode);
  return std::move(buf->m_data_fetch_future);
}

int lbann::partitioned_io_buffer::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const {
  return partitioned_io_buffer::compute_max_num_parallel_readers(data_set_size, mini_batch_size, requested_num_parallel_readers, m_comm);
}

int lbann::partitioned_io_buffer::compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers, const lbann_comm* comm) {
  int num_parallel_readers = requested_num_parallel_readers;

  if(comm->get_procs_per_trainer() != num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " does not match the grid size "
                << comm->get_procs_per_trainer()
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = comm->get_procs_per_trainer();
  }

#if 0
  if(mini_batch_size < num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " is larger than the requested mini-batch size "
                << mini_batch_size
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = mini_batch_size;
  }
#endif
  return num_parallel_readers;
}

void lbann::partitioned_io_buffer::calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) {
  if(data_reader == nullptr) { return; }
  // If the data reader does not have any data bail out (e.g. unused validation reader)
  if(data_reader->get_num_data() == 0) { return; }

  /// Make sure that the mini-batch size is not larger than the data set
  if(max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  bool apportioned = data_reader->is_partitioned();

  /// Check to make sure that there is enough data for all of the parallel readers
  int num_parallel_readers_per_model = compute_max_num_parallel_readers(data_reader->get_num_data(), max_mini_batch_size, m_comm->get_procs_per_trainer());
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if(num_parallel_readers_per_model == 0
     || (num_parallel_readers_per_model != m_comm->get_procs_per_trainer() && num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: partitioned_io_buffer: number of parallel readers is " + std::to_string(num_parallel_readers_per_model)
      + " and there are " + std::to_string(m_comm->get_procs_per_trainer()) + " processes in the model");
  }

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = m_comm->get_num_trainers() * max_mini_batch_size;
  int base_offset  = m_comm->get_rank_in_trainer();
  int model_offset = m_comm->get_trainer_rank() * max_mini_batch_size;

  if (apportioned) {
    batch_stride = max_mini_batch_size;
    model_offset = 0;
  }

  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
  data_reader->set_sample_stride(num_parallel_readers_per_model);
  data_reader->set_iteration_stride(1);
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(model_offset);
  data_reader->set_initial_position();

  int min_stride_across_models = max_mini_batch_size * m_comm->get_num_trainers();  /// Given that each model has to have at least one reader, what is the minimum stride
  if (apportioned) {
    min_stride_across_models = max_mini_batch_size;
  }

  data_reader->set_global_mini_batch_size(min_stride_across_models); /// The global mini-batch is a full mini-batch per model

  data_reader->set_last_mini_batch_size(max_mini_batch_size); /// By default the last mini-batch is a full one
  data_reader->set_global_last_mini_batch_size(min_stride_across_models); /// By default the last mini-batch is a full one per model

  int num_whole_mini_batches_per_model = floor(data_reader->get_num_data() / min_stride_across_models);
  int global_partial_mini_batch_size = data_reader->get_num_data() - (num_whole_mini_batches_per_model * min_stride_across_models);
  int per_model_partial_mini_batch_size = global_partial_mini_batch_size / m_comm->get_num_trainers();
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  int last_mini_batch_threshold = num_whole_mini_batches_per_model * min_stride_across_models;

  int world_master_remainder_adjustment = data_reader->get_num_data()
                                          - (num_whole_mini_batches_per_model * min_stride_across_models)
                                          - (per_model_partial_mini_batch_size * m_comm->get_num_trainers());
  if(m_comm->get_trainer_rank() == 0) {
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

  /// Note that m_comm->get_trainer_rank() + m_comm->get_rank_in_trainer() is not equivalent to m_comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers model rank, how many models have a higher rank

  /// By default the last stride of each reader is part of a regular (full) round
  data_reader->set_stride_to_last_mini_batch(data_reader->get_stride_to_next_mini_batch());

  /// BVE FIXME - I feel like this is wrong  I don't think that the -1
  /// should be there
  int last_mini_batch_offset = std::max(0, num_whole_mini_batches_per_model - 1) * data_reader->get_stride_to_next_mini_batch();

  ///  The last mini-batch may be partial and thus may have a smaller stride
  if(per_model_partial_mini_batch_size > 0 || world_master_remainder_adjustment > 0) {
    data_reader->set_stride_to_last_mini_batch((last_mini_batch_threshold - data_reader->get_base_offset() - data_reader->get_model_offset() - last_mini_batch_offset) + m_comm->get_trainer_rank() * per_model_partial_mini_batch_size + m_comm->get_rank_in_trainer());
  }

  //  cout << "[" << m_comm->get_rank_in_world() << "] " << m_comm->get_trainer_rank() << " model rank, "<< m_comm->get_rank_in_trainer() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " parallel_readers_with_extra_mini_batch " << /*parallel_readers_with_extra_mini_batch <<*/ " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " with a last stride of " << data_reader->get_stride_to_last_mini_batch() << " and stride of " << data_reader->get_stride_to_next_mini_batch() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << /*parallel_readers_with_extra_mini_batch << */" model bracket = " << (/*parallel_readers_with_extra_mini_batch **/ max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;
//cout << "[" << m_comm->get_rank_in_world() << "] " << m_comm->get_trainer_rank() << " model rank, "<< m_comm->get_rank_in_trainer() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << "(m_num_mini_batches_per_reader=" << data_reader->get_num_mini_batches_per_reader() << ") parallel_readers_with_extra_mini_batch " << /*parallel_readers_with_extra_mini_batch <<*/ " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << data_reader->get_last_mini_batch_size() << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << data_reader->get_last_mini_batch_threshold() << " with a last stride of " << data_reader->get_stride_to_last_mini_batch() << " and stride of " << data_reader->get_batch_stride() << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << /*parallel_readers_with_extra_mini_batch << */" model bracket = " << (/*parallel_readers_with_extra_mini_batch **/ max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< data_reader->get_base_offset() << " model offset " << data_reader->get_model_offset() <<endl;
  return;
}

void lbann::partitioned_io_buffer::calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) {
  if(data_reader == nullptr) { return; }
  // If the data reader does not have any data bail out (e.g. unused validation reader)
  if(data_reader->get_num_data() == 0) { return; }

  if(max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel readers
  int num_parallel_readers_per_model = compute_max_num_parallel_readers(data_reader->get_num_data(), max_mini_batch_size, m_comm->get_procs_per_trainer());
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if(num_parallel_readers_per_model == 0
     || (num_parallel_readers_per_model != m_comm->get_procs_per_trainer() && num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_distribution: number of parallel readers is zero");
  }

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = max_mini_batch_size;
  int base_offset  = m_comm->get_rank_in_trainer();
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
