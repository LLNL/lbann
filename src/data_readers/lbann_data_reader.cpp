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
// lbann_data_reader .hpp .cpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader.hpp"

using namespace std;
//using namespace El;

void lbann::DataReader::calculate_multi_model_data_distribution_packed(lbann_comm *comm) {
  int max_mini_batch_size = BatchSize;
  int num_parallel_readers_per_model = (m_stride / comm->get_num_models()) / max_mini_batch_size;

  //  m_max_std_parallel_readers = num_parallel_readers_per_model; /// @todo FIXME BVE this should be computed differently
  m_last_mini_batch_size = 0;

  int num_whole_mini_batches = rint(getNumData() / (comm->get_num_models() * max_mini_batch_size) /*m_stride*/);
  int num_whole_mini_batches_per_model = num_whole_mini_batches / comm->get_num_models();
  int num_whole_mini_batches_per_reader = rint(num_whole_mini_batches_per_model / num_parallel_readers_per_model);
  int parallel_readers_with_extra_mini_batch = num_whole_mini_batches_per_model % num_parallel_readers_per_model;
  int per_model_partial_mini_batch_size = (getNumData() - (num_whole_mini_batches* max_mini_batch_size))/(comm->get_num_models()/* * num_parallel_readers_per_model*/);
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  m_last_mini_batch_threshold = num_whole_mini_batches_per_reader * max_mini_batch_size * num_parallel_readers_per_model;

  if(comm->get_rank_in_model() < parallel_readers_with_extra_mini_batch) {
    num_whole_mini_batches_per_reader += 1;
    m_last_mini_batch_size = max_mini_batch_size;
  }
  //  m_max_partial_parallel_readers = parallel_readers_with_extra_mini_batch;

  int world_master_remainder_adjustment = getNumData() 
    - (num_whole_mini_batches * max_mini_batch_size) 
    - (per_model_partial_mini_batch_size * comm->get_num_models());
  if(comm->am_world_master()) {
    world_master_remainder_data = world_master_remainder_adjustment;
    world_master_remainder_adjustment = 0;
  }
  per_model_partial_mini_batch_size += world_master_remainder_data;

  // if(world_master_remainder_adjustment > 0) {
  //   m_max_partial_parallel_readers++;
  // }
  //      m_last_mini_batch_threshold = m_stride * num_whole_mini_batches;
  //      m_last_mini_batch_size = partial_mini_batch_size;

  /// The first reader that doesn't have an extra mini batch gets
  /// the partial batch
  if(comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch) {
    //        m_last_mini_batch_threshold = num_whole_mini_batches * max_mini_batch_size;
    m_last_mini_batch_size = per_model_partial_mini_batch_size;
  // }else {
  //   m_last_mini_batch_size = 0;
  }

  if(m_last_mini_batch_size > max_mini_batch_size) { throw new lbann_exception("Error in calculating the partial mini-batch size, exceeds the max mini-batch size"); }

  /// Note that comm->get_model_rank() + comm->get_rank_in_model() is not equivalent to comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers rank, how many readers have a higher rank
  int num_readers_at_full_stride = (comm->get_num_models() - comm->get_model_rank()) * num_parallel_readers_per_model;
  /// Given the data readers rank, how many readers have a lower rank
  int num_readers_at_last_stride = comm->get_model_rank() * num_parallel_readers_per_model;
  if(comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch/*< num_parallel_readers_per_model*/) { /// If this rank is one of the readers, adjust the number of readers to account for that
    num_readers_at_full_stride -= comm->get_rank_in_model();
    num_readers_at_last_stride += comm->get_rank_in_model();
  }
  /// Compute how big the stride should be assuming that each higher ranked parallel reader has completed a full mini-batch
  /// and each lower ranked parallel reader has completed a partial mini-batch
  m_last_mini_batch_stride = max_mini_batch_size * num_readers_at_full_stride
    + (per_model_partial_mini_batch_size * (num_readers_at_last_stride)) + world_master_remainder_adjustment;

  m_last_mini_batch_stride = m_stride;

  cout << "[" << comm->get_rank_in_world() << "] " << comm->get_model_rank() << " model rank, num_whole_mini_batches=" << num_whole_mini_batches << " num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << " parallel_readers_with_extra_mini_batch " << parallel_readers_with_extra_mini_batch << " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << m_last_mini_batch_threshold << " with a last stride of " << m_last_mini_batch_stride << " and stride of " << m_stride << " and there are " << num_parallel_readers_per_model << " parallel readers per model " <<endl;

  return;
}
