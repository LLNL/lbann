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

namespace lbann {

void DataReader::setup(int base_offset, int stride, int model_offset,
                       lbann_comm* comm) {
  m_model_offset = model_offset;
  m_base_offset = base_offset;
  m_stride = stride;
  m_last_mini_batch_stride = stride;
  m_current_mini_batch_idx = 0;

  if(comm != NULL) {
    calculate_multi_model_data_distribution(comm);
    m_use_alt_last_mini_batch_size = true;
  }

  CurrentPos = m_base_offset + m_model_offset;
  if (m_shuffle) {
    std::shuffle(ShuffledIndices.begin(), ShuffledIndices.end(),
                 get_data_seq_generator());
  }
}

void DataReader::setup() {
  DataReader::setup(0, BatchSize);
}

bool DataReader::update() {
  /// Is the mini-batch that is about to finish equal to the second to last mini-batch
  if(m_use_alt_last_mini_batch_size && ((m_current_mini_batch_idx+1) >= (m_num_mini_batches_per_reader-1))) {
    CurrentPos += m_last_mini_batch_stride;
  }else {
    CurrentPos += m_stride;
  }
  if (CurrentPos < (int)ShuffledIndices.size()) {
    m_current_mini_batch_idx++;
    return true;
  } else {
    if (m_shuffle) {
      std::shuffle(ShuffledIndices.begin(), ShuffledIndices.end(),
                   get_data_seq_generator());
    }
    m_current_mini_batch_idx = 0;
    CurrentPos = m_base_offset + m_model_offset;
    return false;
  }
}

int DataReader::getBatchSize() {
  if (m_use_alt_last_mini_batch_size &&
      m_current_mini_batch_idx >= (m_num_mini_batches_per_reader-1)) {
    return m_last_mini_batch_size;
  } else {
    return BatchSize; 
  }
}

int DataReader::get_next_position() {
  /// Is the mini-batch that is about to finish equal to the second to last mini-batch
  if (m_use_alt_last_mini_batch_size &&
      ((m_current_mini_batch_idx+1) >= (m_num_mini_batches_per_reader-1))) {
    return CurrentPos + m_last_mini_batch_stride;
  } else {
    return CurrentPos + m_stride;
  }
}

void DataReader::select_subset_of_data() {
  if(!get_firstN()) {
    std::shuffle(ShuffledIndices.begin(), ShuffledIndices.end(), get_data_seq_generator());
  }

  if (not (has_max_sample_count() or has_use_percent() or has_validation_percent())) {
    return;
  }

  if (has_max_sample_count()) {
    size_t count = get_max_sample_count();
    if(count > getNumData()) {
      stringstream err;
      err << __FILE__ << " " << __LINE__ 
          << " :: DataReader::select_subset_of_data() - max_sample_count=" << count
          << " is > getNumData=" << getNumData();
      throw lbann_exception(err.str());
    }
    ShuffledIndices.resize(get_max_sample_count());
  } else if (has_use_percent()) {
    ShuffledIndices.resize(get_use_percent()*getNumData());
  }

  if (has_validation_percent()) {
    long unused = get_validation_percent()*getNumData(); //getNumData() = ShuffledIndices.size()
    long use_me = getNumData() - unused;
    if (unused > 0) {
      m_unused_indices=std::vector<int>(ShuffledIndices.begin() + use_me, ShuffledIndices.end());
      ShuffledIndices.resize(use_me);
    }  
  }

  if(!get_firstN()) {
    std::sort(ShuffledIndices.begin(), ShuffledIndices.end());
    std::sort(m_unused_indices.begin(), m_unused_indices.end());
  }
}

void DataReader::use_unused_index_set() {
  ShuffledIndices.swap(m_unused_indices);
  m_unused_indices.clear();
  std::vector<int>().swap(m_unused_indices); // Trick to force memory reallocation
}

DataReader& DataReader::operator=(const DataReader& source) {
  this->BatchSize = source.BatchSize;
  this->CurrentPos = source.CurrentPos;
  this->m_shuffle = source.m_shuffle;
  this->m_stride = source.m_stride;
  this->m_base_offset = source.m_base_offset;
  this->m_model_offset = source.m_model_offset;
  this->m_use_alt_last_mini_batch_size = source.m_use_alt_last_mini_batch_size;
  this->m_last_mini_batch_threshold = source.m_last_mini_batch_threshold;
  this->m_last_mini_batch_size = source.m_last_mini_batch_size;
  this->m_last_mini_batch_stride = source.m_last_mini_batch_stride;

  // Vectors implement a deep copy
  this->ShuffledIndices = source.ShuffledIndices;
  this->m_unused_indices = source.m_unused_indices;
  return *this;
}


void DataReader::calculate_multi_model_data_distribution(lbann_comm *comm) {
  int max_mini_batch_size = BatchSize;
  int num_parallel_readers_per_model = (m_stride / comm->get_num_models()) / max_mini_batch_size;
  int min_stride_across_models = max_mini_batch_size * comm->get_num_models();  /// Given that each model has to have at least one reader, what is the minimum stride

  m_last_mini_batch_size = max_mini_batch_size; /// By default the last mini-batch is a full one

  int num_whole_mini_batches_per_model = floor(getNumData() / min_stride_across_models);
  int num_whole_mini_batches_per_reader = floor(num_whole_mini_batches_per_model / num_parallel_readers_per_model);
  int parallel_readers_with_extra_mini_batch = num_whole_mini_batches_per_model % num_parallel_readers_per_model;
  int per_model_partial_mini_batch_size = (getNumData() - (num_whole_mini_batches_per_model * min_stride_across_models))/(comm->get_num_models());
  int world_master_remainder_data = 0;

  // Compute how many full "parallel" mini-batches are available
  m_last_mini_batch_threshold = num_whole_mini_batches_per_model * min_stride_across_models;

  if(comm->get_rank_in_model() < parallel_readers_with_extra_mini_batch) {
    num_whole_mini_batches_per_reader += 1;
    m_last_mini_batch_size = max_mini_batch_size;
  }

  m_num_mini_batches_per_reader = num_whole_mini_batches_per_reader;

  int world_master_remainder_adjustment = getNumData() 
    - (num_whole_mini_batches_per_model * min_stride_across_models) 
    - (per_model_partial_mini_batch_size * comm->get_num_models());
  if(comm->am_world_master()) {
    world_master_remainder_data = world_master_remainder_adjustment;
    world_master_remainder_adjustment = 0;
  }
  per_model_partial_mini_batch_size += world_master_remainder_data;

  /// The first reader that doesn't have an extra mini batch gets the partial batch
  if(comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch && per_model_partial_mini_batch_size > 0) {
    m_num_mini_batches_per_reader++;
    m_last_mini_batch_size = per_model_partial_mini_batch_size;
  }

  if(m_last_mini_batch_size > max_mini_batch_size) { throw new lbann_exception("Error in calculating the partial mini-batch size, exceeds the max mini-batch size"); }

  /// Note that comm->get_model_rank() + comm->get_rank_in_model() is not equivalent to comm->get_world_rank() from a parallel I/O perspective
  /// Given the data readers model rank, how many models have a higher rank

  /// By default the last stride of each reader is part of a regular (full) round
  m_last_mini_batch_stride = m_stride;

  int last_mini_batch_offset = max(0, num_whole_mini_batches_per_reader - 1) * m_stride;

  ///  The last mini-batch may be partial and thus may have a smaller stride
  if(comm->get_rank_in_model() == parallel_readers_with_extra_mini_batch && per_model_partial_mini_batch_size > 0) {
    m_last_mini_batch_stride = (m_last_mini_batch_threshold - m_base_offset - m_model_offset - last_mini_batch_offset) 
      + comm->get_model_rank() * per_model_partial_mini_batch_size + world_master_remainder_adjustment; /// BVE 10/18/16
  }

  //  cout << "[" << comm->get_rank_in_world() << "] " << comm->get_model_rank() << " model rank, "<< comm->get_rank_in_model() << " rank in model, num_whole_mini_batches_per_model " << num_whole_mini_batches_per_model << " num_whole_mini_batches_per_reader " << num_whole_mini_batches_per_reader << "(m_num_mini_batches_per_reader=" << m_num_mini_batches_per_reader << ") parallel_readers_with_extra_mini_batch " << parallel_readers_with_extra_mini_batch << " partial_mini_batch_size=" << per_model_partial_mini_batch_size << " last mini bath size=" << m_last_mini_batch_size << " world_master_remainder_data=" << world_master_remainder_data << " threshold " << m_last_mini_batch_threshold << " with a last stride of " << m_last_mini_batch_stride << " and stride of " << m_stride << " and there are " << num_parallel_readers_per_model << " parallel readers per model" << " last mini batch offset = " << last_mini_batch_offset <<  " parallel reader with extra minibatch = " << parallel_readers_with_extra_mini_batch << " model bracket = " << (parallel_readers_with_extra_mini_batch * max_mini_batch_size + per_model_partial_mini_batch_size + world_master_remainder_data) <<" base ofset "<< m_base_offset << " model offset " << m_model_offset <<endl;

  return;
}

/** \brief Given directory to store checkpoint files, write state to file and add to number of bytes written */
bool DataReader::saveToCheckpointShared(persist& p, const char* name)
{
    // rank 0 writes the training state file
    if (p.m_rank == 0) {
        char fieldname[1024];

        // record minibatch index
        snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
        p.write_uint64(persist_type::train, fieldname, (uint64_t) m_current_mini_batch_idx);

        // get size of list of training examples
        int size = ShuffledIndices.size();

        // record size of ShuffleIndices
        snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
        p.write_uint64(persist_type::train, fieldname, (uint64_t) size);

        // TODO: each model may have a different position, need to gather and write these
        // record current position within training data
        snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
        p.write_uint64(persist_type::train, fieldname, (uint64_t) CurrentPos);

        // write list of indices
        snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
        p.write_int32_contig(persist_type::train, fieldname, &ShuffledIndices[0], (uint64_t) size);
    }

    return true;
}

/** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
bool lbann::DataReader::loadFromCheckpointShared(persist& p, const char* name)
{
    // rank 0 reads the training state file
    if (p.m_rank == 0) {
        char fieldname[1024];

        // record minibatch index
        uint64_t val;
        snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
        p.read_uint64(persist_type::train, fieldname, &val);
        m_current_mini_batch_idx = (int) val;

        // get size of ShuffleIndices
        snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
        p.read_uint64(persist_type::train, fieldname, &val);
        int size = (int) val;

        // get current position within data
        snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
        p.read_uint64(persist_type::train, fieldname, &val);
        CurrentPos = (int) val;

        // resize shuffled index array to hold values
        ShuffledIndices.resize(size);

        // read list of indices
        snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
        p.read_int32_contig(persist_type::train, fieldname, &ShuffledIndices[0], (uint64_t) size);
    }

    // broadcast minibatch index
    MPI_Bcast(&m_current_mini_batch_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // TODO: with multiple readers, make this a scatter
    // broadcast current position
    MPI_Bcast(&CurrentPos, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // broadcast values from rank 0
    int size = ShuffledIndices.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // resize shuffled index array to hold values
    if (p.m_rank != 0) {
        ShuffledIndices.resize(size);
    }

    // broadcast index array
    MPI_Bcast(&ShuffledIndices[0], size, MPI_INT, 0, MPI_COMM_WORLD);

    return true;
}

void DataReader::set_file_dir(std::string s) { 
  m_file_dir = s; 
}

std::string DataReader::get_file_dir() { 
  return m_file_dir; 
}

void DataReader::set_data_filename(std::string s) { 
  m_data_fn = s; 
}

std::string DataReader::get_data_filename() { 
    if (m_data_fn == "") {
      std::stringstream s;
      s << __FILE__ << " " << __LINE__ << " :: you apparently did not call "
        << "set_data_filename; this is an error!";
      throw lbann_exception(s.str());
    }
    return m_data_fn; 
}

void DataReader::set_label_filename(std::string s) { 
  m_label_fn = s; 
}

string DataReader::get_label_filename() { 
    if (m_label_fn == "") {
      std::stringstream s;
      s << __FILE__ << " " << __LINE__ << " :: you apparently did not call "
        << "set_label_filename; this is an error!";
      throw lbann_exception(s.str());
    }
    return m_label_fn; 
}

void DataReader::set_max_sample_count(size_t s) {
  m_max_sample_count = s;
  m_max_sample_count_was_set = true;
}

size_t DataReader::get_max_sample_count() {
  return m_max_sample_count;
}

bool DataReader::has_max_sample_count() {
  return m_max_sample_count_was_set;
}

void DataReader::set_firstN(bool b) {
  m_first_n = b;
}

bool DataReader::get_firstN() {
  return m_first_n;
}

void DataReader::set_validation_percent(double s) {
  if (s < 0 or s > 1.0) {
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: set_validation_percent() - must be: s >= 0, s <= 1.0; you passed: " << s;
    throw lbann_exception(err.str());
  }
  m_validation_percent = s;
}

bool DataReader::has_validation_percent() {
  if (m_validation_percent == -1) return false;
  return true;
}

double DataReader::get_validation_percent() {
  return m_validation_percent;
}

void DataReader::set_use_percent(double s) {
  if (s < 0 or s > 1.0) {
    stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: set_use_percent() - must be: s >= 0, s <= 1.0; you passed: " << s;
    throw lbann_exception(err.str());
  }
  m_use_percent = s;
}

bool DataReader::has_use_percent() {
  if (m_use_percent == -1) return false;
  return true;
}

double DataReader::get_use_percent() {
  stringstream err;
  if (not has_use_percent()) {
    err << __FILE__ << " " << __LINE__ << " :: you must call set_use_percent()"
        << " but apparently have not done so";
    throw lbann_exception(err.str());
  }  
  return m_use_percent;
}

}  // namespace lbann
