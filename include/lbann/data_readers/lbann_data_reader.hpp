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
// lbann_data_reader .hpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_HPP
#define LBANN_DATA_READER_HPP

#include "lbann/proto/lbann_proto.hpp"
#include "lbann/lbann_base.hpp"
#include "lbann/utils/lbann_random.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/io/lbann_file_io.hpp"
#include "lbann/io/lbann_persist.hpp"
#include <assert.h>
#include <algorithm>
#include <string>
#include <vector>
#include <unistd.h>


/**
 * @todo - add support for save and restore
 */
namespace lbann
{

class DataReader
{
public:
  DataReader(int batchSize, bool shuffle) :
    BatchSize(batchSize), CurrentPos(0), m_shuffle(shuffle),
    m_stride(batchSize), m_base_offset(0), m_model_offset(0), 
    m_use_alt_last_mini_batch_size(false),
    m_last_mini_batch_threshold(0), m_last_mini_batch_size(batchSize),
    m_last_mini_batch_stride(batchSize) 
  {}
    
  DataReader(int batchSize) :
    DataReader(batchSize, true) {}
    
  DataReader(const DataReader& source) :
    BatchSize(source.BatchSize), CurrentPos(source.CurrentPos), m_shuffle(source.m_shuffle),
    m_stride(source.m_stride), m_base_offset(source.m_base_offset), m_model_offset(source.m_model_offset),
    m_use_alt_last_mini_batch_size(source.m_use_alt_last_mini_batch_size),
    m_last_mini_batch_threshold(source.m_last_mini_batch_threshold), m_last_mini_batch_size(source.m_last_mini_batch_size), m_last_mini_batch_stride(source.m_last_mini_batch_stride),
    ShuffledIndices(source.ShuffledIndices), m_unused_indices(source.m_unused_indices),
    m_name(source.m_name)
  {}

  virtual ~DataReader() {}

  /**
   * Prepare to start processing an epoch of data.
   * If shuffle is true, then shuffle the indices of the data set
   * If the base offset is not specified set it to 0
   * If the stride is not specified set it to batch size
   */
  void setup(int base_offset, int stride, int model_offset = 0, lbann_comm *comm = NULL);
  void setup();

  virtual int fetch_data(Mat& X) { return 0; }
  virtual int fetch_label(Mat& Y) { return 0; }
  virtual int fetch_response(Mat& Y) { return 0; }

  /**
   * During the network's update phase, the data reader will
   * advanced the current position pointer.  If the pointer wraps
   * around, then reshuffle the data indicies.
   */
  virtual bool update();

  virtual int getNumLabels() { return 0; }
  virtual int getNumResponses() { return 1; }
  virtual int get_linearized_data_size() { return 0; }
  virtual int get_linearized_label_size() { return 0; }
  virtual int get_linearized_response_size() { return 1; }

  bool position_valid() { return (CurrentPos < (int)ShuffledIndices.size()); }
  bool at_new_epoch() { return (m_current_mini_batch_idx == 0); }
  int getBatchSize();
  int getPosition() { return CurrentPos; }
  int get_next_position();
  int* getIndices() { return &ShuffledIndices[0]; }
  int getNumData() { return (int)ShuffledIndices.size(); }
  int get_num_unused_data() { return (int)m_unused_indices.size(); }
  int* get_unused_data() { return &m_unused_indices[0]; }

  void select_subset_of_data(size_t max_sample_count, bool firstN);

  bool swap_used_and_unused_index_sets();

  DataReader& operator=(const DataReader& source);

  size_t trim_data_set(double use_percentage, bool firstN=false);

  void calculate_multi_model_data_distribution(lbann_comm *comm);

  /** \brief Given directory to store checkpoint files, write state to file and add to number of bytes written */
  bool saveToCheckpointShared(persist& p, const char* name);

  /** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
  bool loadFromCheckpointShared(persist& p, const char* name);

  /** \brief Returns this class's name **/
  const std::string & name() { return m_name; }

  /** \brief Sets this class's name **/
  void setName(std::string name) { m_name = name; }

protected:
  int BatchSize;
  int CurrentPos;
  int m_shuffle;
  /// Stride is typically batch_size, but may be a multiple of batch size if there are multiple readers
  int m_stride;
  /// If there are multiple instances of the reader, 
  /// then it may not reset to zero
  int m_base_offset;
  /// If there are multiple models with multiple instances of the reader, 
  /// each model's set of readers may not reset to zero
  /// Provide a set of size, strides, and thresholds to handle the last mini batch of a dataset
  int m_model_offset;
  int m_use_alt_last_mini_batch_size;
  int m_last_mini_batch_threshold;
  int m_last_mini_batch_size;
  int m_last_mini_batch_stride;

  int m_current_mini_batch_idx;
  int m_num_mini_batches_per_reader;

  std::vector<int> ShuffledIndices;
  /// Record of the indicies that are not being used for training
  std::vector<int> m_unused_indices;

  std::string m_name;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_HPP
