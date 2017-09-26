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

#ifndef LBANN_DATA_DISTRIBUTION_HPP_INCLUDED
#define LBANN_DATA_DISTRIBUTION_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/data_readers/data_reader.hpp"

namespace lbann
{
class generic_data_distribution {
public:
  generic_data_distribution(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);
  generic_data_distribution(
    const generic_data_distribution&) = default;
  generic_data_distribution& operator=(
    const generic_data_distribution&) = default;
  virtual ~generic_data_distribution() {}

  virtual int fetch_to_local_matrix(Mat& M_local) { return 0; }
  virtual void distribute_from_local_matrix(Mat& M_local, CircMat& Ms) {}
  virtual bool is_data_set_processed() { return false; }
  virtual generic_data_reader *get_data_reader(execution_mode mode);
  virtual generic_data_reader *get_data_reader();
  virtual int get_num_parallel_readers(execution_mode mode);
  virtual int get_num_parallel_readers();
  virtual int get_num_iterations_per_epoch(execution_mode mode);
  virtual int get_num_iterations_per_epoch();
  virtual int get_mini_batch_size(execution_mode mode);
  virtual int get_last_mini_batch_size(execution_mode mode);
  virtual int get_last_mini_batch_size();
  virtual int get_current_mini_batch_size(execution_mode mode);
  virtual int get_current_mini_batch_size();
  virtual int get_global_mini_batch_size(execution_mode mode);
  virtual int get_global_last_mini_batch_size(execution_mode mode);
  virtual int get_current_global_mini_batch_size(execution_mode mode);
  virtual int get_current_global_mini_batch_size();

  virtual void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
  virtual void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
  virtual void calculate_num_iterations_per_epoch_training_spans_models(int mini_batch_size);
  virtual void calculate_num_iterations_per_epoch_training_unique_per_models(int mini_batch_size);
  virtual int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) { return 0; }

  /// @todo BVE replace this with a function pointer that is passed
  /// into the fetch_to_local_matrix function to avoid the
  /// "circular" function dependence
  virtual int fetch_from_data_reader(Mat& M_local) {
    return 0;
  }
  virtual void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {}
  virtual bool update_data_reader() {
    return false;
  }
  virtual execution_mode get_execution_mode() const {
    return execution_mode::invalid;
  }

  /// Is this rank the current root node for the Elemental Distribution
  virtual bool is_current_root() {
    return (m_comm->get_rank_in_model() == m_root);
  }

 protected:
  lbann_comm *m_comm;
  /** Which rank is the root of the CircMat */
  int m_root;
  /** Requested maximum number of parallel readers (I/O streams) */
  int m_requested_max_num_parallel_readers;
  int m_local_reader_done;
  /** Number of samples in the current mini-batch */
  int m_num_samples_in_batch;
  /** Has the layer copied valid data into the local matrix */
  bool m_local_data_valid;

  std::map<execution_mode, generic_data_reader *> m_data_readers;

  int m_cur_step_in_epoch;

  int m_num_data_per_epoch;
};
}

#endif // LBANN_DATA_DISTRIBUTION_HPP_INCLUDED
