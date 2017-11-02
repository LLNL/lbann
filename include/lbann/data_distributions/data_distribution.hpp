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

class fetch_data_functor {
 public:
  fetch_data_functor (bool is_input_layer, bool is_for_regression) : 
    _is_input_layer(is_input_layer), _is_for_regression(is_for_regression) {}
  int operator() (Mat& M_local, generic_data_reader* data_reader) const {
    if (_is_input_layer) {
      return data_reader->fetch_data(M_local);
    } else {
      if (_is_for_regression) {
        return data_reader->fetch_responses(M_local);
      } else {
        return data_reader->fetch_labels(M_local);
      }
    }
  }
 private:
  const bool _is_input_layer;
  const bool _is_for_regression;
};

class update_data_reader_functor {
 public:
  update_data_reader_functor (bool is_input_layer) : 
    _is_input_layer(is_input_layer) {}
  int operator() (bool is_active_reader, generic_data_reader* data_reader) const {
    if (_is_input_layer) {
      return data_reader->update(is_active_reader);
    } else {
      return (data_reader->is_data_reader_done(is_active_reader));
    }
  }
 private:
  const bool _is_input_layer;
};

class generic_data_distribution {
public:
  generic_data_distribution(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);
  generic_data_distribution(
    const generic_data_distribution&);
  generic_data_distribution& operator=(
    const generic_data_distribution&);
  virtual ~generic_data_distribution() {
    if(fetch_data_fn != nullptr) {
      delete fetch_data_fn;
    }
    if(update_data_reader_fn != nullptr) {
      delete update_data_reader_fn;
    }
  }

  virtual int fetch_to_local_matrix(Mat& M_local, generic_data_reader *data_reader) = 0;
  virtual void distribute_from_local_matrix(Mat& M_local, CircMat& Ms, generic_data_reader *data_reader) {}
  virtual bool is_data_set_processed(generic_data_reader *data_reader)  = 0;

  virtual void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
  virtual void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader) = 0;
;
  virtual int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const = 0; 

  virtual void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) = 0;

  /*
  virtual execution_mode get_execution_mode() const {
    return execution_mode::invalid;
  }
  */
  
  /// Return the rank of the current root node for the Elemental Distribution
  virtual int current_root_rank() const {
    return m_root;
  }

  /// Is this rank the current root node for the Elemental Distribution
  bool is_current_root() const {
    return (m_comm->get_rank_in_model() == m_root);
  }

  /// Is the local reader done
  virtual bool is_local_reader_done() const {
    return m_local_reader_done;
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

  const fetch_data_functor *fetch_data_fn;
  const update_data_reader_functor *update_data_reader_fn;

  // BVE FIXME this will be wrong for LTFB
  int m_num_data_per_epoch;
};
}

#endif // LBANN_DATA_DISTRIBUTION_HPP_INCLUDED
