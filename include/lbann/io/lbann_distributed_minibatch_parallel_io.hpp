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
// lbann_distributed_minibatch_parallel_io .hpp .cpp - parallel I/O routines for distriubted minibatches
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DISTRIBUTED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED
#define LBANN_DISTRIBUTED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

namespace lbann
{
  class distributed_minibatch_parallel_io {
  public:
    distributed_minibatch_parallel_io(lbann_comm* comm, int num_parallel_readers, uint mini_batch_size, int training_data_set_size, int testing_data_set_size=0);

    int fetch_to_local_matrix(Mat& M_local);
    void distribute_from_local_matrix(Mat& M_local, CircMat& Ms);
    bool is_data_set_processed();
    int get_num_parallel_readers();

    virtual int fetch_from_data_reader(Mat& M_local) { return 0; }
    virtual void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {}
    virtual bool update_data_reader() { return false; }
    virtual execution_mode get_execution_mode() { return invalid; }

    /// Is this rank the current root node for the Elemental Distribution
    bool is_current_root() { return (comm->get_rank_in_model() == m_root); }

  public:
    lbann_comm* comm;
    int m_root; /** Which rank is the root of the CircMat */
    int m_num_parallel_readers_training; /** Number of parallel readers (I/O streams) for training data */
    int m_num_parallel_readers_testing; /** Number of parallel readers (I/O streams) for testing data  */
    int m_local_reader_done;
    uint m_mini_batch_size; /** Size of the mini-batch */
    bool m_local_data_valid; /** Has the layer copied valid data into the local matrix */

    long m_num_data_per_epoch;
  };
}

#endif  // LBANN_DISTRIBUTED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED
