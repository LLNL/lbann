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
// lbann_partitioned_minibatch_parallel_io .hpp .cpp - parallel I/O routines for distriubted minibatches
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_PARTITIONED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED
#define LBANN_PARTITIONED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_distributions/data_distribution.hpp"

namespace lbann {
class partitioned_minibatch_parallel_io : public generic_data_distribution {
 public:
  partitioned_minibatch_parallel_io(lbann_comm *comm, int num_parallel_readers, int mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers);
  partitioned_minibatch_parallel_io(
    const partitioned_minibatch_parallel_io&) = default;
  partitioned_minibatch_parallel_io& operator=(
    const partitioned_minibatch_parallel_io&) = default;
  virtual ~partitioned_minibatch_parallel_io() {}

  int fetch_to_local_matrix(Mat& M_local);
  void distribute_from_local_matrix(Mat& M_local, CircMat& Ms);
  bool is_data_set_processed();

  void calculate_num_iterations_per_epoch(generic_data_reader *data_reader);
};
}

#endif  // LBANN_PARTITIONED_MINIBATCH_PARALLEL_IO_HPP_INCLUDED
