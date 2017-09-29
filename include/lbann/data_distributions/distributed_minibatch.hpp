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

#ifndef LBANN_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
#define LBANN_DISTRIBUTED_MINIBATCH_HPP_INCLUDED

#include "lbann/data_distributions/data_distribution.hpp"

namespace lbann {

/**
 * Parallel I/O routines for managing distributed minibatches
 */
class distributed_minibatch : public virtual generic_data_distribution {
 public:
  distributed_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers);
  distributed_minibatch(
    const distributed_minibatch&) = default;
  distributed_minibatch& operator=(
    const distributed_minibatch&) = default;
  virtual ~distributed_minibatch() {}

  int fetch_to_local_matrix(Mat& M_local);
  void distribute_from_local_matrix(Mat& M_local, CircMat& Ms);
  bool is_data_set_processed();

  void calculate_num_iterations_per_epoch(int num_models, int model_rank, int max_mini_batch_size, generic_data_reader *data_reader);
  void calculate_num_iterations_per_epoch_spanning_models(int max_mini_batch_size, generic_data_reader *data_reader);
  void calculate_num_iterations_per_epoch_single_model(int max_mini_batch_size, generic_data_reader *data_reader);
  int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers);
};

}  // namespace lbann

#endif  // LBANN_DISTRIBUTED_MINIBATCH_HPP_INCLUDED
