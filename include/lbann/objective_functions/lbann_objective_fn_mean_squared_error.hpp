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

#ifndef LBANN_OBJECTIVE_FN_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_OBJECTIVE_FN_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/objective_functions/lbann_objective_fn.hpp"
#include "lbann/lbann_Elemental_extensions.h"

namespace lbann
{
  class mean_squared_error : public objective_fn {
  public:
    mean_squared_error(lbann_comm* comm);
    ~mean_squared_error();

    void setup(int num_neurons, int mini_batch_size);
    void fp_set_std_matrix_view(int64_t cur_mini_batch_size);
    double compute_mean_squared_error(ElMat &predictions_v, ElMat &groundtruth_v);
    double compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v);

  protected:
    /** Workspace to compute the squared error differences */
    DistMat m_sum_squared_errors;
    DistMat m_sum_squared_errors_v;
    /** Colume-wise sum of the costs of a minibatch. */
    ColSumMat m_minibatch_cost;
  };
}

#endif // LBANN_OBJECTIVE_FN_MEAN_SQUARED_ERROR_HPP_INCLUDED
