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

#ifndef LBANN_OBJECTIVE_FN_HPP_INCLUDED
#define LBANN_OBJECTIVE_FN_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

namespace lbann
{
  class objective_fn {
  public:
    objective_fn() {}
    virtual ~objective_fn() {}
    virtual void setup(int num_neurons, int mini_batch_size) {}
    virtual void fp_set_std_matrix_view(int64_t cur_mini_batch_size) {}
    virtual DataType compute_obj_fn(ElMat &predictions_v, ElMat &groundtruth_v) {}
  };
}

#endif // LBANN_OBJECTIVE_FN_INCLUDED
