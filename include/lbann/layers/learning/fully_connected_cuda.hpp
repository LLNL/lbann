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
// fully_connected_cuda.cu - GPU helper routines for fully connected layer
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_LAYER_FULLY_CONNECTED_CUDA_HPP_INCLUDED
#define LBANN_LAYER_FULLY_CONNECTED_CUDA_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {
namespace fully_connected_cuda {

void row_sum(cudnn::cudnn_manager &cudnn,
             std::vector<DataType*> matrices,
             El::Int h, El::Int w,
             DataType factor,
             Mat &dest,
             const std::vector<DataType*> &work_column);

/// tensor <= tensor * beta + bias * factor
void add_tensor(DataType factor,
                DataType *bias,
                El::Int bias_h, El::Int bias_w,
                DataType beta,                
                DataType *tensor,
                El::Int tensor_h, El::Int tensor_w);


} // namespace fully_connected_cuda
} // namespace lbann

#endif  // LBANN_LAYER_FULLY_CONNECTED_CUDA_HPP_INCLUDED
