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
// softmax_cuda.cu - GPU helper routines for softmax layer
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_LAYER_SOFTMAX_CUDA_HPP_INCLUDED
#define LBANN_LAYER_SOFTMAX_CUDA_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

namespace lbann {
namespace softmax_cuda {

void fp_cutoff(cudnn::cudnn_manager &cudnn,
               const std::vector<DataType*> &activations,
               El::Int h, El::Int w,
               DataType min_output);

void bp_cutoff(cudnn::cudnn_manager &cudnn,
               const std::vector<DataType*> &activations,
               const std::vector<DataType*> &error_signals,               
               El::Int h, El::Int w,
               DataType min_output);

void bp_compute_cross_entropy_shortcut(cudnn::cudnn_manager &cudnn,
                                       const std::vector<DataType*> &activations,
                                       const std::vector<DataType*> &prev_error_signals,
                                       const std::vector<DataType*> &error_signals,
                                       El::Int h, El::Int w,
                                       DataType min_output);


} // namespace softmax_cuda
} // namespace lbann

#endif  // LBANN_LAYER_SOFTMAX_CUDA_HPP_INCLUDED
