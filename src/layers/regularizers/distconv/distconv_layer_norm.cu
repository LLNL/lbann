////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_LAYERS_REGULARIZERS_DISTCONV_LAYER_NORM_INSTANTIATE

#include "../layer_norm_kernel.cuh"
#include "lbann/layers/regularizers/distconv/distonv_layer_norm.hpp"

#ifdef LBANN_HAS_DISTCONV

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization ::calculate_forward_stats(
  const DCTensor<Allocator>& input)
{}

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization ::apply_normalization(
  const DCTensor<Allocator>& input,
  const DCTensor<Allocator>& statistics,
  DCTensor<Allocator>& output)
{}

template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization ::calculate_backward_stats(
  const DCTensor<Allocator>& input,
  const DCTensor<Allocator>& output_grad,
  const DCTensor<Allocator>& statistics,
  DCTensor<Allocator>& statistics_grad)
{}
template <typename Backend, typename DataType>
template <typename Allocator>
void LayerNormalization ::apply_grad(const DCTensor<Allocator>& input,
                                     const DCTensor<Allocator>& output_grad,
                                     const DCTensor<Allocator>& statistics,
                                     const DCTensor<Allocator>& statistics_grad,
                                     DCTensor<Allocator>& input_grad)
{}

#endif // LBANN_HAS_DISTCONV