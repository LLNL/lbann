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

#include "lbann/layers/activations/relu.hpp"

namespace lbann {

/** CPU implementation of forward prop computation. */
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  entrywise_activation_layer::fp_compute_cpu();
}

/** CPU implementation of forward prop computation. */
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  entrywise_activation_layer::fp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
/** GPU implementation of forward prop computation. */
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
  fp_compute_gpu();
}

/** GPU implementation of forward prop computation. */
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  fp_compute_gpu();
}
#endif // LBANN_HAS_GPUs

/** CPU implementation of backward prop computation. */
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  entrywise_activation_layer::bp_compute_cpu();
}

/** CPU implementation of backward prop computation. */
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  entrywise_activation_layer::bp_compute_cpu();
}

#ifdef LBANN_HAS_GPU
/** GPU implementation of backward prop computation. */
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
  bp_compute_gpu();
}

/** GPU implementation of backward prop computation. */
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  bp_compute_gpu();
}
#endif // LBANN_HAS_GPUs

} // namespace lbann
