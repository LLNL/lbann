////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_LOSS_ENTRYWISE_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_ENTRYWISE_HPP_INCLUDED

#include "lbann/layers/math/binary.hpp"

namespace lbann {

#ifndef LBANN_ENTRYWISE_LAYER_INSTANTIATE
#define BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, DEVICE)                 \
  extern template class LAYER_NAME<T, data_layout::DATA_PARALLEL, DEVICE>; \
  extern template class LAYER_NAME<T, data_layout::MODEL_PARALLEL, DEVICE>
#else
#define BINARY_ETI_DECL_MACRO_DEV(...)
#endif // LBANN_BINARY_LAYER_INSTANTIATE

#ifdef LBANN_HAS_GPU
#define BINARY_ETI_DECL_MACRO(LAYER_NAME, T)                      \
  BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, El::Device::CPU);       \
  BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, El::Device::GPU)
#else
#define BINARY_ETI_DECL_MACRO(LAYER_NAME, T)                \
  BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, El::Device::CPU)
#endif // LBANN_HAS_GPU

// Convenience macro to define an entry-wise binary layer class
#define DEFINE_ENTRYWISE_BINARY_LAYER(layer_name, layer_string)         \
  LBANN_DECLARE_ENTRYWISE_BINARY_LAYER(layer_name, layer_string);       \
  BINARY_ETI_DECL_MACRO(layer_name, float);                             \
  BINARY_ETI_DECL_MACRO(layer_name, double)

// Cross entropy loss
DEFINE_ENTRYWISE_BINARY_LAYER(binary_cross_entropy_layer,
                              "binary cross entropy");
DEFINE_ENTRYWISE_BINARY_LAYER(sigmoid_binary_cross_entropy_layer,
                              "sigmoid binary cross entropy");

// Boolean loss functions
DEFINE_ENTRYWISE_BINARY_LAYER(boolean_accuracy_layer, "Boolean accuracy");
DEFINE_ENTRYWISE_BINARY_LAYER(boolean_false_negative_layer,
                              "Boolean false negative rate");
DEFINE_ENTRYWISE_BINARY_LAYER(boolean_false_positive_layer,
                              "Boolean false positive rate");

} // namespace lbann

#undef DEFINE_ENTRYWISE_BINARY_LAYER
#undef BINARY_ETI_DECL_MACRO
#undef BINARY_ETI_DECL_MACRO_DEV

#endif // LBANN_LAYERS_LOSS_ENTRYWISE_HPP_INCLUDED
