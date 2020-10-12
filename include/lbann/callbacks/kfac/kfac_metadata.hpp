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
//
// callback_kfac .hpp .cpp - Callbacks for the K-FAC 2nd-order opt. method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_METADATA_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_METADATA_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/layers/learning/convolution.hpp"

namespace lbann {
namespace callback {

struct kfac_layer_metadata {
  size_t layer_id;
  convolution_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>* l_conv;
  bool is_fc, is_conv, is_bn_after_fc, is_bn_after_conv;
  size_t conv_input_spatial_prod, conv_output_spatial_prod;
  std::vector<int> conv_input_spatial_dims, conv_output_spatial_dims;
  size_t bn_num_channels, bn_spatial_prod;
  int proc_rank;
};

enum kfac_inverse_strategy {
  ALL,  // Apply round-robin assingment to all of the layers. may cause load imbalance.
  EACH, // Apply round-robin assingment to every type of layers. may
  // not work well for small networks.
  ROOT, // Use only the root GPU. This is only for testing.
};

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_METADATA_HPP_INCLUDED
