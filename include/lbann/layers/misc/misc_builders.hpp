////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_LAYERS_MISC_MISC_BUILDERS_HPP_INCLUDED
#define LBANN_LAYERS_MISC_MISC_BUILDERS_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

LBANN_DEFINE_LAYER_BUILDER(argmax);
LBANN_DEFINE_LAYER_BUILDER(argmin);
LBANN_DEFINE_LAYER_BUILDER(channelwise_mean);
LBANN_DEFINE_LAYER_BUILDER(channelwise_softmax);
LBANN_DEFINE_LAYER_BUILDER(covariance);
LBANN_DEFINE_LAYER_BUILDER(dft_abs);
LBANN_DEFINE_LAYER_BUILDER(dist_embedding);
LBANN_DEFINE_LAYER_BUILDER(external);
LBANN_DEFINE_LAYER_BUILDER(mini_batch_index);
LBANN_DEFINE_LAYER_BUILDER(mini_batch_size);
LBANN_DEFINE_LAYER_BUILDER(one_hot);
LBANN_DEFINE_LAYER_BUILDER(rowwise_weights_norms);
LBANN_DEFINE_LAYER_BUILDER(uniform_hash);
LBANN_DEFINE_LAYER_BUILDER(variance);

} // namespace lbann
#endif // LBANN_LAYERS_MISC_MISC_BUILDERS_HPP_INCLUDED
