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
#ifndef LBANN_LAYERS_TRANSFORM_TRANSFORM_BUILDERS_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_TRANSFORM_BUILDERS_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

LBANN_DEFINE_LAYER_BUILDER(batchwise_reduce_sum);
LBANN_DEFINE_LAYER_BUILDER(bernoulli);
LBANN_DEFINE_LAYER_BUILDER(categorical_random);
LBANN_DEFINE_LAYER_BUILDER(concatenate);
LBANN_DEFINE_LAYER_BUILDER(constant);
LBANN_DEFINE_LAYER_BUILDER(crop);
LBANN_DEFINE_LAYER_BUILDER(cross_grid_sum);
LBANN_DEFINE_LAYER_BUILDER(cross_grid_sum_slice);
LBANN_DEFINE_LAYER_BUILDER(discrete_random);
LBANN_DEFINE_LAYER_BUILDER(dummy);
LBANN_DEFINE_LAYER_BUILDER(evaluation);
LBANN_DEFINE_LAYER_BUILDER(gather);
LBANN_DEFINE_LAYER_BUILDER(gaussian);
LBANN_DEFINE_LAYER_BUILDER(hadamard);
LBANN_DEFINE_LAYER_BUILDER(identity_zero);
LBANN_DEFINE_LAYER_BUILDER(in_top_k);
LBANN_DEFINE_LAYER_BUILDER(multidim_reduction);
LBANN_DEFINE_LAYER_BUILDER(permute);
LBANN_DEFINE_LAYER_BUILDER(pooling);
LBANN_DEFINE_LAYER_BUILDER(reduction);
LBANN_DEFINE_LAYER_BUILDER(reshape);
LBANN_DEFINE_LAYER_BUILDER(scatter);
LBANN_DEFINE_LAYER_BUILDER(slice);
LBANN_DEFINE_LAYER_BUILDER(sort);
LBANN_DEFINE_LAYER_BUILDER(split);
LBANN_DEFINE_LAYER_BUILDER(stop_gradient);
LBANN_DEFINE_LAYER_BUILDER(sum);
LBANN_DEFINE_LAYER_BUILDER(tessellate);
LBANN_DEFINE_LAYER_BUILDER(uniform);
LBANN_DEFINE_LAYER_BUILDER(unpooling);
LBANN_DEFINE_LAYER_BUILDER(upsample);
LBANN_DEFINE_LAYER_BUILDER(weighted_sum);
LBANN_DEFINE_LAYER_BUILDER(weights);

} // namespace lbann
#endif // LBANN_LAYERS_TRANSFORM_TRANSFORM_BUILDERS_HPP_INCLUDED
