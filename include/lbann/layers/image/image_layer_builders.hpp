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
#ifndef LBANN_LAYERS_IMAGE_IMAGE_LAYER_BUILDERS_HPP_INCLUDED
#define LBANN_LAYERS_IMAGE_IMAGE_LAYER_BUILDERS_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

LBANN_DEFINE_LAYER_BUILDER(bilinear_resize);
LBANN_DEFINE_LAYER_BUILDER(composite_image_transformation);
LBANN_DEFINE_LAYER_BUILDER(rotation);
LBANN_DEFINE_LAYER_BUILDER(cutout);

} // namespace lbann
#endif // LBANN_LAYERS_IMAGE_IMAGE_LAYER_BUILDERS_HPP_INCLUDED
