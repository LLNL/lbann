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

#ifndef LBANN_LAYERS_MISC_EXTERNAL_HPP_INCLUDED
#define LBANN_LAYERS_MISC_EXTERNAL_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/proto/datatype.pb.h"

namespace lbann
{

typedef Layer* (*external_layer_setup_t)(lbann_data::DataType datatype,
                                         data_layout layout,
                                         El::Device device,
                                         lbann_comm* comm);

/** @brief Create layer from an external library
 *
 *  Expects any number of input tensors. Invokes a shared object (e.g., .so
 *  file) to call the layer.
 */
external_layer_setup_t load_external_library(const std::string& filename,
                                             std::string layer_name);

} // namespace lbann

#endif // LBANN_LAYERS_MISC_EXTERNAL_HPP_INCLUDED
