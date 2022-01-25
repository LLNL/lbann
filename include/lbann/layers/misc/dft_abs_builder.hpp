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
#ifndef LBANN_LAYERS_MATH_DFT_ABS_BUILDER_HPP_
#define LBANN_LAYERS_MATH_DFT_ABS_BUILDER_HPP_

#include <lbann/base.hpp>

#include <memory>

// Forward declarations of Google protobuf classes
namespace lbann_data
{
class Layer;
}// namespace lbann_data

namespace lbann
{

// Forward declarations of LBANN classes
class Layer;
class lbann_comm;

/** @brief Build an dft_abs_layer from a protobuf message.
 *
 *  @note The layout parameter must be here, even though it's not on
 *  the class. This is for a technical reason related to the factory.
*/
template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_dft_abs_layer_from_pbuf(
  lbann_comm*, lbann_data::Layer const&);

}// namespace lbann
#endif // LBANN_LAYERS_MATH_DFT_ABS_BUILDER_HPP_
