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

#define LBANN_UNIFORM_HASH_LAYER_INSTANTIATE
#include "lbann/layers/misc/uniform_hash.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
uniform_hash_layer<TensorDataType, Layout, Device>::uniform_hash_layer(
  lbann_comm* comm)
  : data_type_layer<TensorDataType>(comm)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
uniform_hash_layer<TensorDataType, Layout, Device>*
uniform_hash_layer<TensorDataType, Layout, Device>::copy() const
{
  return new uniform_hash_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string uniform_hash_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "uniform hash";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
uniform_hash_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
uniform_hash_layer<TensorDataType, Layout, Device>::get_device_allocation()
  const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void uniform_hash_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims(this->get_input_dims());
}

// ---------------------------------------------
// Explicit template instantiation
// ---------------------------------------------

#ifdef LBANN_HAS_GPU
#define PROTO(T)                                                               \
  template class uniform_hash_layer<T,                                         \
                                    data_layout::DATA_PARALLEL,                \
                                    El::Device::GPU>;                          \
  template class uniform_hash_layer<T,                                         \
                                    data_layout::MODEL_PARALLEL,               \
                                    El::Device::GPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_HAS_GPU

} // namespace lbann
