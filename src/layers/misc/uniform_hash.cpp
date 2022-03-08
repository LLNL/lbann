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

#define LBANN_UNIFORM_HASH_LAYER_INSTANTIATE
#include "lbann/layers/misc/uniform_hash.hpp"

namespace lbann {

// ---------------------------------------------
// Builder function
// ---------------------------------------------

namespace
{

template <typename T, data_layout L, El::Device D>
struct Builder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR(
      "Attempted to construct uniform_hash_layer ",
      "with invalid parameters ",
      "(TensorDataType=",TypeName<T>(),", ",
      "Layout=",to_string(L),", ",
      "Device=",to_string(D),")");
    return nullptr;
  }
};

#ifdef LBANN_HAS_GPU
template <data_layout Layout>
struct Builder<float,Layout,El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = uniform_hash_layer<float,
                                         Layout,
                                         El::Device::GPU>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
#endif // LBANN_HAS_GPU

#ifdef LBANN_HAS_GPU
template <data_layout Layout>
struct Builder<double,Layout,El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = uniform_hash_layer<double,
                                         Layout,
                                         El::Device::GPU>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
#endif // LBANN_HAS_GPU

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_uniform_hash_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const&)
{
  using BuilderType = Builder<TensorDataType, Layout, Device>;
  return BuilderType::Build(comm);
}

// ---------------------------------------------
// Explicit template instantiation
// ---------------------------------------------

#ifdef LBANN_HAS_GPU
#define PROTO(T)                                        \
  template class uniform_hash_layer<             \
    T, data_layout::DATA_PARALLEL, El::Device::GPU>;    \
  template class uniform_hash_layer<             \
    T, data_layout::MODEL_PARALLEL, El::Device::GPU>
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_HAS_GPU

#define PROTO_DEVICE(T, Device) \
  LBANN_LAYER_BUILDER_ETI(uniform_hash, T, Device)
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

} // namespace lbann
