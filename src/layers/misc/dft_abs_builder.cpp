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

#include "lbann_config.hpp"
#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/misc/dft_abs_builder.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_FFTW
#include "lbann/layers/misc/dft_abs.hpp"
#endif // LBANN_HAS_FFTW

#include <type_traits>

namespace lbann
{
namespace
{

template <typename T, El::Device D>
struct DFTTypeSupported : std::false_type {};

#ifdef LBANN_HAS_FFTW
#ifdef LBANN_HAS_FFTW_FLOAT
template <>
struct DFTTypeSupported<float, El::Device::CPU> : std::true_type {};
#endif // LBANN_HAS_FFTW_FLOAT
#ifdef LBANN_HAS_FFTW_DOUBLE
template <>
struct DFTTypeSupported<double, El::Device::CPU> : std::true_type {};
#endif // LBANN_HAS_FFTW_DOUBLE

#ifdef LBANN_HAS_GPU
template <>
struct DFTTypeSupported<float, El::Device::GPU> : std::true_type {};
template <>
struct DFTTypeSupported<double, El::Device::GPU> : std::true_type {};
#endif // LBANN_HAS_GPU

template <typename T, El::Device D,
          typename=El::EnableWhen<DFTTypeSupported<T,D>>>
std::unique_ptr<Layer> build_layer(lbann_comm* comm)
{
  return make_unique<dft_abs_layer<T,D>>(comm);
}
#endif // LBANN_HAS_FFTW

template <typename T, El::Device D,
          typename=El::EnableUnless<DFTTypeSupported<T,D>>,
          typename=void>
std::unique_ptr<Layer> build_layer(lbann_comm const* const)
{
  LBANN_ERROR("No FFT support for ", El::TypeName<T>(), " on device ",
              El::DeviceName<D>());
  return nullptr;
}

template <typename, data_layout L, El::Device>
struct Builder
{
  static std::unique_ptr<Layer> build(lbann_comm*)
  {
    LBANN_ERROR("dft_abs layers are only supported in DATA_PARALLEL. "
                "Requested layout: ", to_string(L));
    return nullptr;
  }
};

template <typename T, El::Device D>
struct Builder<T, data_layout::DATA_PARALLEL, D>
{
  static std::unique_ptr<Layer> build(lbann_comm* comm)
  {
    return build_layer<T,D>(comm);
  }
};
}// namespace <anon>

template <typename T, data_layout L, El::Device D>
std::unique_ptr<Layer> build_dft_abs_layer_from_pbuf(
    lbann_comm* comm, lbann_data::Layer const&)
{
  return Builder<T, L, D>::build(comm);
}

#define PROTO_DEVICE(T, Device)                 \
  LBANN_LAYER_BUILDER_ETI(dft_abs, T, Device)
#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann
