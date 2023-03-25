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

#include "lbann/utils/dnn_lib/onednn.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann_config.hpp"

#ifdef LBANN_HAS_ONEDNN
namespace lbann {
namespace onednn {
namespace /* <anon> */
{
template <El::Device D>
struct DeviceEnumMapT;

template <El::Device D>
inline static constexpr auto DeviceEnumMap = DeviceEnumMapT<D>::value;

template <>
struct DeviceEnumMapT<El::Device::CPU>
  : std::integral_constant<dnnl::engine::kind, dnnl::engine::kind::cpu>
{
};

#ifdef LBANN_HAS_GPU
template <>
struct DeviceEnumMapT<El::Device::GPU>
  : std::integral_constant<dnnl::engine::kind, dnnl::engine::kind::gpu>
{
};
#endif // LBANN_HAS_GPU

} // namespace

template <El::Device D>
dnnl::engine& get_device_engine()
{
  auto constexpr kind = DeviceEnumMap<D>;
  static dnnl::engine engine(kind, /*index=*/0);
  return engine;
}

template <El::Device D>
dnnl::stream get_stream(dnnl::engine const& e, El::SyncInfo<D> const&)
{
  return dnnl::stream(e);
}

template dnnl::engine& get_device_engine<El::Device::CPU>();
template dnnl::stream get_stream(dnnl::engine const&,
                                 El::SyncInfo<El::Device::CPU> const&);
#ifdef LBANN_HAS_GPU
template dnnl::engine& get_device_engine<El::Device::GPU>();
template dnnl::stream get_stream(dnnl::engine const&,
                                 El::SyncInfo<El::Device::GPU> const&);
#endif // LBANN_HAS_GPU

} // namespace onednn
} // namespace lbann
#endif // LBANN_HAS_ONEDNN
