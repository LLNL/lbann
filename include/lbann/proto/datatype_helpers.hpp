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

#ifndef LBANN_PROTO_DATATYPE_HELPERS_HPP_INCLUDED
#define LBANN_PROTO_DATATYPE_HELPERS_HPP_INCLUDED

#include "lbann/base.hpp"

#include "lbann/proto/datatype.pb.h"

namespace lbann {
namespace proto {

template <typename T>
struct TypeToProtoDataType;

template <>
struct TypeToProtoDataType<float>
{
  static constexpr auto value = lbann_data::FLOAT;
};

template <>
struct TypeToProtoDataType<double>
{
  static constexpr auto value = lbann_data::DOUBLE;
};

template <>
struct TypeToProtoDataType<El::Complex<float>>
{
  static constexpr auto value = lbann_data::COMPLEX_FLOAT;
};

template <>
struct TypeToProtoDataType<El::Complex<double>>
{
  static constexpr auto value = lbann_data::COMPLEX_DOUBLE;
};

#ifdef LBANN_HAS_HALF
template <>
struct TypeToProtoDataType<cpu_fp16>
{
  static constexpr auto value = lbann_data::FP16;
};
#endif // LBANN_HAS_HALF

#ifdef LBANN_HAS_GPU_FP16
template <>
struct TypeToProtoDataType<fp16>
{
  static constexpr auto value = lbann_data::FP16;
};
#endif // LBANN_HAS_GPU_FP16

template <typename T>
auto ProtoDataType = TypeToProtoDataType<T>::value;

/**
 * Resolve a datatype, which may be DEFAULT_DATATYPE, to its actual
 * proto type. The DEFAULT_DATATYPE is mapped to LBANN's DataType.
 */
inline lbann_data::DataType resolve_default_datatype(lbann_data::DataType datatype) {
  if (datatype == lbann_data::DEFAULT_DATATYPE) {
    datatype = ProtoDataType<DataType>;
  }
  return datatype;
}

template <El::Device D>
struct DeviceToProtoDevice;

template <>
struct DeviceToProtoDevice<El::Device::CPU>
{
  static constexpr auto value = lbann_data::CPU;
};

#ifdef LBANN_HAS_GPU
template <>
struct DeviceToProtoDevice<El::Device::GPU>
{
  static constexpr auto value = lbann_data::GPU;
};
#endif

template <El::Device D>
constexpr auto ProtoDevice = DeviceToProtoDevice<D>::value;

inline constexpr lbann_data::DeviceAllocation
resolve_default_device(lbann_data::DeviceAllocation in)
{
  constexpr auto default_device =
#ifdef LBANN_HAS_GPU
    ProtoDevice<El::Device::GPU>
#else
    ProtoDevice<El::Device::CPU>
#endif // LBANN HAS_GPU
    ;
  return (in == lbann_data::DEFAULT_DEVICE ? default_device : in);
}

} // namespace proto
} // namespace lbann
#endif /* LBANN_PROTO_DATATYPE_HELPERS_HPP_INCLUDED */
