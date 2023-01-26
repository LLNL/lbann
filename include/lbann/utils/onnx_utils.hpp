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
#ifndef LBANN_UTILS_ONNX_UTILS_HPP_INCLUDED
#define LBANN_UTILS_ONNX_UTILS_HPP_INCLUDED

#include <lbann_config.hpp>

#ifdef LBANN_HAS_ONNX
#include "El.hpp"

#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/tagged_dispatch.hpp"

#include <onnx/onnx_pb.h>

#include <vector>

namespace lbann {
namespace details {

// TODO: There are two extra considerations. ONNX supports both
// "external data" (stored on disk) and segmented data. Both of these
// could be useful in certain circumstances.

// Basic types first.
inline void set_datatype(onnx::TensorProto& p, TypeTag<float>)
{
  p.set_data_type(onnx::TensorProto::FLOAT);
}
inline void set_datatype(onnx::TensorProto& p, TypeTag<double>)
{
  p.set_data_type(onnx::TensorProto::DOUBLE);
}
inline void set_datatype(onnx::TensorProto& p, TypeTag<El::Complex<float>>)
{
  p.set_data_type(onnx::TensorProto::COMPLEX64);
}
inline void set_datatype(onnx::TensorProto& p, TypeTag<El::Complex<double>>)
{
  p.set_data_type(onnx::TensorProto::COMPLEX128);
}

inline void clear_data(onnx::TensorProto& p, TypeTag<float>)
{
  p.clear_float_data();
}
inline void clear_data(onnx::TensorProto& p, TypeTag<double>)
{
  p.clear_double_data();
}

inline void add_data(onnx::TensorProto& p, float const& x)
{
  p.add_float_data(x);
}
inline void add_data(onnx::TensorProto& p, double const& x)
{
  p.add_double_data(x);
}

// Now FP16 types
#ifdef LBANN_HAS_HALF
inline void set_datatype(onnx::TensorProto& p, TypeTag<cpu_half_type>)
{
  p.set_data_type(onnx::TensorProto::FLOAT);
}
inline void clear_data(onnx::TensorProto& p, TypeTag<cpu_half_type>)
{
  p.clear_float_data();
}
inline void add_data(onnx::TensorProto& p, cpu_half_type const& x)
{
  p.add_float_data(static_cast<float>(x));
}
#if defined LBANN_HAS_GPU && defined LBANN_HAS_GPU_FP16
inline void set_datatype(onnx::TensorProto& p, TypeTag<gpu_half_type>)
{
  p.set_data_type(onnx::TensorProto::FLOAT);
}
inline void clear_data(onnx::TensorProto& p, TypeTag<gpu_half_type>)
{
  p.clear_float_data();
}
inline void add_data(onnx::TensorProto& p, gpu_half_type const& x)
{
  p.add_float_data(static_cast<float>(x));
}
#endif // defined LBANN_HAS_GPU && defined LBANN_HAS_GPU_FP16
#endif // LBANN_HAS_HALF

// Finally, complex types.
template <typename T>
void add_data(onnx::TensorProto& p, El::Complex<T> const& x)
{
  add_data(p, El::RealPart(x));
  add_data(p, El::ImagPart(x));
}

// Clear any data present in the message.
template <typename T>
void clear_msg_data(onnx::TensorProto& p, TypeTag<T>)
{
  p.clear_dims();
  p.clear_data_type();
  // Complex types are stored in their base type.
  clear_data(p, TypeTag<El::Base<T>>{});
}

// Serialize the matrix into the given TensorProto message in
// row-major ordering.
template <typename T>
void serialize_to_onnx_impl(El::AbstractDistMatrix<T> const& m,
                            onnx::TensorProto& p)
{
  using namespace El;
  DistMatrixReadProxy<T, T, STAR, STAR, ELEMENT, Device::CPU> proxy(m);
  auto& mat = proxy.GetLocked().LockedMatrix();
  auto const height = mat.Height();
  auto const width = mat.Width();

  clear_msg_data(p, TypeTag<T>{});
  p.add_dims(height);
  p.add_dims(width);
  set_datatype(p, TypeTag<T>{});
  p.set_data_location(onnx::TensorProto::DEFAULT);
  for (auto r = decltype(height){0}; r < height; ++r)
    for (auto c = decltype(width){0}; c < width; ++c)
      add_data(p, mat.CRef(r, c));
}

template <typename T, typename SizeT>
void serialize_to_onnx_impl(El::AbstractDistMatrix<T> const& m,
                            std::vector<SizeT> const& dims,
                            onnx::TensorProto& p)
{
  using namespace El;
  LBANN_ASSERT(lbann::get_linear_size(dims) == static_cast<size_t>(m.Height()));
  LBANN_ASSERT(m.Width() == static_cast<Int>(1));

  DistMatrixReadProxy<T, T, STAR, STAR, ELEMENT, Device::CPU> proxy(m);
  auto& mat = proxy.GetLocked().LockedMatrix();

  clear_msg_data(p, TypeTag<T>{});
  for (auto const& d : dims)
    p.add_dims(d);
  set_datatype(p, TypeTag<T>{});
  p.set_data_location(onnx::TensorProto::DEFAULT);

  // This may be unnecessary
  auto const height = mat.Height();
  auto const* const buf = mat.LockedBuffer();
  for (auto ii = Int{0}; ii < height; ++ii) {
    add_data(p, buf[ii]);
  }
}

} // namespace details

/** @brief Serialize a DistMatrix to ONNX format.
 *
 *  This function is primarily for serializing the weights
 *  tensors. The exact behavior depends on the dimension arguments.
 *  In all cases, all processes in the matrix's grid must participate
 *  in the call. The serialized values will be written into the
 *  protobuf message on all processes in the grid.
 *
 *  If width_dims is empty, we interpret the matrix as a linearized
 *  packed tensor with folds described by height_dims. The serialized
 *  tensor will have dimensions matching height_dims. This should
 *  match biases as well as kernels for convolution.
 *
 *  Otherwise, we interpret the matrix as a true 2nd order tensor and
 *  serialize it into the ROW-MAJOR format that ONNX expects. The
 *  serialized tensor will have dimensions that match the matrix
 *  dimensions.
 *
 *  The TensorProto message is cleared of data-related content
 *  (data_type, dims, and the actual data).
 *
 *  @param[in] m The distributed matrix to serialize.
 *  @param[in] height_dims The tensor dimensions represented in the
 *                         height of the input matrix.
 *  @param[in] width_dims The tensor dimensions represented in the
 *                        width of the input matrix.
 *  @param[out] p The protobuf message into which to serialize the
 *                matrix.
 */
template <typename T, typename SizeT>
void serialize_to_onnx(El::AbstractDistMatrix<T> const& m,
                       std::vector<SizeT> const& height_dims,
                       std::vector<SizeT> const& width_dims,
                       onnx::TensorProto& p)
{
  if (width_dims.empty())
    details::serialize_to_onnx_impl(m, height_dims, p);
  else {
    LBANN_ASSERT_DEBUG(lbann::get_linear_size(height_dims) ==
                       static_cast<size_t>(m.Height()));
    LBANN_ASSERT_DEBUG(lbann::get_linear_size(width_dims) ==
                       static_cast<size_t>(m.Width()));
    details::serialize_to_onnx_impl(m, p);
  }
}

} // namespace lbann
#endif // LBANN_HAS_ONNX
#endif // LBANN_UTILS_ONNX_UTILS_HPP_INCLUDED
