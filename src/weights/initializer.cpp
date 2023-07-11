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

#define LBANN_INITIALIZER_INSTANTIATE
#include "lbann/weights/initializer.hpp"

#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/random.hpp"

#include "lbann/proto/weights.pb.h"
#ifdef LBANN_HAS_CNPY
#include <cnpy.h>
#endif // LBANN_HAS_CNPY

#include <sstream>

namespace lbann {

description weights_initializer::get_description() const
{
  return description(get_type() + " weights initializer");
}

template <typename TensorDataType>
description constant_initializer<TensorDataType>::get_description() const
{
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  desc.add("Value", m_value);
  return desc;
}

template <typename TensorDataType>
void constant_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix)
{
  if (m_value == TensorDataType(0.)) {
    El::Zero(matrix);
  }
  else {
    El::Fill(matrix, m_value);
  }
}

template <typename TensorDataType>
void constant_initializer<TensorDataType>::write_proto(
  lbann_data::Initializer& init) const
{
  init.mutable_constant_initializer()->set_value(m_value);
}

template <typename TensorDataType>
void value_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix)
{
  LBANN_CALIPER_MARK_SCOPE("value_initializer::fill");

  // Check that number of values matches weights matrix
  if (matrix.Height() * matrix.Width() != (El::Int)m_values.size()) {
    std::stringstream err;
    err << "a value initializer with " << m_values.size() << " values "
        << "attempted to initialize a " << matrix.Height() << " x "
        << matrix.Width() << " "
        << "weights matrix";
    LBANN_ERROR(err.str());
  }

  // Copy values to a CPU matrix
  // Note: If the weights matrix is on CPU, the CPU matrix is a matrix
  // view. Otherwise, the CPU matrix values are copied to the weights
  // matrix.
  El::Matrix<TensorDataType, El::Device::CPU> matrix_cpu;
  if (matrix.GetLocalDevice() == El::Device::CPU) {
    El::View(matrix_cpu, matrix.Matrix());
  }
  else {
    matrix_cpu.Resize(matrix.LocalHeight(), matrix.LocalWidth());
  }
  auto const width = matrix.LocalWidth();
  auto const height = matrix.LocalHeight();
  LBANN_OMP_PARALLEL_FOR_COLLAPSE2
  for (El::Int local_col = 0; local_col < width; ++local_col) {
    for (El::Int local_row = 0; local_row < height; ++local_row) {
      const auto& global_row = matrix.GlobalRow(local_row);
      const auto& global_col = matrix.GlobalCol(local_col);
      const auto& global_pos = global_row + matrix.Height() * global_col;
      matrix_cpu(local_row, local_col) = m_values[global_pos];
    }
  }
  if (matrix.GetLocalDevice() != El::Device::CPU) {
    El::Copy(matrix_cpu, matrix.Matrix());
#ifdef HYDROGEN_HAVE_GPU
    Synchronize(hydrogen::gpu::DefaultSyncInfo()); /// @todo Use new Hydrogen
                                                   /// synchronization semantics
                                                   /// when available
#endif                                             // HYDROGEN_HAVE_GPU
  }
}

template <typename TensorDataType>
void value_initializer<TensorDataType>::write_proto(
  lbann_data::Initializer& init) const
{
  protobuf::assign_to_repeated(
    *init.mutable_value_initializer()->mutable_values(),
    m_values);
}

template <typename TensorDataType>
void numpy_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix)
{
#ifndef LBANN_HAS_CNPY
  LBANN_ERROR("CNPY not detected");
#else

  // Load NumPy file
  cnpy::NpyArray array = cnpy::npy_load(m_file);
  const size_t num_values = array.num_bytes() / array.word_size;
  if (matrix.Height() * matrix.Width() != (El::Int)num_values) {
    LBANN_ERROR("NumPy weight initializer attempted to initialize a ",
                matrix.Height(),
                " x ",
                matrix.Width(),
                " weights matrix, "
                "but ",
                m_file,
                " contains ",
                num_values,
                " values");
  }
  if (array.fortran_order) {
    LBANN_ERROR("NumPy weight initializer does not support Fortran order ",
                "(error while loading ",
                m_file,
                ")");
  }

  // Extract weight values from NumPy array
  // Note: Consider viewing instead of copying when the array is
  // already in the right datatype.
  std::vector<TensorDataType> values(num_values);
  switch (array.word_size) {
  case 4: {
    const auto* src = array.data<float>();
    auto* dst = values.data();
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < num_values; ++i) {
      dst[i] = src[i];
    }
    break;
  }
  case 8: {
    const auto* src = array.data<double>();
    auto* dst = values.data();
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < num_values; ++i) {
      dst[i] = src[i];
    }
    break;
  }
  default:
    LBANN_ERROR(
      "NumPy weight initializer only supports float32 and float64 data",
      "(error while loading ",
      m_file,
      ")");
  }

  // Construct CPU matrix from weight values
  using CPUMatType = El::DistMatrix<TensorDataType,
                                    El::STAR,
                                    El::STAR,
                                    El::ELEMENT,
                                    El::Device::CPU>;
  CPUMatType cpu_matrix(matrix.Grid(), matrix.Root());
  if (matrix.Width() == 1) {
    cpu_matrix.LockedAttach(matrix.Height(),
                            matrix.Width(),
                            matrix.Grid(),
                            matrix.ColAlign(),
                            matrix.RowAlign(),
                            values.data(),
                            matrix.Height(),
                            matrix.Root());
  }
  else {
    // Weights in fully-connected layer are in Fortran-order. Need to
    // transpose NumPy array before copying in Hydrogen matrix
    if (array.shape.size() != 2) {
      LBANN_ERROR("NumPy weight initializer attempted to initialize a ",
                  matrix.Height(),
                  " x ",
                  matrix.Width(),
                  " weights matrix, "
                  "but ",
                  m_file,
                  " contains a ",
                  array.shape.size(),
                  "-D array");
    }
    if ((El::Int)array.shape[0] != matrix.Height() ||
        (El::Int)array.shape[1] != matrix.Width()) {
      LBANN_ERROR("NumPy weight initializer attempted to initialize a ",
                  matrix.Height(),
                  " x ",
                  matrix.Width(),
                  " weights matrix, "
                  "but ",
                  m_file,
                  " contains a ",
                  array.shape[0],
                  " x ",
                  array.shape[1],
                  " array");
    }
    El::Matrix<TensorDataType, El::Device::CPU> cpu_matrix_trans(
      matrix.Width(),
      matrix.Height(),
      values.data(),
      matrix.Width());
    cpu_matrix.Resize(matrix.Height(), matrix.Width());
    El::Transpose(cpu_matrix_trans, cpu_matrix.Matrix());
  }

  // Copy CPU matrix to weights matrix
  El::Copy(cpu_matrix, matrix);

#endif // LBANN_HAS_CNPY
}

template <typename TensorDataType>
void numpy_initializer<TensorDataType>::write_proto(
  lbann_data::Initializer& init) const
{
  init.mutable_numpy_initializer()->set_file(m_file);
}

template <typename TensorDataType>
description uniform_initializer<TensorDataType>::get_description() const
{
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  std::stringstream ss;
  ss << "[" << m_min << "," << m_max << ")";
  desc.add("Range", ss.str());
  return desc;
}

template <typename TensorDataType>
void uniform_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix)
{
  uniform_fill(matrix,
               matrix.Height(),
               matrix.Width(),
               (m_max + m_min) / El::To<TensorDataType>(2),
               (m_max - m_min) / El::To<TensorDataType>(2));
}

template <typename TensorDataType>
void uniform_initializer<TensorDataType>::write_proto(
  lbann_data::Initializer& init) const
{
  auto* msg = init.mutable_uniform_initializer();
  msg->set_min(m_min);
  msg->set_max(m_max);
}

template <typename TensorDataType>
description normal_initializer<TensorDataType>::get_description() const
{
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  desc.add("Mean", m_mean);
  desc.add("Standard deviation", m_standard_deviation);
  return desc;
}

template <typename TensorDataType>
void normal_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix)
{
  gaussian_fill(matrix,
                matrix.Height(),
                matrix.Width(),
                m_mean,
                m_standard_deviation);
}

template <typename TensorDataType>
void normal_initializer<TensorDataType>::write_proto(
  lbann_data::Initializer& init) const
{
  auto* msg = init.mutable_normal_initializer();
  msg->set_mean(m_mean);
  msg->set_standard_deviation(m_standard_deviation);
}

//
// Builder functions
//

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_constant_initializer_from_pbuf(google::protobuf::Message const& msg)
{
  const auto& params =
    dynamic_cast<lbann_data::Initializer::ConstantInitializer const&>(msg);
  return std::make_unique<constant_initializer<TensorDataType>>(
    El::To<TensorDataType>(params.value()));
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_value_initializer_from_pbuf(google::protobuf::Message const& msg)
{
  const auto& params =
    dynamic_cast<lbann_data::Initializer::ValueInitializer const&>(msg);
  return std::make_unique<value_initializer<TensorDataType>>(
    protobuf::to_vector<TensorDataType>(params.values()));
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_numpy_initializer_from_pbuf(google::protobuf::Message const& msg)
{
  const auto& params =
    dynamic_cast<lbann_data::Initializer::NumpyInitializer const&>(msg);
  return std::make_unique<numpy_initializer<TensorDataType>>(params.file());
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_uniform_initializer_from_pbuf(google::protobuf::Message const& msg)
{
  const auto& params =
    dynamic_cast<lbann_data::Initializer::UniformInitializer const&>(msg);
  const auto& min = El::To<TensorDataType>(params.min());
  const auto& max = El::To<TensorDataType>(params.max());
  if (min != 0.0 || max != 0.0) {
    return std::make_unique<uniform_initializer<TensorDataType>>(min, max);
  }
  else {
    return std::make_unique<uniform_initializer<TensorDataType>>();
  }
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_normal_initializer_from_pbuf(google::protobuf::Message const& msg)
{
  const auto& params =
    dynamic_cast<lbann_data::Initializer::NormalInitializer const&>(msg);
  const auto& mean = El::To<TensorDataType>(params.mean());
  const auto& standard_deviation =
    El::To<TensorDataType>(params.standard_deviation());
  if (mean != 0.0 || standard_deviation != 0.0) {
    return std::make_unique<normal_initializer<TensorDataType>>(
      mean,
      standard_deviation);
  }
  else {
    return std::make_unique<normal_initializer<TensorDataType>>();
  }
}

#define PROTO(T)                                                               \
  template class data_type_weights_initializer<T>;                             \
  template class constant_initializer<T>;                                      \
  template class value_initializer<T>;                                         \
  template class numpy_initializer<T>;                                         \
  template class uniform_initializer<T>;                                       \
  template class normal_initializer<T>;                                        \
  template std::unique_ptr<weights_initializer>                                \
  build_constant_initializer_from_pbuf<T>(google::protobuf::Message const&);   \
  template std::unique_ptr<weights_initializer>                                \
  build_value_initializer_from_pbuf<T>(google::protobuf::Message const&);      \
  template std::unique_ptr<weights_initializer>                                \
  build_numpy_initializer_from_pbuf<T>(google::protobuf::Message const&);      \
  template std::unique_ptr<weights_initializer>                                \
  build_uniform_initializer_from_pbuf<T>(google::protobuf::Message const&);    \
  template std::unique_ptr<weights_initializer>                                \
  build_normal_initializer_from_pbuf<T>(google::protobuf::Message const&)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
