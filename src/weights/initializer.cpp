////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/random.hpp"

#include <weights.pb.h>

#include <sstream>

namespace lbann {

description weights_initializer::get_description() const {
  return description(get_type() + " weights initializer");
}

template <typename TensorDataType>
description constant_initializer<TensorDataType>::get_description() const {
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  desc.add("Value", m_value);
  return desc;
}

template <typename TensorDataType>
void constant_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix) {
  if (m_value == TensorDataType(0.)) {
    El::Zero(matrix);
  } else {
    El::Fill(matrix, m_value);
  }
}

template <typename TensorDataType>
void value_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix) {

  // Check that number of values matches weights matrix
  if (matrix.Height() * matrix.Width() != (El::Int) m_values.size()) {
    std::stringstream err;
    err << "a value initializer with " << m_values.size() << " values "
        << "attempted to initialize a "
        << matrix.Height() << " x " << matrix.Width() << " "
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
  } else {
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
    Synchronize(hydrogen::gpu::DefaultSyncInfo()); /// @todo Use new Hydrogen synchronization semantics when available
#endif // HYDROGEN_HAVE_GPU
  }

}

template <typename TensorDataType>
description uniform_initializer<TensorDataType>::get_description() const {
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  std::stringstream ss;
  ss << "[" << m_min << "," << m_max << ")";
  desc.add("Range", ss.str());
  return desc;
}

template <typename TensorDataType>
void uniform_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix) {
  uniform_fill(matrix, matrix.Height(), matrix.Width(),
               (m_max + m_min) / El::To<TensorDataType>(2),
               (m_max - m_min) / El::To<TensorDataType>(2));
}

template <typename TensorDataType>
description normal_initializer<TensorDataType>::get_description() const {
  auto desc = data_type_weights_initializer<TensorDataType>::get_description();
  desc.add("Mean", m_mean);
  desc.add("Standard deviation", m_standard_deviation);
  return desc;
}

template <typename TensorDataType>
void normal_initializer<TensorDataType>::fill(AbsDistMatrixType& matrix) {
  gaussian_fill(matrix, matrix.Height(), matrix.Width(),
                m_mean, m_standard_deviation);
}

//
// Builder functions
//

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_constant_initializer_from_pbuf(google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Initializer::ConstantInitializer const&>(msg);
  return make_unique<constant_initializer<TensorDataType>>(El::To<TensorDataType>(params.value()));
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_value_initializer_from_pbuf(google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Initializer::ValueInitializer const&>(msg);
  return make_unique<value_initializer<TensorDataType>>(parse_list<TensorDataType>(params.values()));
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_uniform_initializer_from_pbuf(google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Initializer::UniformInitializer const&>(msg);
  const auto& min = El::To<TensorDataType>(params.min());
  const auto& max = El::To<TensorDataType>(params.max());
  if (min != 0.0 || max != 0.0) {
    return make_unique<uniform_initializer<TensorDataType>>(min, max);
  } else {
    return make_unique<uniform_initializer<TensorDataType>>();
  }
}

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_normal_initializer_from_pbuf(google::protobuf::Message const& msg) {
  const auto& params =
    dynamic_cast<lbann_data::Initializer::NormalInitializer const&>(msg);
  const auto& mean = El::To<TensorDataType>(params.mean());
  const auto& standard_deviation = El::To<TensorDataType>(params.standard_deviation());
  if (mean != 0.0 || standard_deviation != 0.0) {
    return make_unique<normal_initializer<TensorDataType>>(mean, standard_deviation);
  } else {
    return make_unique<normal_initializer<TensorDataType>>();
  }
}


#define PROTO(T)                                                             \
  template class data_type_weights_initializer<T>;                           \
  template class constant_initializer<T>;                                    \
  template class value_initializer<T>;                                       \
  template class uniform_initializer<T>;                                     \
  template class normal_initializer<T>;                                      \
  template std::unique_ptr<weights_initializer>                              \
  build_constant_initializer_from_pbuf<T>(google::protobuf::Message const&); \
  template std::unique_ptr<weights_initializer>                              \
  build_value_initializer_from_pbuf<T>(google::protobuf::Message const&);    \
  template std::unique_ptr<weights_initializer>                              \
  build_uniform_initializer_from_pbuf<T>(google::protobuf::Message const&);  \
  template std::unique_ptr<weights_initializer>                              \
  build_normal_initializer_from_pbuf<T>(google::protobuf::Message const&)

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
