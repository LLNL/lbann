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

#ifndef LBANN_UTILS_ENTRYWISE_OPERATOR_HPP
#define LBANN_UTILS_ENTRYWISE_OPERATOR_HPP

#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Apply an entry-wise unary operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions.
 */
template <template <typename> class Op, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractMatrix<TensorDataType>& input,
  El::AbstractMatrix<TensorDataType>& output)
{
  using UnaryOperator = Op<TensorDataType>;
  // Check that input and output are valid
  std::stringstream err;
  if (input.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("input is not on CPU");
  }
  else if (output.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("output is not on CPU");
  }
  else if (input.Height() != output.Height() ||
           input.Width() != output.Width()) {
    err << "input matrix dimensions "
        << "(" << input.Height() << " x " << input.Width() << ")"
        << "don't match output matrix dimensions "
        << "(" << output.Height() << " x " << output.Width() << ")";
    LBANN_ERROR(err.str());
  }

  // Apply unary operator
  if (input.Contiguous() && output.Contiguous()) {
    const auto* input_buffer = input.LockedBuffer();
    auto* output_buffer = output.Buffer();
    const size_t size = input.Height() * input.Width();
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < size; ++i) {
      UnaryOperator op;
      output_buffer[i] = op(input_buffer[i]);
    }
  }
  else {
    auto const width = input.Width();
    auto const height = input.Height();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        UnaryOperator op;
        output(row, col) = op(input(row, col));
      }
    }
  }
}

/** Apply an entry-wise binary operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions.
 */
template <template <typename> class Op, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractMatrix<TensorDataType>& input1,
  const El::AbstractMatrix<TensorDataType>& input2,
  El::AbstractMatrix<TensorDataType>& output)
{
  using BinaryOperator = Op<TensorDataType>;
  // Check that input and output are valid
  if (input1.GetDevice() != El::Device::CPU ||
      input2.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("input is not on CPU");
  }
  else if (output.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("output is not on CPU");
  }
  else if (input1.Height() != input2.Height() ||
           input1.Width() != input2.Width() ||
           input1.Height() != output.Height() ||
           input1.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input1.Height(),
                " x ",
                input1.Width(),
                ", ",
                input2.Height(),
                " x ",
                input2.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }

  // Apply binary operator
  if (input1.Contiguous() && input2.Contiguous() && output.Contiguous()) {
    const auto* input1_buffer = input1.LockedBuffer();
    const auto* input2_buffer = input2.LockedBuffer();
    auto* output_buffer = output.Buffer();
    const size_t size = input1.Height() * input1.Width();
    LBANN_OMP_PARALLEL_FOR
    for (size_t i = 0; i < size; ++i) {
      BinaryOperator op;
      output_buffer[i] = op(input1_buffer[i], input2_buffer[i]);
    }
  }
  else {
    auto const width = input1.Width();
    auto const height = input1.Height();
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        BinaryOperator op;
        output(row, col) = op(input1(row, col), input2(row, col));
      }
    }
  }
}

/** Apply an entry-wise unary operator to CPU data.
 *  The input and output data must be on CPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class Op, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& output)
{
  if (input.Height() != output.Height() || input.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input.Height(),
                " x ",
                input.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }
  else if (input.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_unary_operator<Op>(input.LockedMatrix(), output.Matrix());
}

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class Op, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input1,
  const El::AbstractDistMatrix<TensorDataType>& input2,
  El::AbstractDistMatrix<TensorDataType>& output)
{
  if (input1.Height() != input2.Height() || input1.Width() != input2.Width() ||
      input1.Height() != output.Height() || input1.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input1.Height(),
                " x ",
                input1.Width(),
                ", ",
                input2.Height(),
                " x ",
                input2.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }
  else if (input1.DistData() != input2.DistData() ||
           input1.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_binary_operator<Op>(input1.LockedMatrix(),
                                      input2.LockedMatrix(),
                                      output.Matrix());
}

} // namespace lbann

#endif // LBANN_UTILS_ENTRYWISE_OPERATOR_HPP
