////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_SRC_OPERATORS_MATH_UNIT_TEST_OPERATORTRAITS_HPP_INCLUDED
#define LBANN_SRC_OPERATORS_MATH_UNIT_TEST_OPERATORTRAITS_HPP_INCLUDED

#include <lbann/operators/operator.hpp>

namespace lbann {

/** @brief The data type for data-parallel computation */
template <typename T, El::Device D>
using DataParallelMatrixType =
  El::DistMatrix<T, El::Dist::STAR, El::Dist::VC, El::DistWrap::ELEMENT, D>;
template <typename T, El::Device D>
using ModelParallelMatrixType =
  El::DistMatrix<T, El::Dist::MC, El::Dist::MR, El::DistWrap::ELEMENT, D>;

template <typename OpT>
struct OperatorTraits;

template <typename InputT, typename OutputT, El::Device D>
struct OperatorTraits<Operator<InputT, OutputT, D>>
{
  using input_value_type = InputT;
  using output_value_type = OutputT;
  using base_type = Operator<InputT, OutputT, D>;
  using input_data_parallel_mat_type = DataParallelMatrixType<InputT, D>;
  using output_data_parallel_mat_type = DataParallelMatrixType<OutputT, D>;
  using input_model_parallel_mat_type = ModelParallelMatrixType<InputT, D>;
  using output_model_parallel_mat_type = ModelParallelMatrixType<OutputT, D>;
  using input_tensor_type = utils::DistTensorView<InputT, D>;
  using output_tensor_type = utils::DistTensorView<OutputT, D>;
  using input_const_tensor_type = utils::ConstDistTensorView<InputT, D>;
  using output_const_tensor_type = utils::ConstDistTensorView<OutputT, D>;
  static constexpr El::Device device = D;
};

template <typename OpT>
constexpr El::Device Device = OperatorTraits<OpT>::device;

template <typename OpT>
using InputValueType = typename OperatorTraits<OpT>::input_value_type;
template <typename OpT>
using OutputValueType = typename OperatorTraits<OpT>::output_value_type;
template <typename OpT>

using BaseOperatorType = typename OperatorTraits<OpT>::base_type;
template <typename OpT>

using InputDataParallelMatType =
  typename OperatorTraits<OpT>::input_data_parallel_mat_type;
template <typename OpT>
using OutputDataParallelMatType =
  typename OperatorTraits<OpT>::output_data_parallel_mat_type;

template <typename OpT>
using InputModelParallelMatType =
  typename OperatorTraits<OpT>::input_model_parallel_mat_type;
template <typename OpT>
using OutputModelParallelMatType =
  typename OperatorTraits<OpT>::output_model_parallel_mat_type;

template <typename OpT>
using InputTensorType = typename OperatorTraits<OpT>::input_tensor_type;
template <typename OpT>
using OutputTensorType = typename OperatorTraits<OpT>::output_tensor_type;

template <typename OpT>
using InputConstTensorType =
  typename OperatorTraits<OpT>::input_const_tensor_type;
template <typename OpT>
using OutputConstTensorType =
  typename OperatorTraits<OpT>::output_const_tensor_type;

} // namespace lbann

#endif // LBANN_SRC_OPERATORS_MATH_UNIT_TEST_OPERATORTRAITS_HPP_INCLUDED
