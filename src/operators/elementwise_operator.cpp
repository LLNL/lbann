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

#define LBANN_DATA_TYPE_OPERATOR_INSTANTIATE
#include "lbann/operators/elementwise_operator.hpp"
#include "lbann/utils/vector_dynamic_cast.hpp"

namespace lbann {

namespace {

template <typename T>
auto vector_dist_to_local(std::vector<El::AbstractDistMatrix<T>*>const& v_in) -> std::vector<El::AbstractMatrix<T>*>
{
  std::vector<El::AbstractMatrix<T>*> v;
  v.reserve(v_in.size());
  for (auto& e : v_in) {
#ifdef LBANN_DEBUG
    auto* ptr = &(e->Matrix());
    LBANN_ASSERT(ptr);
    v.push_back(ptr);
#else
    v.push_back(&(e->Matrix()));
#endif
  }
  return v;
}

template <typename T>
auto vector_dist_to_local_const(std::vector<El::AbstractDistMatrix<T> const*>const& v_in) -> std::vector<El::AbstractMatrix<T> const*>
{
  std::vector<El::AbstractMatrix<T> const*> v;
  v.reserve(v_in.size());
  for (auto& e : v_in) {
#ifdef LBANN_DEBUG
    auto* ptr = &(e->LockedMatrix());
    LBANN_ASSERT(ptr);
    v.push_back(ptr);
#else
    v.push_back(&(e->LockedMatrix()));
#endif
  }
  return v;
}

} // anonymous

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute(std::vector<InputAbsDistMatrixType const*> inputs,
           std::vector<OutputAbsDistMatrixType*> outputs) const {
  auto local_inputs = vector_dist_to_local_const<InputTensorDataType>(inputs);
  auto local_outputs = vector_dist_to_local<OutputTensorDataType>(outputs);
  fp_compute_local(local_inputs, local_outputs);
};

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute(std::vector<InputAbsDistMatrixType const*> inputs,
           std::vector<OutputAbsDistMatrixType const*> gradient_wrt_outputs,
           std::vector<InputAbsDistMatrixType*> gradient_wrt_inputs) const {
  auto local_inputs = vector_dist_to_local_const<InputTensorDataType>(inputs);
  auto local_gradient_wrt_outputs = vector_dist_to_local_const<OutputTensorDataType>(gradient_wrt_outputs);
  auto local_gradient_wrt_inputs = vector_dist_to_local<InputTensorDataType>(gradient_wrt_inputs);
  bp_compute_local(local_inputs,
                   local_gradient_wrt_outputs,
                   local_gradient_wrt_inputs);
};

/////////////////////////////////////////////////////////////////////////////////
// Local operations:
// Dynamic dispatch between CPU and GPUs
/////////////////////////////////////////////////////////////////////////////////
template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute_local(std::vector<InputAbsMatrixType const*> inputs,
                 std::vector<OutputAbsMatrixType*> outputs) const {

  if(outputs.size() == 0) {
    LBANN_ERROR("operator requires non-zero output vector");
  }
  switch (outputs[0]->GetDevice()) {
  case El::Device::CPU:
    return fp_compute_local(vector_dynamic_cast<InputCPUMatrixType const>(inputs),
                            vector_dynamic_cast<OutputCPUMatrixType>(outputs));
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return fp_compute_local(vector_dynamic_cast<InputGPUMatrixType const>(inputs),
                            vector_dynamic_cast<OutputGPUMatrixType>(outputs));
    break;
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Unknown AbsMatrixType");
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute_local(std::vector<InputAbsMatrixType const*> inputs,
                 std::vector<OutputAbsMatrixType const*> gradient_wrt_outputs,
                 std::vector<InputAbsMatrixType*> gradient_wrt_inputs) const {

  if(gradient_wrt_inputs.size() == 0) {
    LBANN_ERROR("operator requires non-zero gradient_wrt_inputs vector");
  }
  switch (gradient_wrt_inputs[0]->GetDevice()) {
  case El::Device::CPU:
    return bp_compute_local(vector_dynamic_cast<InputCPUMatrixType const>(inputs),
                            vector_dynamic_cast<OutputCPUMatrixType const>(gradient_wrt_outputs),
                            vector_dynamic_cast<InputCPUMatrixType>(gradient_wrt_inputs));
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return bp_compute_local(vector_dynamic_cast<InputGPUMatrixType const>(inputs),
                            vector_dynamic_cast<OutputGPUMatrixType const>(gradient_wrt_outputs),
                            vector_dynamic_cast<InputGPUMatrixType>(gradient_wrt_inputs));
    break;
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Unknown AbsMatrixType");
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute_local(std::vector<InputCPUMatrixType const*> inputs,
                 std::vector<OutputCPUMatrixType*> outputs) const {
  LBANN_ERROR("Unsupported CPU matrix type");
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute_local(std::vector<InputCPUMatrixType const*> inputs,
                 std::vector<OutputCPUMatrixType const*> gradient_wrt_outputs,
                 std::vector<InputCPUMatrixType*> gradient_wrt_inputs) const {
  LBANN_ERROR("Unsupported CPU matrix type");
}

#ifdef LBANN_HAS_GPU
template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute_local(std::vector<InputGPUMatrixType const*> inputs,
                 std::vector<OutputGPUMatrixType*> outputs) const {
  LBANN_ERROR("Unsupported GPU matrix type");
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute_local(std::vector<InputGPUMatrixType const*> inputs,
                 std::vector<OutputGPUMatrixType const*> gradient_wrt_outputs,
                 std::vector<InputGPUMatrixType*> gradient_wrt_inputs) const {
  LBANN_ERROR("Unsupported GPU matrix type");
}
#endif // LBANN_HAS_GPU


#define PROTO(T)                     \
  template class ElementwiseOperator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
