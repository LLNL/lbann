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

namespace lbann {


template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute(InputAbsDistMatrixType const& input,
           OutputAbsDistMatrixType& output) const {
  fp_compute_local(input.LockedMatrix(), output.Matrix());
};

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute(InputAbsDistMatrixType const& input,
           OutputAbsDistMatrixType const& gradient_wrt_output,
           InputAbsDistMatrixType& gradient_wrt_input) const {
  bp_compute_local(input.LockedMatrix(),
                   gradient_wrt_output.LockedMatrix(),
                   gradient_wrt_input.Matrix());
};

/////////////////////////////////////////////////////////////////////////////////
// Local operations:
// Dynamic dispatch between CPU and GPUs
/////////////////////////////////////////////////////////////////////////////////
template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute_local(InputAbsMatrixType const& input,
                 OutputAbsMatrixType& output) const {

  switch (output.GetDevice()) {
  case El::Device::CPU:
    return fp_compute_local(static_cast<InputCPUMatrixType const&>(input),
                            static_cast<OutputCPUMatrixType&>(output));
    break;
  case El::Device::GPU:
    return fp_compute_local(static_cast<InputGPUMatrixType const&>(input),
                            static_cast<OutputGPUMatrixType&>(output));
    break;
  default:
    LBANN_ERROR("Unknown AbsMatrixType");
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute_local(InputAbsMatrixType const& input,
                 OutputAbsMatrixType const& gradient_wrt_output,
                 InputAbsMatrixType& gradient_wrt_input) const {
  switch (gradient_wrt_input.GetDevice()) {
  case El::Device::CPU:
    return bp_compute_local(static_cast<InputCPUMatrixType const&>(input),
                            static_cast<OutputCPUMatrixType const&>(gradient_wrt_output),
                            static_cast<InputCPUMatrixType&>(gradient_wrt_input));
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return bp_compute_local(static_cast<InputGPUMatrixType const&>(input),
                            static_cast<OutputGPUMatrixType const&>(gradient_wrt_output),
                            static_cast<InputGPUMatrixType&>(gradient_wrt_input));
    break;
#endif // LBANN_HAS_GPU
  default:
    LBANN_ERROR("Unknown AbsMatrixType");
  }
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute_local(InputCPUMatrixType const& input,
                 OutputCPUMatrixType& output) const {
  LBANN_ERROR("Unsupported CPU matrix type");
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute_local(InputCPUMatrixType const& input,
                 OutputCPUMatrixType const& gradient_wrt_output,
                 InputCPUMatrixType& gradient_wrt_input) const {
  LBANN_ERROR("Unsupported CPU matrix type");
}

#ifdef LBANN_HAS_GPU
template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
fp_compute_local(InputGPUMatrixType const& input,
                 OutputGPUMatrixType& output) const {
  LBANN_ERROR("Unsupported GPU matrix type");
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void ElementwiseOperator<InputTensorDataType, OutputTensorDataType>::
bp_compute_local(InputGPUMatrixType const& input,
                 OutputGPUMatrixType const& gradient_wrt_output,
                 InputGPUMatrixType& gradient_wrt_input) const {
  LBANN_ERROR("Unsupported GPU matrix type");
}
#endif // LBANN_HAS_GPU


#define PROTO(T)                     \
  template class ElementwiseOperator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
