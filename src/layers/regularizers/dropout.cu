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

#define LBANN_DROPOUT_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
//#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

template <typename TensorDataType, data_layout layout, El::Device device>
void dropout<TensorDataType, layout, device>::fp_compute_gpu() {
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else

    // Matrices
    const auto& input = this->get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& output = this->get_activations();
    auto& local_output = output.Matrix();

    // Do nothing if dropout is disabled or there is no local data
    const auto& mode = this->m_model->get_execution_context().get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::Copy(input, output);
      return;
    }
    if (local_input.Height() < 1 || local_input.Width() < 1) { return; }

    // Initialize DNN library objects
    auto&& input_desc = m_tensors_dnn_desc.get_prev_activations();
    auto&& output_desc = m_tensors_dnn_desc.get_activations();
    size_t size = dnn_lib::get_dropout_reserve_space_size(input_desc);
    m_reserve_space.Resize((size + sizeof(TensorDataType) - 1) / sizeof(TensorDataType), 1);

    // Apply dropout on the GPU
    dnn_lib::dropout_forward(m_dropout_dnn_desc,
                             input_desc,
                             local_input,
                             output_desc,
                             local_output,
                             m_reserve_space);

#endif // LBANN_HAS_DNN_LIB
  }

  void bp_compute_gpu() {
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else

    // Matrices
    const auto& gradient_wrt_output = this->get_prev_error_signals();
    const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
    auto& gradient_wrt_input = this->get_error_signals();
    auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

    // Copy error signal if dropout is disabled
    const auto& mode = this->m_model->get_execution_context().get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::Copy(gradient_wrt_output, gradient_wrt_input);
    } else {
      if (local_gradient_wrt_input.Height() > 0
          && local_gradient_wrt_input.Width() > 0) {
        dnn_lib::dropout_backward(m_dropout_dnn_desc,
                                  m_tensors_dnn_desc.get_prev_error_signals(),
                                  local_gradient_wrt_output,
                                  m_tensors_dnn_desc.get_error_signals(),
                                  local_gradient_wrt_input,
                                  m_reserve_space);
      }
    }
#endif // LBANN_HAS_DNN_LIB
  }

#define PROTO(T)                                         \
  template std::unique_ptr<Layer>                                       \
  build_dropout_layer_from_pbuf<T, data_layout::DATA_PARALLEL, El::Device::GPU>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_dropout_layer_from_pbuf<T, data_layout::MODEL_PARALLEL, El::Device::GPU>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template class dropout<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class dropout<T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#include "lbann/macros/instantiate.hpp"

}// namespace lbann
