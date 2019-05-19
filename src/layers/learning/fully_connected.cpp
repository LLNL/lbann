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

#include "lbann/layers/learning/fully_connected.hpp"

namespace lbann {

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
  m_bias_gradient = new MCStarMat<El::Device::CPU>(grid);
}

template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
  m_bias_gradient = new StarMat<El::Device::CPU>(grid);
}

#ifdef LBANN_HAS_GPU
template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
}

template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer::setup_matrices(grid);
  deallocate_matrices();
}
#endif // LBANN_HAS_GPU

/** CPU implementation of forward prop computation. */
template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {

  // Matrices
  const auto& input = get_prev_activations();
  auto& output = get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = m_weights[0]->get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             DataType(1), linearity.LockedMatrix(), input.LockedMatrix(),
             DataType(0), output.Matrix());
  } else {
    El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             DataType(1), linearity, input,
             DataType(0), output);
  }

  // Apply bias if needed
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
    auto& local_output = output.Matrix();
    El::IndexDependentMap(local_output,
                          (std::function<DataType(El::Int,El::Int,const DataType&)>)
                          ([this,&local_bias](El::Int r, El::Int c,const DataType& z)
                           ->DataType {
                            return z + m_bias_scaling_factor * local_bias(r, 0);
                          }));
  }

}

/** CPU implementation of backward prop computation. */
template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {

  // Effective mini-batch size
  const int mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Matrices
  const auto& linearity = m_weights[0]->get_values();
  const auto& input = get_prev_activations();
  const auto& gradient_wrt_output = get_prev_error_signals();
  auto& gradient_wrt_input = get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // Compute gradient w.r.t. bias if needed
  if (m_bias_scaling_factor != DataType(0)) {
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output,
                 m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(
        *m_bias_gradient,
        m_bias_scaling_factor / mini_batch_size,
        true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    DataType dst_scale = DataType(0), gradient_scale = DataType(1);
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      gradient_scale /= mini_batch_size;
      if (m_transpose) {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, local_input, local_gradient_wrt_output,
                 dst_scale, linearity_gradient.Matrix());
      } else {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, local_gradient_wrt_output, local_input,
                 dst_scale, linearity_gradient.Matrix());
      }
    } else {
      auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale);
      gradient_scale /= mini_batch_size;
      if (m_transpose) {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, input, gradient_wrt_output,
                 dst_scale, linearity_gradient);
      } else {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, gradient_wrt_output, input,
                 dst_scale, linearity_gradient);
      }
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             DataType(1), local_linearity, local_gradient_wrt_output,
             DataType(0), local_gradient_wrt_input);
  } else {
    El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             DataType(1), linearity, gradient_wrt_output,
             DataType(0), gradient_wrt_input);
  }

}

/** CPU implementation of forward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {

  // Matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  // Apply linearity
  const auto& local_linearity = m_weights[0]->get_values().LockedMatrix();
  El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           DataType(1), local_linearity, local_input,
           DataType(0), local_output);

  // Apply bias if needed
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
    El::IndexDependentMap(local_output,
                          (std::function<DataType(El::Int,El::Int,const DataType&)>)
                          ([this,&local_bias](El::Int r, El::Int c,const DataType& z)
                           ->DataType {
                            return z + m_bias_scaling_factor * local_bias(r, 0);
                          }));
  }

}

/** CPU implementation of backward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {

  // Effective mini-batch size
  const int mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Matrices
  const auto& local_linearity = m_weights[0]->get_values().LockedMatrix();
  const auto& local_input = get_local_prev_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (m_bias_scaling_factor != DataType(0)) {
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output,
                 m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(
        *m_bias_gradient,
        m_bias_scaling_factor / mini_batch_size,
        true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    DataType dst_scale = DataType(0), gradient_scale = DataType(0);
    auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    gradient_scale /= mini_batch_size;
    if (m_transpose) {
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               gradient_scale, local_input, local_gradient_wrt_output,
               dst_scale, linearity_gradient.Matrix());
    } else {
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               gradient_scale, local_gradient_wrt_output, local_input,
               dst_scale, linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. input
  El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           DataType(1), local_linearity, local_gradient_wrt_output,
           DataType(0), local_gradient_wrt_input);

}

#ifdef LBANN_HAS_GPU
/** GPU implementation of forward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {

  // Matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  // Apply linearity
  const auto& local_linearity = m_weights[0]->get_values().LockedMatrix();
  El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           DataType(1), local_linearity, local_input,
           DataType(0), local_output);

  // Apply bias if needed
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
    GPUMat ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
    ones.Resize(local_input.Width(), 1);
    El::Fill(ones, DataType(1));
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             m_bias_scaling_factor, local_bias, ones,
             DataType(1), local_output);
  }

}

/** GPU implementation of backward prop computation. */
template <>
void fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {

  // Effective mini-batch size
  const int mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Matrices
  const auto& local_linearity = m_weights[0]->get_values().LockedMatrix();
  const auto& local_input = get_local_prev_activations();
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (m_bias_scaling_factor != DataType(0)) {
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      DataType dst_scale = DataType(0), gradient_scale = DataType(0);
      auto& bias_gradient = bias_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      gradient_scale /= mini_batch_size;
      if (local_gradient_wrt_output.Height() < 1
          || local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      } else {
        GPUMat ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, DataType(1));
        El::Gemv(El::NORMAL,
                 gradient_scale, local_gradient_wrt_output, ones,
                 dst_scale, bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    DataType dst_scale = DataType(0), gradient_scale = DataType(0);
    auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    gradient_scale /= mini_batch_size;
    if (m_transpose) {
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               gradient_scale, local_input, local_gradient_wrt_output,
               dst_scale, linearity_gradient.Matrix());
    } else {
      El::Gemm(El::NORMAL, El::TRANSPOSE,
               gradient_scale, local_gradient_wrt_output, local_input,
               dst_scale, linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. input
  El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           DataType(1), local_linearity, local_gradient_wrt_output,
           DataType(0), local_gradient_wrt_input);

}

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {

  // Matrices
  const auto& input = get_prev_activations();
  auto& output = get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = m_weights[0]->get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             DataType(1), linearity.LockedMatrix(), input.LockedMatrix(),
             DataType(0), output.Matrix());
  } else {
    El::Gemm(m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             DataType(1), linearity, input,
             DataType(0), output);
  }

  // Apply bias if needed
  // Note: local outer product is sufficient, no need for global GEMM
  if(m_bias_scaling_factor != DataType(0)) {
    const auto& bias = m_weights[1]->get_values();
    GPUMat ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
    ones.Resize(input.LocalWidth(), 1);
    El::Fill(ones, DataType(1));
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             m_bias_scaling_factor, bias.LockedMatrix(), ones,
             DataType(1), output.Matrix());
  }

}

template <>
void fully_connected_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {

  // Effective mini-batch size
  const int mini_batch_size = this->m_model->get_effective_mini_batch_size();

  // Matrices
  const auto& linearity = m_weights[0]->get_values();
  const auto& input = get_prev_activations();
  const auto& gradient_wrt_output = get_prev_error_signals();
  auto& gradient_wrt_input = get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // Compute gradient w.r.t. bias if needed
  // Note: local GEMV is sufficient, no need for global row sum
  if (m_bias_scaling_factor != DataType(0)) {
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      DataType dst_scale = DataType(0), gradient_scale = DataType(0);
      auto& bias_gradient = bias_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      gradient_scale /= mini_batch_size;
      if (local_gradient_wrt_output.Height() < 1
          || local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      } else {
        GPUMat ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, DataType(1));
        El::Gemv(El::NORMAL,
                 gradient_scale, local_gradient_wrt_output, ones,
                 dst_scale, bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    DataType dst_scale = DataType(0), gradient_scale = DataType(0);
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      gradient_scale /= mini_batch_size;
      if (m_transpose) {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, local_input, local_gradient_wrt_output,
                 dst_scale, linearity_gradient.Matrix());
      } else {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, local_gradient_wrt_output, local_input,
                 dst_scale, linearity_gradient.Matrix());
      }
    } else {
      auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale);
      gradient_scale /= mini_batch_size;
      if (m_transpose) {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, input, gradient_wrt_output,
                 dst_scale, linearity_gradient);
      } else {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 gradient_scale, gradient_wrt_output, input,
                 dst_scale, linearity_gradient);
      }
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             DataType(1), local_linearity, local_gradient_wrt_output,
             DataType(0), local_gradient_wrt_input);
  } else {
    El::Gemm(m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             DataType(1), linearity, gradient_wrt_output,
             DataType(0), gradient_wrt_input);
  }

}
#endif // LBANN_HAS_GPU

} // namespace lbann
