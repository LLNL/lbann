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

#define LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE
#include "lbann/layers/learning/fully_connected.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>
  ::setup_matrices(const El::Grid& grid) {
  learning_layer<TensorDataType>::setup_matrices(grid);
  deallocate_matrices();
  if(Dev == El::Device::CPU) {
    if(T_layout == data_layout::MODEL_PARALLEL) {
      // Allocate a MCStarMat (RowSumMat)
      this->m_bias_gradient = new El::DistMatrix<TensorDataType, El::MC, El::STAR, El::ELEMENT, El::Device::CPU>(grid);
    } else if(T_layout == data_layout::DATA_PARALLEL) {
      // Allocate a StarMat
      this->m_bias_gradient = new El::DistMatrix<TensorDataType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>(grid);
    }
  }
}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::CPU>& l) {

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.get_data_type_weights()[0]->get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             TensorDataType(1), linearity.LockedMatrix(), input.LockedMatrix(),
             TensorDataType(0), output.Matrix());
  } else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             TensorDataType(1), linearity, input,
             TensorDataType(0), output);
  }

  // Apply bias if needed
  if(l.m_bias_scaling_factor != TensorDataType(0)) {
    const auto& local_bias = l.get_data_type_weights()[1]->get_values().LockedMatrix();
    auto& local_output = output.Matrix();
    El::IndexDependentMap(local_output,
                          (std::function<TensorDataType(El::Int,El::Int,const TensorDataType&)>)
                          ([&l,&local_bias](El::Int r, El::Int c,const TensorDataType& z)
                           ->TensorDataType {
                            return z + l.m_bias_scaling_factor * local_bias(r, 0);
                          }));
  }

}

/** CPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::CPU>& l) {

  // Matrices
  const auto& linearity = l.get_data_type_weights()[0]->get_values();
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != TensorDataType(0)) {
    data_type_optimizer<TensorDataType>* bias_optimizer = l.get_data_type_weights()[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output,
                 l.m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(
        *l.m_bias_gradient,
        l.m_bias_scaling_factor,
        true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  data_type_optimizer<TensorDataType>* linearity_optimizer = l.get_data_type_weights()[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    DataType dst_scale = TensorDataType(0), gradient_scale = TensorDataType(1);
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      if (l.m_transpose) {
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
      if (l.m_transpose) {
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
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             TensorDataType(1), local_linearity, local_gradient_wrt_output,
             TensorDataType(0), local_gradient_wrt_input);
  } else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             TensorDataType(1), linearity, gradient_wrt_output,
             TensorDataType(0), gradient_wrt_input);
  }

}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.get_data_type_weights()[0]->get_values().LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           TensorDataType(1), local_linearity, local_input,
           TensorDataType(0), local_output);

  // Apply bias if needed
  if(l.m_bias_scaling_factor != TensorDataType(0)) {
    const auto& local_bias = l.get_data_type_weights()[1]->get_values().LockedMatrix();
    El::IndexDependentMap(local_output,
                          (std::function<TensorDataType(El::Int,El::Int,const TensorDataType&)>)
                          ([&l,&local_bias](El::Int r, El::Int c,const TensorDataType& z)
                           ->TensorDataType {
                            return z + l.m_bias_scaling_factor * local_bias(r, 0);
                          }));
  }

}

/** CPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {

  // Matrices
  const auto& local_linearity = l.get_data_type_weights()[0]->get_values().LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != TensorDataType(0)) {
    data_type_optimizer<TensorDataType>* bias_optimizer = l.get_data_type_weights()[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output,
                 l.m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(
        *l.m_bias_gradient,
        l.m_bias_scaling_factor,
        true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  data_type_optimizer<TensorDataType>* linearity_optimizer = l.get_data_type_weights()[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    DataType dst_scale = TensorDataType(0), gradient_scale = TensorDataType(0);
    auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    if (l.m_transpose) {
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
  El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           TensorDataType(1), local_linearity, local_gradient_wrt_output,
           TensorDataType(0), local_gradient_wrt_input);

}

#ifdef LBANN_HAS_GPU
/** GPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.get_data_type_weights()[0]->get_values().LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           TensorDataType(1), local_linearity, local_input,
           TensorDataType(0), local_output);

  // Apply bias if needed
  if(l.m_bias_scaling_factor != TensorDataType(0)) {
    const auto& local_bias = l.get_data_type_weights()[1]->get_values().LockedMatrix();
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
    ones.Resize(local_input.Width(), 1);
    El::Fill(ones, TensorDataType(1));
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             l.m_bias_scaling_factor, local_bias, ones,
             TensorDataType(1), local_output);
  }

}

/** GPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& local_linearity = l.get_data_type_weights()[0]->get_values().LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != TensorDataType(0)) {
    data_type_optimizer<TensorDataType>* bias_optimizer = l.get_data_type_weights()[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      DataType dst_scale = TensorDataType(0), gradient_scale = TensorDataType(0);
      auto& bias_gradient = bias_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      if (local_gradient_wrt_output.Height() < 1
          || local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      } else {
        El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, TensorDataType(1));
        El::Gemv(El::NORMAL,
                 gradient_scale, local_gradient_wrt_output, ones,
                 dst_scale, bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  data_type_optimizer<TensorDataType>* linearity_optimizer = l.get_data_type_weights()[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = TensorDataType(0), gradient_scale = TensorDataType(0);
    auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    if (l.m_transpose) {
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
  El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           TensorDataType(1), local_linearity, local_gradient_wrt_output,
           TensorDataType(0), local_gradient_wrt_input);

}

template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.get_data_type_weights()[0]->get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             TensorDataType(1), linearity.LockedMatrix(), input.LockedMatrix(),
             TensorDataType(0), output.Matrix());
  } else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             TensorDataType(1), linearity, input,
             TensorDataType(0), output);
  }

  // Apply bias if needed
  // Note: local outer product is sufficient, no need for global GEMM
  if(l.m_bias_scaling_factor != TensorDataType(0)) {
    const auto& bias = l.get_data_type_weights()[1]->get_values();
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
    ones.Resize(input.LocalWidth(), 1);
    El::Fill(ones, TensorDataType(1));
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             l.m_bias_scaling_factor, bias.LockedMatrix(), ones,
             TensorDataType(1), output.Matrix());
  }

}

template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& linearity = l.get_data_type_weights()[0]->get_values();
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // Compute gradient w.r.t. bias if needed
  // Note: local GEMV is sufficient, no need for global row sum
  if (l.m_bias_scaling_factor != TensorDataType(0)) {
    data_type_optimizer<TensorDataType>* bias_optimizer = l.get_data_type_weights()[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = TensorDataType(0), gradient_scale = TensorDataType(0);
      auto& bias_gradient = bias_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      if (local_gradient_wrt_output.Height() < 1
          || local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      } else {
        El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, TensorDataType(1));
        El::Gemv(El::NORMAL,
                 gradient_scale, local_gradient_wrt_output, ones,
                 dst_scale, bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  data_type_optimizer<TensorDataType>* linearity_optimizer = l.get_data_type_weights()[0]->get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = TensorDataType(0), gradient_scale = TensorDataType(0);
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient = linearity_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
      if (l.m_transpose) {
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
      if (l.m_transpose) {
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
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             TensorDataType(1), local_linearity, local_gradient_wrt_output,
             TensorDataType(0), local_gradient_wrt_input);
  } else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             TensorDataType(1), linearity, gradient_wrt_output,
             TensorDataType(0), gradient_wrt_input);
  }

}

#endif // LBANN_HAS_GPU

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_compute_impl<TensorDataType>(*this);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::bp_compute() {
  bp_compute_impl<TensorDataType>(*this);
}

template class fully_connected_layer<
  DataType, data_layout::DATA_PARALLEL, El::Device::CPU>;
// template class fully_connected_layer<double, data_layout::DATA_PARALLEL, El::Device::CPU>;
template class fully_connected_layer<
  DataType, data_layout::MODEL_PARALLEL, El::Device::CPU>;
// template class fully_connected_layer<double, data_layout::MODEL_PARALLEL, El::Device::CPU>;

#ifdef LBANN_HAS_GPU
template class fully_connected_layer<
  DataType, data_layout::DATA_PARALLEL, El::Device::GPU>;
// template class fully_connected_layer<double, data_layout::DATA_PARALLEL, El::Device::GPU>;
template class fully_connected_layer<
  DataType, data_layout::MODEL_PARALLEL, El::Device::GPU>;
// template class fully_connected_layer<double, data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
