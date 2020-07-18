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

#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include <layers.pb.h>

#include <string>
#include <sstream>

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer(
  lbann_comm *comm,
  int output_size,
  bool transpose,
  WeightsType* weight,
  bool has_bias)
  : learning_layer<TensorDataType>(comm),
  m_bias_gradient(nullptr),
  m_transpose(transpose) {

  // Initialize output tensor dimensions
  this->set_output_dims({output_size});

  // Initialize bias
  m_bias_scaling_factor = (has_bias
                           ? El::TypeTraits<TensorDataType>::One()
                           : El::TypeTraits<TensorDataType>::Zero());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer(
  const fully_connected_layer& other)
  : learning_layer<TensorDataType>(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor),
    m_transpose(other.m_transpose) {

  // Deep matrix copies
  m_bias_gradient = other.m_bias_gradient;
  if (m_bias_gradient != nullptr) {
    m_bias_gradient = m_bias_gradient->Copy();
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
auto fully_connected_layer<TensorDataType, T_layout, Dev>::operator=(
  const fully_connected_layer& other) -> fully_connected_layer& {
  learning_layer<TensorDataType>::operator=(other);
  m_bias_scaling_factor = other.m_bias_scaling_factor;
  m_transpose = other.m_transpose;

  // Deep matrix copies
  deallocate_matrices();
  m_bias_gradient = other.m_bias_gradient;
  if (m_bias_gradient != nullptr) {
    m_bias_gradient = m_bias_gradient->Copy();
  }

  return *this;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::~fully_connected_layer() {
  deallocate_matrices();
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
description
fully_connected_layer<TensorDataType, T_layout, Dev>::get_description() const {
  auto desc = learning_layer<TensorDataType>::get_description();
  const auto& bias_str = (m_bias_scaling_factor == El::TypeTraits<TensorDataType>::Zero()
                          ? "disabled"
                          : "enabled");
  desc.add("Bias", bias_str);
  return desc;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>
::setup_matrices(const El::Grid& grid) {
  learning_layer<TensorDataType>::setup_matrices(grid);
  deallocate_matrices();
  if(Dev == El::Device::CPU) {
    if(T_layout == data_layout::MODEL_PARALLEL) {
      // Allocate a MCStarMat (RowSumMat)
      this->m_bias_gradient =
        new El::DistMatrix<TensorDataType,
                           El::MC, El::STAR,
                           El::ELEMENT,
                           El::Device::CPU>(grid);
    } else if(T_layout == data_layout::DATA_PARALLEL) {
      // Allocate a StarMat
      this->m_bias_gradient =
        new El::DistMatrix<TensorDataType,
                           El::STAR, El::STAR,
                           El::ELEMENT,
                           El::Device::CPU>(grid);
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>
::setup_data(size_t max_mini_batch_size) {
  learning_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize default weights if none are provided
  if (this->num_weights() > 2) {
    LBANN_ERROR("attempted to setup ", this->get_name(), " with an invalid number of weights");
  }
  if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    this->set_num_data_type_weights(2);
  } else {
    this->set_num_data_type_weights(1);
  }
  if (!this->has_data_type_weights(0)) {
    auto w = make_unique<WeightsType>(this->get_comm());
    //w->set_resources(*(this->subgrid_ranks));
    auto init = make_unique<he_initializer<TensorDataType>>(probability_distribution::gaussian);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_linearity_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->set_data_type_weights(0, w.get());
    this->m_model->add_weights(std::move(w));
  }
  auto& linearity_weights = this->get_data_type_weights(0);

  // Initialize variance scaling initialization
  auto* cast_initializer
    = dynamic_cast<variance_scaling_initializer<TensorDataType>*>(linearity_weights.get_initializer());
  if (cast_initializer != nullptr) {
    cast_initializer->set_fan_in(this->get_input_size());
    cast_initializer->set_fan_out(this->get_output_size());
  }

  // Setup linearity weights
  auto linearity_dist = this->get_prev_activations().DistData();
  if (linearity_dist.colDist != El::MC
      || linearity_dist.rowDist != El::MR) {
    linearity_dist.colDist = El::STAR;
    linearity_dist.rowDist = El::STAR;
  }
  if (m_transpose) {
    linearity_weights.set_dims(this->get_input_dims(), this->get_output_dims());
  } else {
    linearity_weights.set_dims(this->get_output_dims(), this->get_input_dims());
  }
  linearity_weights.set_matrix_distribution(linearity_dist);

  // Set up bias if needed.
  if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    if (!this->has_data_type_weights(1)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      //w->set_resources(*(this->subgrid_ranks));
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias_weights");
      w->set_optimizer(std::move(opt));
      this->set_data_type_weights(1, w.get());
      this->m_model->add_weights(std::move(w));
    }
    auto& bias_weights = this->get_data_type_weights(1);
    // Setup bias weights
    auto bias_dist = this->get_activations().DistData();
    bias_dist.rowDist = El::STAR;
    bias_weights.set_dims(this->get_output_dims());
    bias_weights.set_matrix_distribution(bias_dist);
    if (this->m_bias_gradient != nullptr) {
      El::Zeros(*this->m_bias_gradient,
                bias_weights.get_matrix_height(),
                bias_weights.get_matrix_width());
    }
  }

  // Initialize freeze state
  for (auto&& w : this->get_data_type_weights()) {
    if (this->m_frozen) {
      w->freeze();
    } else {
      w->unfreeze();
    }
  }
  for (auto&& w : this->get_data_type_weights()) {
    if (w->is_frozen() != this->is_frozen()) {
      LBANN_ERROR((this->is_frozen() ? "" : "un"), "frozen ",
                  "layer \"", this->get_name(), "\" has ",
                  (w->is_frozen() ? "" : "un"), "frozen ",
                  "weights \"", w->get_name(), "\"");
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
  const auto& linearity = l.get_data_type_weights(0).get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), linearity.LockedMatrix(), input.LockedMatrix(),
             El::TypeTraits<TensorDataType>::Zero(), output.Matrix());
  } else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), linearity, input,
             El::TypeTraits<TensorDataType>::Zero(), output);
  }

  // Apply bias if needed
  if(l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.get_data_type_weights(1).get_values().LockedMatrix();
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
  const auto& linearity = l.get_data_type_weights(0).get_values();
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // auto parents = l.get_parent_layers();
  // auto parent = parents[0];

  // if(parent->get_type()=="sum")
  // {
  //   std::cout<<"Running Error layer name:"<<l.get_name()<<" BP grad wrt output Height:"<<gradient_wrt_output.LocalHeight()<<" Width:"<<gradient_wrt_output.LocalWidth()<<"\n";
  // }
  // else{
  //   std::cout<<"Running Error layer name:"<<l.get_name()<<" BP grad wrt output Height:"<<gradient_wrt_output.LocalHeight()<<" Width:"<<gradient_wrt_output.LocalWidth()<<"\n";

  // }

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_data_type_weights(1).get_optimizer();
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
  auto* linearity_optimizer = l.get_data_type_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::One();
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
             El::TypeTraits<TensorDataType>::One(), local_linearity, local_gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(), local_gradient_wrt_input);
  } else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), linearity, gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(), gradient_wrt_input);
  }

}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {
  // const auto& myoutput = l.get_activations();
  // const auto& input = l.get_prev_activations();

  //std::cout<<"Running FP(Grid check) Layer name:"<< l.get_name()<<" grad wrt output Height:"<<l.get_local_prev_error_signals().Height()<<" Width:"<<l.get_local_prev_error_signals().Width()<<" Grid Size"<<l.get_prev_error_signals().Grid().Size()<< " input grid size"<<input.Grid().Size()<< " out gird size"<<myoutput.Grid().Size()<<"\n";
  

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.get_data_type_weights(0).get_values().LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(), local_linearity, local_input,
           El::TypeTraits<TensorDataType>::Zero(), local_output);

  // Apply bias if needed
  if(l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.get_data_type_weights(1).get_values().LockedMatrix();
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
  const auto& local_linearity = l.get_data_type_weights(0).get_values().LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // const auto& myoutput = l.get_activations();
  // const auto& input = l.get_prev_activations();


  auto childs = l.get_child_layers();


  // if(childs[0]->get_type()=="sum")
  // {
  //   std::cout<<"Running Error BP FC(DP) grad wrt output Height:"<<local_gradient_wrt_output.Height()<<" Width:"<<local_gradient_wrt_output.Width()<<" Grid Size"<<l.get_prev_error_signals().Grid().Size()<< " input grid size"<<input.Grid().Size()<< " out gird size"<<myoutput.Grid().Size()<<"\n";
  // }
  // if(childs[0]->get_type()=="sum")
  // {
  //   std::cout<<"Running Size Error layer name:"<<l.get_name()<<" BP grad wrt output Height:"<<local_gradient_wrt_output.Height()<<" Width:"<<local_gradient_wrt_output.Width()<<"\n";
  // }
  // else{
  //   std::cout<<"Running Size Error layer name:"<<l.get_name()<<" BP grad wrt output Height:"<<local_gradient_wrt_output.Height()<<" Width:"<<local_gradient_wrt_output.Width()<<"\n";

  // }

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_data_type_weights(1).get_optimizer();
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
  auto* linearity_optimizer = l.get_data_type_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
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
           El::TypeTraits<TensorDataType>::One(), local_linearity, local_gradient_wrt_output,
           El::TypeTraits<TensorDataType>::Zero(), local_gradient_wrt_input);

}

#ifdef LBANN_HAS_GPU
/** GPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.get_data_type_weights(0).get_values().LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(), local_linearity, local_input,
           El::TypeTraits<TensorDataType>::Zero(), local_output);

  // Apply bias if needed
  if(l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.get_data_type_weights(1).get_values().LockedMatrix();
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
    ones.Resize(local_input.Width(), 1);
    El::Fill(ones, El::TypeTraits<TensorDataType>::One());
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             l.m_bias_scaling_factor, local_bias, ones,
             El::TypeTraits<TensorDataType>::One(), local_output);
  }

}

/** GPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& local_linearity = l.get_data_type_weights(0).get_values().LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_data_type_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
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
        El::Fill(ones, El::TypeTraits<TensorDataType>::One());
        El::Gemv(El::NORMAL,
                 gradient_scale, local_gradient_wrt_output, ones,
                 dst_scale, bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  auto* linearity_optimizer = l.get_data_type_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
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
           El::TypeTraits<TensorDataType>::One(), local_linearity, local_gradient_wrt_output,
           El::TypeTraits<TensorDataType>::Zero(), local_gradient_wrt_input);

}

template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.get_data_type_weights(0).get_values();
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), linearity.LockedMatrix(), input.LockedMatrix(),
             El::TypeTraits<TensorDataType>::Zero(), output.Matrix());
  } else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), linearity, input,
             El::TypeTraits<TensorDataType>::Zero(), output);
  }

  // Apply bias if needed
  // Note: local outer product is sufficient, no need for global GEMM
  if(l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& bias = l.get_data_type_weights(1).get_values();
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
    ones.Resize(input.LocalWidth(), 1);
    El::Fill(ones, El::TypeTraits<TensorDataType>::One());
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             l.m_bias_scaling_factor, bias.LockedMatrix(), ones,
             El::TypeTraits<TensorDataType>::One(), output.Matrix());
  }

}

template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType, data_layout::MODEL_PARALLEL, El::Device::GPU>& l) {

  // Matrices
  const auto& linearity = l.get_data_type_weights(0).get_values();
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

  // Compute gradient w.r.t. bias if needed
  // Note: local GEMV is sufficient, no need for global row sum
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_data_type_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
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
        El::Fill(ones, El::TypeTraits<TensorDataType>::One());
        El::Gemv(El::NORMAL,
                 gradient_scale, local_gradient_wrt_output, ones,
                 dst_scale, bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  auto* linearity_optimizer = l.get_data_type_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
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
             El::TypeTraits<TensorDataType>::One(), local_linearity, local_gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(), local_gradient_wrt_input);
  } else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), linearity, gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(), gradient_wrt_input);
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

template <typename TensorDataType, data_layout layout, El::Device device>
std::unique_ptr<Layer> build_fully_connected_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& layer_msg)
{
  using LayerType = fully_connected_layer<TensorDataType, layout, device>;
  const auto& params = layer_msg.fully_connected();
  return lbann::make_unique<LayerType>(
    comm,
    params.num_neurons(),
    params.transpose(),
    nullptr,
    params.has_bias());
}

#define PROTO_DEVICE(T, Device) \
  template class fully_connected_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class fully_connected_layer<T, data_layout::MODEL_PARALLEL, Device>; \
  template std::unique_ptr<Layer>                                       \
  build_fully_connected_layer_from_pbuf<T, data_layout::DATA_PARALLEL, Device>( \
    lbann_comm*, lbann_data::Layer const&);                             \
  template std::unique_ptr<Layer>                                       \
  build_fully_connected_layer_from_pbuf<T, data_layout::MODEL_PARALLEL, Device>( \
    lbann_comm*, lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
