////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/transform/split.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace {

template <typename TensorDataType>
struct accumulate {
  __device__ void operator()(TensorDataType &x, const TensorDataType &y) const {
    x += y;
  }
};

template <typename TensorDataType>
struct accumulate2 {
  __device__ void operator()(TensorDataType &x, const TensorDataType &y,
                             const TensorDataType &z) const {
    x = y + z;
  }
};

template <typename Tensor>
void bp_compute_distconv(Tensor &error_signals,
                         Tensor &prev_error_signals,
                         std::vector<Tensor> &prev_error_signals_siblings,
                         int num_children) {
  using TensorDataType = typename Tensor::data_type;
  switch (num_children) {
    case 0:
      dc::MPIPrintStreamInfo() << "No parent for this sum layer";
      error_signals.zero(dc::get_stream());
      break;
    case 1:
      dc::tensor::Copy(error_signals, prev_error_signals,
                       dc::get_stream());
      break;
    case 2:
      prev_error_signals_siblings.at(0).set_outermost_dimension(
          error_signals.get_shape()[-1]);
      dc::tensor::Transform(error_signals, prev_error_signals,
                            prev_error_signals_siblings.at(0),
                            accumulate2<TensorDataType>(),
                            dc::get_backend().get_stream());
      break;
    default:
      dc::tensor::Copy(error_signals, prev_error_signals,
                       dc::get_stream());
      for (auto &child: prev_error_signals_siblings) {
        child.set_outermost_dimension(error_signals.get_shape()[-1]);
        dc::tensor::Transform(error_signals, child, accumulate<TensorDataType>(),
                              dc::get_backend().get_stream());
      }
  }
}
} // namespace
#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void split_layer<TensorDataType, Layout, Dev>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    bp_compute_distconv(this->get_error_signals_t(),
                        this->get_prev_error_signals_t(),
                        m_prev_error_signals_siblings,
                        this->get_num_children());
    this->copy_out_error_signals();
    if (!this->early_terminate_last_iteration()) {
      return;
    }
  }
#endif // LBANN_HAS_DISTCONV
  auto& gradient_wrt_input = this->get_error_signals();
  if (this->get_num_children() > 0) {
    El::Copy(this->get_prev_error_signals(0), gradient_wrt_input);
  } else {
    El::Zero(gradient_wrt_input);
  }
  for (int i = 1; i < this->get_num_children(); ++i) {
    El::Axpy(TensorDataType{1}, this->get_prev_error_signals(i),
             gradient_wrt_input);
  }
}

template class split_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>;
template class split_layer<DataType, data_layout::MODEL_PARALLEL, El::Device::GPU>;

} // namespace lbann
