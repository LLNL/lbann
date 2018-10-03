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

template <typename DataType>
struct accumulate {
  __device__ void operator()(DataType &x, DataType &y) const {
    x += y;
  }
};

template <typename DataType>
struct accumulate2 {
  __device__ void operator()(DataType &x, DataType &y, DataType &z) const {
    x = y + z;
  }
};

void bp_compute_distconv(dc::TensorDev &error_signals,
                         dc::TensorDev &prev_error_signals,
                         std::vector<dc::TensorDev> &prev_error_signals_siblings,
                         int num_children) {
  switch (num_children) {
    case 0:
      dc::MPIPrintStreamInfo() << "No parent for this sum layer";
      error_signals.zero();
      break;
    case 1:
      dc::tensor::Copy(error_signals, prev_error_signals);
      break;
    case 2:
      prev_error_signals_siblings.at(0).set_outermost_dimension(
          error_signals.get_shape()[-1]);
      dc::tensor::Transform(error_signals, prev_error_signals,
                            prev_error_signals_siblings.at(0),
                            accumulate2<DataType>(),
                            dc::get_backend().get_stream());
      break;
    default:
      dc::tensor::Copy(error_signals, prev_error_signals);
      for (auto &child: prev_error_signals_siblings) {
        child.set_outermost_dimension(error_signals.get_shape()[-1]);
        dc::tensor::Transform(error_signals, child, accumulate<DataType>(),
                              dc::get_backend().get_stream());
      }
  }
}
} // namespace
#endif // LBANN_HAS_DISTCONV

template <>
void split_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    bp_compute_distconv(m_error_signals_t, m_prev_error_signals_t,
                        m_prev_error_signals_siblings,
                        get_num_children());
    copy_out_error_signals();
    if (!early_terminate_last_iteration()) {
      return;
    }
  }
#endif
  auto& gradient_wrt_input = get_error_signals();
  if (get_num_children() > 0) {
    El::Copy(get_prev_error_signals(0), gradient_wrt_input);
  } else {
    El::Zero(gradient_wrt_input);
  }
  for (int i = 1; i < get_num_children(); ++i) {
    El::Axpy(DataType(1), get_prev_error_signals(i),
             gradient_wrt_input);
  }
}

} // namespace lbann
