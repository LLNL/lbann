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

#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/utils/exception.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/cublas.hpp"
#endif // LBANN_HAS_GPU

namespace lbann {

namespace {

/** CPU implementation of evaluation layer forward prop. */
void fp_cpu(lbann_comm& comm,
            const AbsDistMat& input,
            DataType& value,
            Al::request& req) {
  const auto& local_input = input.LockedMatrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();
  const auto& mini_batch_size = input.Width();
  value = 0;
  LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:value) collapse(2))
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      value += local_input(row, col);
    }
  }
  value = value / mini_batch_size;
  comm.nb_allreduce(&value, 1, input.DistComm(), req);
}

#ifdef LBANN_HAS_GPU
/** GPU implementation of evaluation layer forward prop. */
void fp_gpu(lbann_comm& comm,
            const AbsDistMat& input,
            DataType& value,
            cuda::event_wrapper& copy_event) {
  constexpr DataType zero = 0;
  constexpr DataType one = 1;

  // Local matrix
  const auto& local_input = input.LockedMatrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();
  const auto& mini_batch_size = input.Width();

  // GPU objects
  GPUMat sum_d, ones_d;
#ifdef HYDROGEN_HAVE_CUB
  sum_d.SetMemoryMode(1);  // Use CUB GPU memory pool
  ones_d.SetMemoryMode(1); // Use CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
  sum_d.Resize(1, 1);
  auto&& handle = El::GPUManager::cuBLASHandle();
  auto&& stream = El::GPUManager::Stream();
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

  // Compute sum of local input matrix entries
  if (local_input.IsEmpty()) {
    El::Zero(sum_d);
  } else if (local_input.Contiguous()) {
    ones_d.Resize(local_height * local_width, 1);
    El::Fill(ones_d, one);
    cublas::dot(handle,
                local_height * local_width,
                local_input.LockedBuffer(), 1,
                ones_d.LockedBuffer(), 1,
                sum_d.Buffer());
  } else if (local_height == 1) {
    ones_d.Resize(local_width, 1);
    El::Fill(ones_d, one);
    cublas::dot(handle,
                local_width,
                local_input.LockedBuffer(), local_input.LDim(),
                ones_d.LockedBuffer(), 1,
                sum_d.Buffer());
  } else {
    GPUMat col_sums_d;
#ifdef HYDROGEN_HAVE_CUB
    col_sums_d.SetMemoryMode(1);  // Use CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
    col_sums_d.Resize(local_width, 1);
    ones_d.Resize(local_height, 1);
    El::Fill(ones_d, one);
    El::Gemv(El::TRANSPOSE, one, local_input, ones_d, zero, col_sums_d);
    if (local_width > local_height) {
      ones_d.Resize(local_width, 1);
      El::Fill(ones_d, one);
    }
    cublas::dot(handle,
                local_width,
                col_sums_d.LockedBuffer(), 1,
                ones_d.LockedBuffer(), 1,
                sum_d.Buffer());
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  // Compute average value across mini-batch
  El::Scale(one / mini_batch_size, sum_d);
  comm.allreduce(static_cast<AbsMat&>(sum_d), input.DistComm());
  CHECK_CUDA(cudaMemcpyAsync(&value,
                             sum_d.LockedBuffer(),
                             sizeof(DataType),
                             cudaMemcpyDeviceToHost,
                             stream));
  copy_event.record(stream);

}
#endif // LBANN_HAS_GPU

} // namespace

EvalType abstract_evaluation_layer::get_value(bool scaled) {
  switch (get_device_allocation()) {
  case El::Device::CPU: get_comm()->wait(m_allreduce_req); break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU: m_copy_event.synchronize(); break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
  if (scaled) { return m_scale * m_value(0, 0); }
  else        { return m_value(0, 0); }
}

abstract_evaluation_layer::abstract_evaluation_layer(lbann_comm *comm)
  : transform_layer(comm) {
  this->m_expected_num_child_layers = 0;
}

void abstract_evaluation_layer::setup_dims() {
  transform_layer::setup_dims();
  if (get_input_size() != 1) {
    std::stringstream err;
    const auto& dims = get_input_dims();
    err << get_type() << " layer \"" << get_name() << "\" "
        << "expects a scalar input, but "
        << "parent layer \"" << m_parent_layers[0]->get_name() << "\" "
        << "has dimensions of ";
    for (size_t i = 0; i < dims.size(); ++i) {
      err << (i > 0 ? " x " : "") << dims[i];
    }
    LBANN_ERROR(err.str());
  }
}

void abstract_evaluation_layer::setup_data() {
  transform_layer::setup_data();
#ifdef LBANN_HAS_GPU
  m_value.SetMemoryMode(1); // Use pinned memory on host
#endif // LBANN_HAS_GPU
  El::Zeros(m_value, 1, 1);
}

void abstract_evaluation_layer::fp_compute() {
  switch (get_device_allocation()) {
  case El::Device::CPU:
    fp_cpu(*get_comm(), get_prev_activations(), m_value(0, 0),
           m_allreduce_req);
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    fp_gpu(*get_comm(), get_prev_activations(), m_value(0, 0),
           m_copy_event);
    break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
}

void abstract_evaluation_layer::bp_compute() {
  El::Fill(get_error_signals(), DataType(m_scale));
}

abstract_evaluation_layer*
abstract_evaluation_layer::construct(lbann_comm *comm,
                                     data_layout layout,
                                     El::Device device) {
#define EVAL_LAYER_CONSTRUCT(T_layout, T_device)                \
  do {                                                          \
    if (layout == T_layout && device == T_device) {             \
      return new evaluation_layer<T_layout, T_device>(comm);    \
    }                                                           \
  } while (false)
  EVAL_LAYER_CONSTRUCT(data_layout::DATA_PARALLEL, El::Device::CPU);
  EVAL_LAYER_CONSTRUCT(data_layout::MODEL_PARALLEL, El::Device::CPU);
#ifdef LBANN_HAS_GPU
  EVAL_LAYER_CONSTRUCT(data_layout::DATA_PARALLEL, El::Device::GPU);
  EVAL_LAYER_CONSTRUCT(data_layout::MODEL_PARALLEL, El::Device::GPU);
#endif // LBANN_HAS_GPU
#undef EVAL_LAYER_CONSTRUCT

  // Could not construct evaluation layer
  std::stringstream err;
  err << "attempted to construct evaluation layer "
      << "with invalid parameters "
      << "(data layout type " << static_cast<int>(layout) << ", "
      << "device type " << static_cast<int>(device) << ")";
  LBANN_ERROR(err.str());
  return nullptr;

}

} // namespace lbann
