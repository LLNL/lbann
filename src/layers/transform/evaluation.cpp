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
            DataType& value) {
  const auto& local_input = input.LockedMatrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();
  const auto& mini_batch_size = input.Width();
  value = DataType(0);
  int nthreads = omp_get_num_threads();
  std::vector<EvalType> local_value(nthreads, EvalType(0));
  LBANN_OMP_TASKLOOP_COLLAPSE2
  for (El::Int col = 0; col < local_width; ++col) {
    for (El::Int row = 0; row < local_height; ++row) {
      const int tid = omp_get_thread_num();
      local_value[tid] += local_input(row, col);
    }
  }
  for (int i = 0; i < nthreads; ++i) {
    value += local_value[i];
  }
  value = value / mini_batch_size;
  comm.allreduce(&value, 1, input.DistComm());
}

#ifdef LBANN_HAS_GPU
/** GPU implementation of evaluation layer forward prop. */
void fp_gpu(lbann_comm& comm,
            const AbsDistMat& input,
            DataType& value) {

  // Local matrix
  const auto& local_input = input.LockedMatrix();
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();
  const auto& mini_batch_size = input.Width();

  // GPU objects
  GPUMat sum_d, ones_d;
#ifdef HYDROGEN_HAVE_CUB
  sum_d.SetMemoryMode(1);  // Use CUB GPU memory pool if possible
  ones_d.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
  auto&& handle = El::GPUManager::cuBLASHandle();
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

  // Compute sum of local input matrix entries
  if (local_height < 1 || local_width < 1) {
    El::Zeros(sum_d, 1, 1);
  } else if (local_height == local_input.LDim() || local_width == 1) {
    sum_d.Resize(1, 1);
    ones_d.Resize(local_height * local_width, 1);
    El::Fill(ones_d, DataType(1));
    cublas::dot(handle,
                local_height * local_width,
                local_input.LockedBuffer(), 1,
                ones_d.LockedBuffer(), 1,
                sum_d.Buffer());
  } else if (local_height == 1) {
    sum_d.Resize(1, 1);
    ones_d.Resize(local_width, 1);
    El::Fill(ones_d, DataType(1));
    cublas::dot(handle,
                local_width,
                local_input.LockedBuffer(), local_input.LDim(),
                ones_d.LockedBuffer(), 1,
                sum_d.Buffer());
  } else {
    sum_d.Resize(local_width + 1, 1);
    ones_d.Resize(std::max(local_height, local_width), 1);
    El::Fill(ones_d, DataType(1));
    for (El::Int col = 0; col < local_width; ++col) {
      cublas::dot(handle,
                  local_height,
                  local_input.LockedBuffer(0, col), 1,
                  ones_d.LockedBuffer(), 1,
                  sum_d.Buffer(col+1, 0));
    }
    cublas::dot(handle,
                local_width,
                sum_d.LockedBuffer(1, 0), 1,
                ones_d.LockedBuffer(), 1,
                sum_d.Buffer(0, 0));
  }
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  // Compute average value across mini-batch
  CHECK_CUDA(cudaMemcpy(&value, sum_d.LockedBuffer(), sizeof(DataType),
                        cudaMemcpyDeviceToHost));
  value = value / mini_batch_size;
  comm.allreduce(&value, 1, input.DistComm());

}
#endif // LBANN_HAS_GPU

} // namespace

EvalType abstract_evaluation_layer::get_value(bool scaled) {
  if (scaled) { return m_scale * m_value; }
  else        { return m_value; }
}

abstract_evaluation_layer::abstract_evaluation_layer(lbann_comm *comm)
  : transform_layer(comm), m_scale(0), m_value(0) {

  // Evaluation layer has no children
  m_expected_num_child_layers = 0;

}

void abstract_evaluation_layer::fp_compute() {
  switch (get_device_allocation()) {
  case El::Device::CPU:
    fp_cpu(*get_comm(), get_prev_activations(), m_value);
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    fp_gpu(*get_comm(), get_prev_activations(), m_value);
    break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
}

void abstract_evaluation_layer::bp_compute() {
  El::Fill(get_error_signals(), DataType(m_scale));
}

} // namespace lbann
