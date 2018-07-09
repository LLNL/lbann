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

/** Compute sum of entries in matrix.
 *  Result is recorded in 1x1 matrix 'sum'.
 */
void matrix_sum(const AbsMat& matrix, AbsMat& sum) {
  if (matrix.GetDevice() != sum.GetDevice()) {
    LBANN_ERROR("input matrix and accumulation variable "
                "must be on same device");
  }
  sum.Resize(1, 1);
  switch (matrix.GetDevice()) {
  case El::Device::CPU:
    {
      auto& s = sum(0,0);
      s = DataType(0);
#pragma omp parallel for reduction(+:s) collapse(2)
      for (El::Int col = 0; col < matrix.Height(); ++col) {
        for (El::Int row = 0; row < matrix.Width(); ++row) {
          s += matrix(row, col);
        }
      }
    }
    break;

#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    if (matrix.Height() > 0 && matrix.Width() > 0) {

      // Initialize temporary matrices
      // Note: 'matrix' is copied into a contiguous temporary matrix
      // if it is not contiguous.
      const auto& size = matrix.Height() * matrix.Width();
      GPUMat contig_matrix, ones;
#ifdef HYDROGEN_HAVE_CUB
      contig_matrix.SetMemoryMode(1); // Use CUB GPU memory pool if possible
      ones.SetMemoryMode(1);          // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
      if (matrix.Height() == matrix.LDim() || matrix.Width() == 1) {
        El::LockedView(contig_matrix, matrix);
      } else {
        El::Copy(matrix, contig_matrix);
      }
      ones.Resize(size, 1);
      El::Fill(ones, DataType(1));

      // Compute sum with cuBLAS
      auto&& handle = El::GPUManager::cuBLASHandle();
      CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
      cublas::dot(handle,
                  size,
                  contig_matrix.LockedBuffer(), 1,
                  ones.LockedBuffer(), 1,
                  sum.Buffer());
      CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    }
    break;
#endif // LBANN_HAS_GPU

  default: LBANN_ERROR("invalid device");
  }
}

} // namespace

EvalType abstract_evaluation_layer::get_value(bool scaled) {
  get_comm()->wait(m_allreduce_req);
  const EvalType value = m_value->Get(0,0);
  if (scaled) { return m_scale * value; }
  else        { return value; }
}

abstract_evaluation_layer::abstract_evaluation_layer(lbann_comm *comm)
  : transform_layer(comm), m_scale(0) {

  // Evaluation layer has no children
  m_expected_num_child_layers = 0;

}

abstract_evaluation_layer
::abstract_evaluation_layer(const abstract_evaluation_layer& other)
  : transform_layer(other),
    m_scale(other.m_scale),
    m_value(other.m_value ? other.m_value->Copy() : nullptr),
    m_allreduce_req(other.m_allreduce_req) {}

abstract_evaluation_layer&
abstract_evaluation_layer::operator=(const abstract_evaluation_layer& other) {
  transform_layer::operator=(other);
  m_scale = other.m_scale;
  m_value.reset(other.m_value ? other.m_value->Copy() : nullptr);
  m_allreduce_req = other.m_allreduce_req;
  return *this;
}

void abstract_evaluation_layer::setup_matrices(const El::Grid& grid) {
  transform_layer::setup_matrices(grid);
  switch (get_device_allocation()) {
  case El::Device::CPU:
    m_value.reset(new StarMat<El::Device::CPU>(grid)); break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    m_value.reset(new StarMat<El::Device::GPU>(grid)); break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
  El::Zeros(*m_value, 1, 1);
}

void abstract_evaluation_layer::fp_compute() {
  const auto& input = get_prev_activations();
  const auto& mini_batch_size = input.Width();
  matrix_sum(input.LockedMatrix(), m_value->Matrix());
  *m_value *= DataType(1) / mini_batch_size;
  get_comm()->nb_allreduce(*m_value, input.DistComm(), m_allreduce_req);
}

void abstract_evaluation_layer::bp_compute() {
  El::Fill(get_error_signals(), DataType(m_scale));
}

} // namespace lbann
