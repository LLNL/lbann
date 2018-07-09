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

namespace lbann {

namespace {

/** Compute sum of entries in local matrix.
 *  Result is recorded in 1x1 matrix 'sum'.
 */
void local_matrix_sum(const AbsMat& local_matrix,
                      El::AbstractMatrix<EvalType>& local_sum) {
  if (local_matrix.GetDevice() != local_sum.GetDevice()) {
    LBANN_ERROR("input matrix and accumulation variable "
                "must be on same device");
  }
  El::Zeros(local_sum, 1, 1);
  switch (local_matrix.GetDevice()) {
  case El::Device::CPU:
    {
      auto& sum = local_sum(0,0);
#pragma omp parallel for reduction(+:sum) collapse(2)
      for (El::Int col = 0; col < local_matrix.Height(); ++col) {
        for (El::Int row = 0; row < local_matrix.Width(); ++row) {
          sum += EvalType(local_matrix(row, col));
        }
      }
    }
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    {
      LBANN_ERROR("not yet implemented");
    }
    break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
}

} // namespace

EvalType abstract_evaluation_layer::get_value(bool scaled) {
  get_comm()->wait(m_allreduce_req);
  const auto& value = m_value->Get(0,0);
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
    m_value.reset(new El::DistMatrix<EvalType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>());
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    m_value.reset(new El::DistMatrix<EvalType, El::STAR, El::STAR, El::ELEMENT, El::Device::GPU>());
    break;
#endif // LBANN_HAS_GPU
  default: LBANN_ERROR("invalid device");
  }
  El::Zeros(*m_value, 1, 1);
}

void abstract_evaluation_layer::fp_compute() {
  const auto& input = get_prev_activations();
  const auto& mini_batch_size = input.Width();
  local_matrix_sum(input.LockedMatrix(), m_value->Matrix());
  *m_value *= EvalType(1) / mini_batch_size;
  get_comm()->nb_allreduce(m_value->Buffer(), 1, input.DistComm(),
                           m_allreduce_req);
}

void abstract_evaluation_layer::bp_compute() {
  El::Fill(get_error_signals(), DataType(m_scale));
}

} // namespace lbann
