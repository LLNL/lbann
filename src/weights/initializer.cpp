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

#include "lbann/weights/initializer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

void constant_initializer::fill(AbsDistMat& matrix) {
  if (m_value == DataType(0)) {
    El::Zero(matrix);
  } else {
    El::Fill(matrix, m_value);
  }
}

void value_initializer::fill(AbsDistMat& matrix) {

  // Check that number of values matches weights matrix
  if (matrix.Height() * matrix.Width() != (El::Int) m_values.size()) {
    std::stringstream err;
    err << "a value initializer with " << m_values.size() << " values "
        << "attempted to initialize a "
        << matrix.Height() << " x " << matrix.Width() << " "
        << "weights matrix";
    LBANN_ERROR(err.str());
  }

  // Copy values to a CPU matrix
  // Note: If the weights matrix is on CPU, the CPU matrix is a matrix
  // view. Otherwise, the CPU matrix values are copied to the weights
  // matrix.
  CPUMat matrix_cpu;
  if (matrix.GetLocalDevice() == El::Device::CPU) {
    El::View(matrix_cpu, matrix.Matrix());
  } else {
    matrix_cpu.Resize(matrix.LocalHeight(), matrix.LocalWidth());
  }
#pragma omp parallel for collapse(2)
  for (El::Int local_col = 0; local_col < matrix.LocalWidth(); ++ local_col) {
    for (El::Int local_row = 0; local_row < matrix.LocalHeight(); ++ local_row) {
      const auto& global_row = matrix.GlobalRow(local_row);
      const auto& global_col = matrix.GlobalCol(local_col);
      const auto& global_pos = global_row + matrix.Height() * global_col;
      matrix_cpu(local_row, local_col) = m_values[global_pos];
    }
  }
  if (matrix.GetLocalDevice() != El::Device::CPU) {
    El::Copy(matrix_cpu, matrix.Matrix());
#ifdef HYDROGEN_HAVE_CUDA
    El::GPUManager::SynchronizeStream(); /// @todo Use new Hydrogen synchronization semantics when available
#endif // HYDROGEN_HAVE_CUDA
  }
  
}
  
void uniform_initializer::fill(AbsDistMat& matrix) {
  uniform_fill(matrix, matrix.Height(), matrix.Width(), 
               (m_max + m_min) / 2, (m_max - m_min) / 2);
}

void normal_initializer::fill(AbsDistMat& matrix) {
  gaussian_fill(matrix, matrix.Height(), matrix.Width(), 
                m_mean, m_standard_deviation);
}

}  // namespace lbann
