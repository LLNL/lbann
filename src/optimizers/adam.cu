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
//
// adam .hpp .cpp .cu - SGD with Adam
// Reference:
// Kingma, D. and Ba, J. 2014. Adam: A Method for Stochastic Optimization.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/exception.hpp"

namespace lbann {

namespace {

__global__ void update_kernel(DataType *values, const DataType *gradient,
                              DataType *moment1, DataType *moment2,
                              El::Int height, El::Int width,
                              DataType correction, DataType eps,
                              DataType m_beta1, DataType m_beta2) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= height * width) return;
  DataType g = gradient[offset] + eps;
  DataType &m1 = moment1[offset];
  DataType &m2 = moment2[offset];
  m1 = m_beta1 * m1 + (DataType(1) - m_beta1) * g;
  m2 = m_beta2 * m2 + (DataType(1) - m_beta2) * g * g;
  values[offset] -= correction * m1 / (sqrt(m2) + eps);
}

}

void adam::update_gpu(const std::vector<DataType *> &gradient_d) {
  m_current_beta1 *= m_beta1;
  m_current_beta2 *= m_beta2;
  // Precompute the bias correction and learning rate.
  const DataType correction = m_learning_rate *
                              (std::sqrt(DataType(1) - m_current_beta2)
                               / (DataType(1) - m_current_beta1));

  const int local_height = m_values->LocalHeight();
  const int local_width = m_values->LocalWidth();
      
  int tb_dim = 256;
  int grid_dim = local_height * local_width / tb_dim
      + ((local_height * local_width) % tb_dim ? 1 : 0);
  for (int i = 0; i < m_cudnn->get_num_gpus(); ++i) {
    FORCE_CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    update_kernel<<<grid_dim, tb_dim>>>(m_values_d[i], gradient_d[i],
                                        m_moment1_d[i], m_moment2_d[i],
                                        local_height, local_width,
                                        correction, m_eps,
                                        m_beta1, m_beta2);
  }
}

}  // namespace lbann
