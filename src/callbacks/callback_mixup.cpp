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

#include <algorithm>
#include "lbann/callbacks/callback_mixup.hpp"
#include "lbann/utils/beta.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/image.hpp"

namespace lbann {

void callback_mixup::on_forward_prop_end(model *m, Layer *l) {
  if (!m_layers.count(l->get_name())) {
    return;
  }
  if (m->get_execution_mode() != execution_mode::training) {
    return;  // No mixup outside of training.
  }

  auto& samples = l->get_local_activations(0);
  auto& targets = l->get_local_activations(1);
  El::Int mbsize = samples.Width();
  const El::Int samples_height = samples.Height();
  const El::Int targets_height = targets.Height();
  auto& gen = get_fast_generator();
  beta_distribution<float> dist(m_alpha, m_alpha);

  // For now, data must be on the CPU.
  if (samples.GetDevice() != El::Device::CPU ||
      targets.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("mixup only works with CPU data");
  }

  // Decide how to mix the mini-batch.
  std::vector<El::Int> shuffled_indices(mbsize);
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
  std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);

  for (El::Int i = 0; i < mbsize; ++i) {
    const El::Int j = shuffled_indices[i];
    if (i == j) {
      continue;
    }
    float lambda = dist(gen);
    lambda = std::max(lambda, 1.0f - lambda);
    const float lambda_sub = 1.0f - lambda;
    DataType* __restrict__ x1_buf = samples.Buffer() + i*samples.LDim();
    const DataType* __restrict__ x2_buf = samples.LockedBuffer() + j*samples.LDim();
    DataType* __restrict__ y1_buf = targets.Buffer() + i*targets.LDim();
    const DataType* __restrict__ y2_buf = targets.LockedBuffer() + j*targets.LDim();
    for (El::Int k = 0; k < samples_height; ++k) {
      x1_buf[k] = lambda*x1_buf[k] + lambda_sub*x2_buf[k];
    }
    for (El::Int k = 0; k < targets_height; ++k) {
      y1_buf[k] = lambda*y1_buf[k] + lambda_sub*y2_buf[k];
    }
  }
}

}  // namespace lbann
