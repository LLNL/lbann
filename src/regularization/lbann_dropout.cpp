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
// lbann_dropout .cpp .hpp - Dropout implementation
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann_base.hpp"
#include "lbann/regularization/lbann_dropout.hpp"

using namespace El;

namespace lbann {

dropout::dropout(float keep_prob) : m_keep_prob(keep_prob) {}

void dropout::fp_activations() {

  // Terminate early if dropout is disabled
  if(m_layer->m_execution_mode != training
     || m_keep_prob < 0.0f) return;

  // Get local activations
  ElMat* acts = m_layer->Acts;
  Mat& local_acts = acts->Matrix();
  const Int global_height = acts->Height();
  const Int local_height = local_acts.Height();
  const Int local_width = local_acts.Width();

  // Construct dropout mask
  // Note: Construct Bernoulli matrix and scale by
  //   1/m_keep_prob. Entries corresponding to bias row are set to 1
  //   to ensure that mask doesn't affect bias row. This
  //   implementation assumes 'acts' is in MC,MR; Star,VC; Star,VR; or
  //   similar format.
  Bernoulli(m_cur_mask, local_height, local_width, m_keep_prob);
  m_cur_mask *= 1.0 / m_keep_prob;
  if(acts->GlobalRow(local_height-1) == global_height-1) {
    for(int j=0; j<local_width; ++j)
      m_cur_mask.Set(local_height-1, j, 1.0);
  }

  // Apply dropout mask to local activations
  Hadamard(local_acts, m_cur_mask, local_acts);

}

void dropout::bp_activations() {

  // Terminate early if dropout is disabled
  if(m_layer->m_execution_mode != training
     || m_keep_prob < 0.0f) return;

  // Re-weight the incoming loss using dropout mask
  Mat& local_Ds = m_layer->Ds->Matrix();
  Hadamard(local_Ds, m_cur_mask, local_Ds);

}

}  // namespace lbann
