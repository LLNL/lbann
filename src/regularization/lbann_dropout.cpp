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
#include "lbann/utils/lbann_random.hpp"

using namespace El;

namespace lbann {

dropout::dropout(lbann_comm* comm, float keep_prob) :
  comm(comm), m_keep_prob(keep_prob)
#ifdef LBANN_PROCDET_DROPOUT
  , m_cur_mask(comm->get_model_grid())
#endif
{}

void dropout::fp_activations() {
  // Terminate early if dropout is disabled
  if (m_layer->m_execution_mode != execution_mode::training
      || m_keep_prob < 0.0f) return;

  // Get local activations
  ElMat* acts = m_layer->m_activations;
  const Int local_height = acts->LocalHeight();
  const Int local_width = acts->LocalWidth();
  const Int global_height = acts->Height();

#ifdef LBANN_PROCDET_DROPOUT
  bernoulli_fill_procdet(m_cur_mask, acts->Height(), acts->Width(),
                         m_keep_prob);
  m_cur_mask *= 1.0 / m_keep_prob;
  if (acts->GlobalRow(local_height - 1) == global_height - 1) {
    for (int col = 0; col < local_width; ++col) {
      m_cur_mask.SetLocal(local_height - 1, col, 1.0f);
    }
  }
  Hadamard(*acts, m_cur_mask, *acts);
#else
  Mat local_acts = acts->Matrix();

  // Construct dropout mask
  // Note: Construct Bernoulli matrix and scale by
  //   1/m_keep_prob. Entries corresponding to bias row are set to 1
  //   to ensure that mask doesn't affect bias row. This
  //   implementation assumes 'acts' is in MC,MR; Star,VC; Star,VR; or
  //   similar format.
  Bernoulli(m_cur_mask, local_height, local_width, m_keep_prob);
  m_cur_mask *= 1.0 / m_keep_prob;
  if (acts->GlobalRow(local_height - 1) == global_height - 1) {
    for (int col = 0; col < local_width; ++col) {
      m_cur_mask.Set(local_height - 1, col, 1.0f);
    }
  }
  // Apply dropout mask to local activations
  Hadamard(local_acts, m_cur_mask, local_acts);
#endif  // LBANN_PROCDET_DROPOUT
}

void dropout::bp_activations() {
  // Terminate early if dropout is disabled
  if (m_layer->m_execution_mode != execution_mode::training
      || m_keep_prob < 0.0f) return;

#ifdef LBANN_PROCDET_DROPOUT
  Hadamard(*(m_layer->m_prev_error_signal), m_cur_mask, *(m_layer->m_prev_error_signal));
#else
  // Re-weight the incoming loss using dropout mask
  Mat local_prev_error_signal = m_layer->m_prev_error_signal->Matrix();
  Hadamard(local_prev_error_signal, m_cur_mask, local_prev_error_signal);
#endif  // LBANN_PROCDET_DROPOUT
}

}  // namespace lbann
