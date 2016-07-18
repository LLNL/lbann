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

#include "lbann/regularization/lbann_dropout.hpp"

namespace lbann {

dropout::dropout(float keep_prob) : m_keep_prob(keep_prob) {}

void dropout::fp_activations() {
  if (m_keep_prob == -1.0f) return;
  // Exclude the bottom row, which is 1s.
  // TODO: handle case where Acts is in other distribution
  auto acts = ((DistMat&) *(m_layer->Acts))(IR(0, m_layer->Acts->Height() - 1),
                                 IR(0, m_layer->Acts->Width()));
  Mat& local_mat = acts.Matrix();
  Mat local_mat_copy(local_mat);
  // Compute acts = (1/p)r * acts where r's elements are IID Bernoulli and * is
  // element-wise multiplication.
  El::Bernoulli(m_cur_mask, local_mat.Height(), local_mat.Width(), m_keep_prob);
  El::Scale(1.0f / m_keep_prob, m_cur_mask);
  El::Hadamard(m_cur_mask, local_mat_copy, local_mat);
}

void dropout::bp_activations() {
  if (m_keep_prob == -1.0f) return;
  // Exclude the bottom row, which is 1s.
  auto bp_input = ((DistMat&) *(m_layer->bp_input))(IR(0, m_layer->bp_input->Height() - 1),
                                         IR(0, m_layer->bp_input->Width()));
  Mat& local_mat = bp_input.Matrix();
  Mat local_mat_copy(local_mat);
  // Re-weight the incoming loss according to how we adjusted weights during
  // forward propagation.
  El::Hadamard(m_cur_mask, local_mat_copy, local_mat);
}

}  // namespace lbann
