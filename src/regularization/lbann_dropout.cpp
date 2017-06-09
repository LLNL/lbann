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

dropout::dropout(data_layout data_dist, lbann_comm* comm, float keep_prob) :
  m_comm(comm), m_keep_prob(keep_prob)
{
  // Setup the data distribution
  switch(data_dist) {
  case data_layout::MODEL_PARALLEL:
    initialize_model_parallel_distribution();
    break;
  case data_layout::DATA_PARALLEL:
    initialize_data_parallel_distribution();
    break;
  default:
    throw lbann_exception(std::string{} + __FILE__ + " " +
                          std::to_string(__LINE__) +
                          "Invalid data layout selected");
  }
}

dropout::~dropout() {
  delete m_cur_mask;
}

/// Matrices should be in MC,MR distributions
void dropout::initialize_model_parallel_distribution() {
#ifdef LBANN_PROCDET_DROPOUT
  m_cur_mask = new DistMat(m_comm->get_model_grid());
#else
  m_cur_mask = new Mat;
#endif
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
void dropout::initialize_data_parallel_distribution() {
#ifdef LBANN_PROCDET_DROPOUT
  m_cur_mask = new StarMat(m_comm->get_model_grid());
#else
  m_cur_mask = new Mat;
#endif
}

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
  bernoulli_fill_procdet(*m_cur_mask, acts->Height(), acts->Width(),
                         m_keep_prob);
  *m_cur_mask *= 1.0 / m_keep_prob;
  if (acts->GlobalRow(local_height - 1) == global_height - 1) {
    for (Int col = 0; col < local_width; ++col) {
      m_cur_mask->SetLocal(local_height - 1, col, 1.0f);
    }
  }
  Hadamard(*acts, *m_cur_mask, *acts);
#else
  Mat local_acts = acts->Matrix();

  // Construct dropout mask
  // Note: Construct Bernoulli matrix and scale by 1/m_keep_prob.
  m_cur_mask->Resize(local_height, local_width);
  EntrywiseMap(*m_cur_mask,
               (std::function<DataType(const DataType&)>)
               ([this](const DataType& z)->DataType {
                 auto& gen = get_fast_generator();
                 std::bernoulli_distribution dist(m_keep_prob);
                 return dist(gen) ? DataType(1) / m_keep_prob : DataType(0);
               }));
  // Apply dropout mask to local activations
  Hadamard(local_acts, *m_cur_mask, local_acts);
#endif  // LBANN_PROCDET_DROPOUT
}

void dropout::bp_activations() {
  // Terminate early if dropout is disabled
  if (m_layer->m_execution_mode != execution_mode::training
      || m_keep_prob < 0.0f) return;

#ifdef LBANN_PROCDET_DROPOUT
  Hadamard(*(m_layer->m_prev_error_signal), *m_cur_mask, *(m_layer->m_prev_error_signal));
#else
  // Re-weight the incoming loss using dropout mask
  Mat local_prev_error_signal = m_layer->m_prev_error_signal->Matrix();
  Hadamard(local_prev_error_signal, *m_cur_mask, local_prev_error_signal);
#endif  // LBANN_PROCDET_DROPOUT
}

}  // namespace lbann
