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

#include "lbann/objective_functions/kl_divergence.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/statistics.hpp"


namespace lbann {

kl_divergence::kl_divergence(std::string layer1, std::string layer2)
               : objective_function_term(),
                 m_z_mean_layer_name(layer1), 
                 m_z_log_sigma_layer_name(layer2),
                 m_z_mean_layer(nullptr),
                 m_z_log_sigma_layer(nullptr) 
                 { } 

void kl_divergence::setup(model& m) {
  objective_function_term::setup(m);
  //set up layers of interest
  for(const auto& l : m.get_layers()) {
    if(l->get_name() == m_z_mean_layer_name) m_z_mean_layer = l; 
    if(l->get_name() == m_z_log_sigma_layer_name) m_z_log_sigma_layer = l;
  }
  
  if (m_z_mean_layer == nullptr || m_z_log_sigma_layer == nullptr)
  {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "Error: Null pointer(s) to layer(s)";
    throw lbann_exception(err.str());
  }
}

EvalType kl_divergence::evaluate() {
  if (m_scale_factor == EvalType(0)) { return EvalType(0); }
  
  // Matrices
  const auto& global_z_mean = m_z_mean_layer->get_activations();
  const auto& local_z_mean = global_z_mean.LockedMatrix();
  const auto& local_z_log_sigma = m_z_log_sigma_layer->get_local_activations();
  
  // Matrix dimensions
  const int height = global_z_mean.Height();
  const int width = global_z_mean.Width();
  const int local_height = local_z_mean.Height();
  const int local_width = local_z_mean.Width();

  // Compute KL divergence
  EvalType sum = 0;
  #pragma omp parallel for reduction(+:sum) collapse(2)
  for (int col = 0; col < local_width; ++col) {
    for (int row = 0; row < local_height; ++row) {
      const auto z_mean = local_z_mean(row, col);
      const auto z_log_sigma = local_z_log_sigma(row, col);
      sum += (std::exp(z_log_sigma) + z_mean * z_mean
              - z_log_sigma - DataType(1)) / (2 * height);
    }
  }
  const EvalType val = get_comm().allreduce(sum / width,
                                            global_z_mean.DistComm());
  return m_scale_factor * val;

}

void kl_divergence::differentiate() {
  if (m_scale_factor == EvalType(0)) { return; }

  // Matrices
  const auto& global_z_mean = m_z_mean_layer->get_activations();
  const auto& global_z_log_sigma = m_z_log_sigma_layer->get_activations();
  auto&& global_dz_mean = global_z_mean.Copy();
  auto&& global_dz_log_sigma = global_z_log_sigma.Copy();
  const auto& local_z_mean = global_z_mean.LockedMatrix();
  const auto& local_z_log_sigma = global_z_log_sigma.LockedMatrix();
  auto& local_dz_mean = global_dz_mean->Matrix();
  auto& local_dz_log_sigma = global_dz_log_sigma->Matrix();

  // Matrix dimensions
  const int height = global_z_mean.Height();
  const int local_height = local_z_mean.Height();
  const int local_width = local_z_mean.Width();

  // Compute gradient of KL divergence
  #pragma omp parallel for collapse(2)
  for (int col = 0; col < local_width; ++col) {
    for (int row = 0; row < local_height; ++row) {
      const auto z_mean = local_z_mean(row, col);
      const auto z_log_sigma = local_z_log_sigma(row, col);
      local_dz_mean(row, col) = z_mean / height;
      local_dz_log_sigma(row, col) = (std::exp(z_log_sigma) - DataType(1)) / (2 * height);
    }
  }
  
  // Add to error signal
  auto&& z_mean_child_layer = const_cast<Layer*>(m_z_mean_layer->get_child_layers().front());
  auto&& z_log_sigma_child_layer = const_cast<Layer*>(m_z_log_sigma_layer->get_child_layers().front());
  z_mean_child_layer->add_to_error_signal(*global_dz_mean,
                                          DataType(m_scale_factor));
  z_log_sigma_child_layer->add_to_error_signal(*global_dz_log_sigma,
                                               DataType(m_scale_factor));

  // Clean up
  delete global_dz_mean;
  delete global_dz_log_sigma;

}

}  // namespace lbann
