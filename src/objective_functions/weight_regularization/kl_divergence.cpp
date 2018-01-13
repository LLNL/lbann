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

#include "lbann/objective_functions/weight_regularization/kl_divergence.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/statistics.hpp"


namespace lbann {

void kl_divergence::setup(model& m) {
  objective_function_term::setup(m);

  //set up layers of interest
  for(const auto& l : m.get_layers()) {
    if(l->get_name() == m_z_mean_layer_name) m_z_mean_layer = l; 
    if(l->get_name() == m_z_log_sigma_layer_name) m_z_log_sigma_layer = l;
  }

}

EvalType kl_divergence::evaluate() {

  //Get matrices of input layers
  AbsDistMat& z_mean_acts = m_z_mean_layer->get_activations();
  AbsDistMat& z_log_sigma_acts = m_z_log_sigma_layer->get_activations();
  
  AbsDistMat* z_log_sigma_copy = z_log_sigma_acts.Construct(z_log_sigma_acts.Grid(),
                                                     z_log_sigma_acts.Root());

  DataType mean_value = DataType(0); 
  DataType std_value = DataType(0);
  
  auto sq = [&](const DataType& x) {
    return (x*x);
  };
  auto expn  = [&](const DataType& x) {
    return std::exp(x);
  };
  auto addone  = [&](const DataType& x) {
    return (1+x);
  };


  El::Copy(z_log_sigma_acts, *z_log_sigma_copy);

  //Compute entrywise add, square and exponent of the matrices
  El::EntrywiseMap(z_log_sigma_acts,El::MakeFunction(addone));
  El::EntrywiseMap(z_mean_acts,El::MakeFunction(sq));
  El::EntrywiseMap(*z_log_sigma_copy,El::MakeFunction(expn));

  //Entrywise substraction of matrices
  El::Axpy(DataType(-1),z_mean_acts,z_log_sigma_acts);
  El::Axpy(DataType(-1),*z_log_sigma_copy,z_log_sigma_acts);
  
  //Entrywise mean of all the previous computation (stats)
  entrywise_mean_and_stdev(z_log_sigma_acts,mean_value,std_value);
  m_kl_loss = 0.5 * mean_value;
  return m_kl_loss;
}

void kl_divergence::differentiate() {

}
}  // namespace lbann
