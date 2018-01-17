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

  //Get matrices of input (latent space) layers
  AbsDistMat& z_mean_acts = m_z_mean_layer->get_activations();
  AbsDistMat& z_log_sigma_acts = m_z_log_sigma_layer->get_activations();
  
  AbsDistMat* z_mean = z_mean_acts.Construct(z_mean_acts.Grid(),
                                             z_mean_acts.Root());
  AbsDistMat* z_log1 = z_log_sigma_acts.Construct(z_log_sigma_acts.Grid(),
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
  
  //Local copies
  El::Copy(z_mean_acts, *z_mean);
  El::Copy(z_log_sigma_acts, *z_log1);
  AbsDistMat* z_log2 = z_log1->Copy();

  //Compute entrywise add, square and exponent of the matrices
  El::EntrywiseMap(*z_log1,El::MakeFunction(addone));
  El::EntrywiseMap(*z_mean,El::MakeFunction(sq));
  El::EntrywiseMap(*z_log2,El::MakeFunction(expn));

  //Entrywise substraction of matrices
  El::Axpy(DataType(-1),*z_mean,*z_log1);
  El::Axpy(DataType(-1),*z_log2,*z_log1);
  
  //Entrywise mean of all the previous computation (stats)
  entrywise_mean_and_stdev(*z_log1,mean_value,std_value);
  return (0.5 * mean_value);
}

void kl_divergence::differentiate() {
  const AbsDistMat& z_mean_acts = m_z_mean_layer->get_activations();
  const AbsDistMat& z_log_sigma_acts = m_z_log_sigma_layer->get_activations();
  auto height= z_mean_acts.Height();
  
  AbsDistMat* z_mean_gradient = z_mean_acts.Construct(z_mean_acts.Grid(),
                                                      z_mean_acts.Root());
  
  AbsDistMat* z_log_sigma_gradient = z_log_sigma_acts.Construct(z_log_sigma_acts.Grid(),
                                                     z_log_sigma_acts.Root());
  El::Copy(z_mean_acts,*z_mean_gradient); 
  El::Copy(z_log_sigma_acts,*z_log_sigma_gradient);

  const DataType scale = DataType(1) / DataType(2)*height;
  
  //Compute z_log_sigma_gradient = -(1/2n)(1-exp(Z_logsigma))
  auto expn  = [&](const DataType& x) {
    return std::exp(x);
  };
  El::EntrywiseMap(*z_log_sigma_gradient,El::MakeFunction(expn));
  El::Axpy(DataType(-1),*z_log_sigma_gradient, *z_log_sigma_gradient);

  //add gradients to respective error signal
  //z_mean_gradient = (1/2n)z_mean 
  m_z_mean_layer->add_to_error_signal(*z_mean_gradient, DataType(scale));
  m_z_log_sigma_layer->add_to_error_signal(*z_log_sigma_gradient,DataType(-scale));
  
  delete z_mean_gradient;
  delete z_log_sigma_gradient;
}

}  // namespace lbann
