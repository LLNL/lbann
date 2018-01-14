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

kl_divergence::kl_divergence(std::string layer1, std::string layer2)
               : objective_function_term(),
                 m_z_mean_layer_name(layer1), 
                 m_z_log_sigma_layer_name(layer2), 
                 m_kl_loss(EvalType(0)),
                 m_target_layer(nullptr),
                 m_gradient(nullptr) { } 

kl_divergence::kl_divergence(const kl_divergence& other)
  : objective_function_term(other),
    m_gradient(other.m_gradient) {
  if (m_gradient != nullptr) { m_gradient = m_gradient->Copy(); }
}

kl_divergence& kl_divergence::operator=(const kl_divergence& other) {
  objective_function_term::operator=(other);
  if (m_gradient != nullptr && other.m_gradient != nullptr
      && m_gradient->DistData() == other.m_gradient->DistData()) {
    El::Copy(*other.m_gradient, *m_gradient);
  }
  else {
    if (m_gradient != nullptr) { delete m_gradient; }
    m_gradient = other.m_gradient;
    if (m_gradient != nullptr) { m_gradient = m_gradient->Copy(); }
  }
  return *this;
}

kl_divergence::~kl_divergence() {
  if (m_gradient != nullptr) { delete m_gradient; }
}

void kl_divergence::setup(model& m) {
  objective_function_term::setup(m);
  //set up layers of interest
  for(const auto& l : m.get_layers()) {
    if(dynamic_cast<target_layer*>(l) != nullptr) m_target_layer = (target_layer*)l;
    if(l->get_name() == m_z_mean_layer_name) m_z_mean_layer = l; 
    if(l->get_name() == m_z_log_sigma_layer_name) m_z_log_sigma_layer = l;
  }

  if(m_target_layer != nullptr) {
    const AbsDistMat& ground_truth = m_target_layer->get_activations();
    m_gradient = ground_truth.Construct(ground_truth.Grid(),
                                        ground_truth.Root());
    El::Zeros(*m_gradient, ground_truth.Height(), ground_truth.Width());
  } else {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "Null pointer to target layer";
    throw lbann_exception(err.str());
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
  m_kl_loss = EvalType(0);
  
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
  const AbsDistMat& prediction = m_target_layer->get_prediction();
  El::Zeros(*m_gradient, prediction.Height(), prediction.Width());
  El::Fill(*m_gradient,DataType(m_kl_loss));
  m_target_layer->add_to_error_signal(*m_gradient);
}

}  // namespace lbann
