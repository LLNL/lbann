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

#ifndef LBANN_OBJECTIVE_FUNCTION_KL_DIVERGENCE_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_KL_DIVERGENCE_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"
#include "lbann/layers/io/target/target_layer.hpp"

namespace lbann {
/** Kullback Leibler (KL) divergence regularizer
 * It compute latent loss between the distribution of latent space
 * induced by an encoder on input data and some prior.
 * It is acts as a kind of regularizer added to reconstruction cost.
 * A good use is in Variational Autoencoder (VAE).
 * kl_loss = -0.5 * mean(1 + z_log_sigma - square(z_mean) - exp(z_log_sigma))
 * See the following reference:
 * Kingma and Welling, "Auto-Encoding Variational Bayes"
 * Doersch, "Tutorial on Variational Autoencoders"
 * https://blogs.keras.io/building-autoencoder-in-keras.html
 * TODO: Add checkpoint */

class kl_divergence : public objective_function_term {
 public:
  /** Default constructor. */
  kl_divergence(std::string layer1, std::string layer2);
  /** Copy constructor. */
  kl_divergence(const kl_divergence& other);
  /** Copy assignment operator. */
  kl_divergence& operator=(const kl_divergence& other);
  /** Destructor. */
  ~kl_divergence() override;
  
  /** Get the name of the objective function term. */
  std::string name() const override { return "kl_divergence"; }

  /** Setup KL divergence regularization term. */
  void setup(model& m) override;

  /** Get the value of the KL divergence regularization term. */
  EvalType evaluate() override;

  /** Regularization terms are applied to the objective function.
  *Add kl_loss to gradient/error signal.*/
  void differentiate() override;
  
  void compute_weight_regularization() override {};

 protected:
 /** input layer parameters to the KL-divergence function. 
 * z_mean is mean of Gaussian distribution in latent space
 * z_log_sigma is (log) variance of distribution in latent space
 * string and pointer variables are layer name and layer pointer respectively*/
 std::string m_z_mean_layer_name;
 std::string m_z_log_sigma_layer_name;
 Layer* m_z_mean_layer;
 Layer* m_z_log_sigma_layer; 
 EvalType m_kl_loss;
 /**target layer for adding kl_loss to gradient. */
 target_layer* m_target_layer;
 /** Gradient matrix. */
 AbsDistMat* m_gradient;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_WEIGHT_REGULARIZATION_KL_DIVERGENCE_HPP_INCLUDED
