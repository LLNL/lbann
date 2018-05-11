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

#include "lbann/proto/factories.hpp"

namespace lbann {
namespace proto {

objective_function* construct_objective_function(const lbann_data::ObjectiveFunction& proto_obj) {

  // Instantiate objective function
  objective_function* obj = new objective_function();

  // Loss functions
  for (int i=0; i<proto_obj.mean_squared_error_size(); ++i) {
    const auto& params = proto_obj.mean_squared_error(i);
    obj->add_term(new mean_squared_error_loss(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.mean_absolute_deviation_size(); ++i) {
    const auto& params = proto_obj.mean_absolute_deviation(i);
    obj->add_term(new mean_absolute_deviation_loss(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.mean_absolute_error_size(); ++i) {
    const auto& params = proto_obj.mean_absolute_error(i);
    obj->add_term(new mean_absolute_error_loss(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.cross_entropy_size(); ++i) {
    const auto& params = proto_obj.cross_entropy(i);
    obj->add_term(new cross_entropy(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.binary_cross_entropy_size(); ++i) {
    const auto& params = proto_obj.binary_cross_entropy(i);
    obj->add_term(new binary_cross_entropy(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.cross_entropy_with_uncertainty_size(); ++i) {
    const auto& params = proto_obj.cross_entropy_with_uncertainty(i);
    obj->add_term(new cross_entropy_with_uncertainty(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.geom_negloglike_size(); ++i) {
    const auto& params = proto_obj.geom_negloglike(i);
    obj->add_term(new geom_negloglike(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.poisson_negloglike_size(); ++i) {
    const auto& params = proto_obj.poisson_negloglike(i);
    obj->add_term(new poisson_negloglike(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.polya_negloglike_size(); ++i) {
    const auto& params = proto_obj.polya_negloglike(i);
    obj->add_term(new polya_negloglike(params.scale_factor()));
  }

  // Weight regularization terms
  for (int i=0; i<proto_obj.l1_weight_regularization_size(); ++i) {
    const auto& params = proto_obj.l1_weight_regularization(i);
    obj->add_term(new l1_weight_regularization(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.l2_weight_regularization_size(); ++i) {
    const auto& params = proto_obj.l2_weight_regularization(i);
    obj->add_term(new l2_weight_regularization(params.scale_factor()));
  }
  for (int i=0; i<proto_obj.group_lasso_weight_regularization_size(); ++i) {
    const auto& params = proto_obj.group_lasso_weight_regularization(i);
    obj->add_term(new group_lasso_weight_regularization(params.scale_factor()));
  }

  // Activation regularization terms
  for (int i=0; i<proto_obj.kl_divergence_size(); ++i) {
    const auto& params = proto_obj.kl_divergence(i);
    obj->add_term(new kl_divergence(params.layer1(), params.layer2()));
  }

  // Layer terms
  for (int i=0; i<proto_obj.layer_term_size(); ++i) {
    const auto& params = proto_obj.layer_term(i);
    obj->add_term(new layer_term(params.scale_factor()));
  }

  // Return objective function
  return obj;

}

} // namespace proto
} // namespace lbann
