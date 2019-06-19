////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

  // Weight regularization terms
  for (int i=0; i<proto_obj.l2_weight_regularization_size(); ++i) {
    const auto& params = proto_obj.l2_weight_regularization(i);
    obj->add_term(new l2_weight_regularization(params.scale_factor()));
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
