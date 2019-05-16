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

#ifndef LBANN_OBJECTIVE_FUNCTION_LAYER_TERM_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LAYER_TERM_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"
#include "lbann/layers/transform/evaluation.hpp"

namespace lbann {

class layer_term : public objective_function_term {
public:
  layer_term(EvalType scale_factor = EvalType(1));
  layer_term* copy() const override { return new layer_term(*this); }
  std::string name() const override { return "evaluation layer term"; }

  /** Set corresponding layer. */
  void set_layer(Layer& l);
  /** Get corresponding layer. */
  Layer& get_layer();
  /** Get corresponding layer (const). */
  const Layer& get_layer() const;

  void setup(model& m) override;

  void start_evaluation() override;

  EvalType finish_evaluation() override;

  void differentiate() override;

  void compute_weight_regularization() override {};

private:

  /** Get corresponding evaluation layer. */
  abstract_evaluation_layer& get_evaluation_layer();

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_LAYER_TERM_HPP_INCLUDED
