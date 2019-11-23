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

#ifndef LBANN_OBJECTIVE_FUNCTION_TERM_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_TERM_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/weights/weights.hpp"

namespace lbann {

/** Abstract class for objective function terms. */
class objective_function_term {
 public:

  /** Default constructor. */
  objective_function_term(EvalType scale_factor = EvalType(1));

  /** Copy constructor. */
  objective_function_term(const objective_function_term& other) = default;
  /** Copy assignment operator. */
  objective_function_term& operator=(const objective_function_term& other) = default;
  /** Destructor. */
  virtual ~objective_function_term() = default;
  /** Copy function. */
  virtual objective_function_term* copy() const = 0;

  /** Get the name of the objective function term. */
  virtual std::string name() const = 0;

  /** Setup objective function term. */
  virtual void setup(model& m);

  /** Start evaluation of the objective function term.
   *  This should include the scaling factor. The result is not available until
   *  finish_evaluation has been called.
   */
  virtual void start_evaluation() = 0;

  /** Complete evaluation of the objective function term. */
  virtual EvalType finish_evaluation() = 0;

  /** Compute the gradient of the objective function term.
   *  The gradient is computed w.r.t. the objective function term
   *  inputs. This should include the scaling factor.
   */
  virtual void differentiate() = 0;

  /** Compute the gradient of the weight regularization term.
   *  The gradient is computed w.r.t. the weights.
   */
  virtual void compute_weight_regularization() = 0;

  /** Get list of pointers to layers. */
  std::vector<Layer*> get_layer_pointers() const { return m_layers; }
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<Layer*> layers) { m_layers = layers; }
  /** Get list of pointers to weights. */
  std::vector<weights*> get_weights_pointers() const { return m_weights; }
  /** Set list of pointers to weights. */
  void set_weights_pointers(std::vector<weights*> w) { m_weights = w; }

 protected:

  /** Scaling factor for objective function term. */
  EvalType m_scale_factor;

  /** Layers used to compute objective function term. */
  std::vector<Layer*> m_layers;
  /** Weights used to compute objective function term. */
  std::vector<weights*> m_weights;

  /** Get LBANN communicator. */
  lbann_comm& get_comm() { return *m_comm; }

 private:

  /** LBANN communicator. */
  lbann_comm* m_comm;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_TERM_INCLUDED
