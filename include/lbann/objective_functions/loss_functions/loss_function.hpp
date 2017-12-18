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

#ifndef LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_HPP_INCLUDED

#include "lbann/objective_functions/objective_function_term.hpp"
#include "lbann/layers/io/target/target_layer.hpp"

namespace lbann {

/** Abstract class for loss functions. */
class loss_function : public objective_function_term {
 public:
  /** Default constructor. */
  loss_function(DataType scale_factor = DataType(1));

  /** Copy constructor. */
  loss_function(const loss_function& other);
  /** Copy assignment operator. */
  loss_function& operator=(const loss_function& other);
  /** Destructor. */
  ~loss_function() override;

  void set_target_layer(target_layer* layer);

  /** Setup objective function term. */
  virtual void setup(model& m) override;
  
  /** Evaluate the objective function term. */
  DataType evaluate() override;

  /** Compute the gradient of the objective function term.
   *  The gradient is computed w.r.t. the objective function term
   *  inputs.
   */
  void differentiate() override;

  /** Evaluate the loss function.
   *  This should not include the scale factor.
   */
  virtual DataType evaluate_compute(const AbsDistMat& prediction,
                                    const AbsDistMat& ground_truth) = 0;

  /** Compute the loss function gradient.
   *  The gradient should be w.r.t. the prediction vector. This should
   *  not include the scale factor.
   */
  virtual void differentiate_compute(const AbsDistMat& prediction,
                                     const AbsDistMat& ground_truth,
                                     AbsDistMat& gradient) = 0;

  bool save_to_checkpoint_shared(lbann::persist& p) override; 
  //  char l_name[512];
  //  sprintf(l_name, "gradient_loss_func_%lldx%lld", m_gradient->Height(), m_gradient->Width());
  //  p.write_distmat(persist_type::model, l_name, (DistMat *)m_gradient);
  //  return true;
  //}
  bool load_from_checkpoint_shared(lbann::persist& p) override;
  //  char l_name[512];
  //  sprintf(l_name, "gradient_loss_func%lldx%lld.bin", m_gradient->Height(), m_gradient->Width());
  //  p.read_distmat(persist_type::model, l_name, (DistMat *)m_gradient);
  //  return true;
  //}

 protected:

  /** Gradient matrix. */
  AbsDistMat* m_gradient;

};

} // namespace lbann

#endif // LBANN_OBJECTIVE_FUNCTION_LOSS_FUNCTION_HPP_INCLUDED
