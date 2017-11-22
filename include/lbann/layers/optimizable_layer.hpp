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
//
// optimizable_layer.hpp - Interface for layers that use optimizers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZABLE_LAYER_HPP_INCLUDED
#define LBANN_OPTIMIZABLE_LAYER_HPP_INCLUDED

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

/** Abstract base class for layers with optimizable parameters. */
class optimizable_layer {
 public:
  optimizable_layer() {}
  optimizable_layer(const optimizable_layer&) = default;
  optimizable_layer& operator=(const optimizable_layer&) = default;

  /** Destructor. */
  virtual ~optimizable_layer() = default;

  /** Get optimizer for layer parameters. */
  virtual optimizer* get_optimizer() = 0;

  /** Get layer parameters. */
  virtual AbsDistMat& get_parameters() = 0;

  /** Get objective function gradient.
   *  With respect to layer parameters.
   */
  virtual AbsDistMat& get_parameters_gradient() = 0;

  /** Set layer parameters. */
  virtual void set_parameters(const AbsDistMat& parameters) {
    El::LockedView(get_parameters(), parameters);
  }
  /** Set objective function gradient.
   *  With respect to layer parameters.
   */
  virtual void set_parameters_gradient(const AbsDistMat& gradient) {
    El::Copy(gradient, get_parameters_gradient());
  }
  /** Add to objective function gradient.
   *  With respect to layer parameters.
   */
  virtual void add_to_parameters_gradient(const AbsDistMat& term) {
    El::Axpy(DataType(1), term, get_parameters_gradient());
  }
  /** Set objective function gradient to zero.
   *  With respect to layer parameters.
   */
  virtual void clear_parameters_gradient() {
    El::Zero(get_parameters_gradient());
  }

};

}  // namespace lbann

#endif  // LBANN_OPTIMIZABLE_LAYER_HPP_INCLUDED
