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
// lbann_regularizer .hpp - Base class for regularization methods
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_REGULARIZATION_REGULARIZER_HPP_INCLUDED
#define LBANN_REGULARIZATION_REGULARIZER_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/layers/lbann_layer.hpp"

namespace lbann {

/**
 * Base class for different kinds of regularization.
 * Implement some subset of the forward/backward propagation methods, which are
 * automatically called by layers.
 */
class regularizer {
 public:
  virtual ~regularizer() {}
  /**
   * Forward-propagation regularization of incoming connections.
   * This is called before applying a layer's linearity.
   * Example: DropConnect.
   */
  virtual void fp_connections() {}
  /** Corresponding backward-propagation regularization to fp_connections. */
  virtual void bp_connections() {}
  /**
   * Forward propagation regularization of weights.
   * This is called after applying a layer's linearity and before its
   * nonlinearity.
   * Example: L2 normalization.
   */
  virtual void fp_weights() {}
  /** Corresponding backward-propagation regularization to fp_weights. */
  virtual void bp_weights() {}
  /**
   * Forward propagation regularization of activations.
   * This is called after applying a layer's nonlinearity.
   * Example: Dropout.
   */
  virtual void fp_activations() {}
  /** Corresponding backward-propagation regularization to fp_activations. */
  virtual void bp_activations() {}
  /** Regularization of weight gradients. */
  virtual void update_gradients() {}

  /** Update the regularizer after backprop. */
  virtual void update() {}

  /** Set up to regularize layer l. */
  virtual void setup(Layer *l) {
    m_layer = l;
  }
 protected:
  /** Layer being regularized. */
  Layer *m_layer;
};

}  // namespace lbann

#endif  // LBANN_REGULARIZATION_REGULARIZER_HPP_INCLUDED
