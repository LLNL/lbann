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

#ifndef LBANN_LAYER_MATH_UNARY_HPP_INCLUDED
#define LBANN_LAYER_MATH_UNARY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

// Convenience macro to define a unary math layer class
#define LBANN_DEFINE_UNARY_MATH_LAYER(layer_name, layer_string)         \
  template <data_layout Layout, El::Device Device>                      \
  class layer_name : public Layer {                                     \
  public:                                                               \
  layer_name(lbann_comm *comm) : Layer(comm) {}                         \
  layer_name* copy() const override { return new layer_name(*this); }   \
  std::string get_type() const override { return layer_string; }        \
  data_layout get_data_layout() const override { return Layout; }       \
  El::Device get_device_allocation() const override { return Device; }  \
  protected:                                                            \
  void fp_compute() override;                                           \
  void bp_compute() override;                                           \
  };

// Logical operations
LBANN_DEFINE_UNARY_MATH_LAYER(not_layer, "logical not");
  
// Sign operations
LBANN_DEFINE_UNARY_MATH_LAYER(abs_layer,      "absolute value");
LBANN_DEFINE_UNARY_MATH_LAYER(negative_layer, "negative");
LBANN_DEFINE_UNARY_MATH_LAYER(sign_layer,     "sign");

// Rounding operations
LBANN_DEFINE_UNARY_MATH_LAYER(round_layer, "round");
LBANN_DEFINE_UNARY_MATH_LAYER(ceil_layer,  "ceil");
LBANN_DEFINE_UNARY_MATH_LAYER(floor_layer, "floor");

// Power operations
LBANN_DEFINE_UNARY_MATH_LAYER(reciprocal_layer, "reciprocal");
LBANN_DEFINE_UNARY_MATH_LAYER(square_layer,     "square");
LBANN_DEFINE_UNARY_MATH_LAYER(sqrt_layer,       "square root");
LBANN_DEFINE_UNARY_MATH_LAYER(rsqrt_layer,      "reciprocal square root");

// Exponential and logarithmic operations
LBANN_DEFINE_UNARY_MATH_LAYER(exp_layer,   "exponential");
LBANN_DEFINE_UNARY_MATH_LAYER(expm1_layer, "expm1");
LBANN_DEFINE_UNARY_MATH_LAYER(log_layer,   "natural logarithm");
LBANN_DEFINE_UNARY_MATH_LAYER(log1p_layer, "log1p");

// Trigonometric operations
LBANN_DEFINE_UNARY_MATH_LAYER(cos_layer,  "cosine");
LBANN_DEFINE_UNARY_MATH_LAYER(sin_layer,  "sine");
LBANN_DEFINE_UNARY_MATH_LAYER(tan_layer,  "tangent");
LBANN_DEFINE_UNARY_MATH_LAYER(acos_layer, "arccosine");
LBANN_DEFINE_UNARY_MATH_LAYER(asin_layer, "arcsine");
LBANN_DEFINE_UNARY_MATH_LAYER(atan_layer, "arctangent");

// Hyperbolic operations
LBANN_DEFINE_UNARY_MATH_LAYER(cosh_layer,  "hyperbolic cosine");
LBANN_DEFINE_UNARY_MATH_LAYER(sinh_layer,  "hyperbolic sine");
LBANN_DEFINE_UNARY_MATH_LAYER(tanh_layer,  "hyperbolic tangent");
LBANN_DEFINE_UNARY_MATH_LAYER(acosh_layer, "hyperbolic arccosine");
LBANN_DEFINE_UNARY_MATH_LAYER(asinh_layer, "hyperbolic arcsine");
LBANN_DEFINE_UNARY_MATH_LAYER(atanh_layer, "hyperbolic arctangent");

} // namespace lbann

#undef LBANN_DEFINE_UNARY_MATH_LAYER 
#endif // LBANN_LAYER_MATH_UNARY_HPP_INCLUDED
