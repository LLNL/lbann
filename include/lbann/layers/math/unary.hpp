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

#ifndef LBANN_LAYERS_MATH_UNARY_HPP_INCLUDED
#define LBANN_LAYERS_MATH_UNARY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** Templated class for entry-wise unary layers.
 *  'Name' should be a type such that Name() returns a human-readable
 *  layer name, e.g. an empty struct that can be converted to a
 *  string.
 */
template <data_layout Layout, El::Device Device, typename Name>
class entrywise_unary_layer : public Layer {
public:
  entrywise_unary_layer(lbann_comm *comm) : Layer(comm) {}
  entrywise_unary_layer* copy() const override {
    return new entrywise_unary_layer<Layout,Device,Name>(*this);
  }
  std::string get_type() const override { return Name(); }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
protected:
  void setup_dims() override {
    set_output_dims(get_input_dims());
    Layer::setup_dims();
  }
  void fp_compute() override;
  void bp_compute() override;
};

// Convenience macro to define an entry-wise unary layer class
#define DEFINE_ENTRYWISE_UNARY_LAYER(layer_name, layer_string)      \
  struct layer_name##_name_struct {                             \
    inline operator std::string() { return layer_string; }      \
  };                                                            \
  template <data_layout Layout, El::Device Device>              \
  using layer_name                                              \
  = entrywise_unary_layer<Layout, Device, layer_name##_name_struct>;

// Logical operations
DEFINE_ENTRYWISE_UNARY_LAYER(logical_not_layer, "logical not");

// Sign operations
DEFINE_ENTRYWISE_UNARY_LAYER(abs_layer,      "absolute value");
DEFINE_ENTRYWISE_UNARY_LAYER(negative_layer, "negative");
DEFINE_ENTRYWISE_UNARY_LAYER(sign_layer,     "sign");

// Rounding operations
DEFINE_ENTRYWISE_UNARY_LAYER(round_layer, "round");
DEFINE_ENTRYWISE_UNARY_LAYER(ceil_layer,  "ceil");
DEFINE_ENTRYWISE_UNARY_LAYER(floor_layer, "floor");

// Power operations
DEFINE_ENTRYWISE_UNARY_LAYER(reciprocal_layer,      "reciprocal");
DEFINE_ENTRYWISE_UNARY_LAYER(square_layer,          "square");
DEFINE_ENTRYWISE_UNARY_LAYER(sqrt_layer,            "square root");
DEFINE_ENTRYWISE_UNARY_LAYER(rsqrt_layer,           "reciprocal square root");
DEFINE_ENTRYWISE_UNARY_LAYER(safe_reciprocal_layer, "safe reciprocal");

// Exponential and logarithmic operations
DEFINE_ENTRYWISE_UNARY_LAYER(exp_layer,   "exponential");
DEFINE_ENTRYWISE_UNARY_LAYER(expm1_layer, "expm1");
DEFINE_ENTRYWISE_UNARY_LAYER(log_layer,   "natural logarithm");
DEFINE_ENTRYWISE_UNARY_LAYER(log1p_layer, "log1p");

// Trigonometric operations
DEFINE_ENTRYWISE_UNARY_LAYER(cos_layer,  "cosine");
DEFINE_ENTRYWISE_UNARY_LAYER(sin_layer,  "sine");
DEFINE_ENTRYWISE_UNARY_LAYER(tan_layer,  "tangent");
DEFINE_ENTRYWISE_UNARY_LAYER(acos_layer, "arccosine");
DEFINE_ENTRYWISE_UNARY_LAYER(asin_layer, "arcsine");
DEFINE_ENTRYWISE_UNARY_LAYER(atan_layer, "arctangent");

// Hyperbolic operations
DEFINE_ENTRYWISE_UNARY_LAYER(cosh_layer,  "hyperbolic cosine");
DEFINE_ENTRYWISE_UNARY_LAYER(sinh_layer,  "hyperbolic sine");
DEFINE_ENTRYWISE_UNARY_LAYER(tanh_layer,  "hyperbolic tangent");
DEFINE_ENTRYWISE_UNARY_LAYER(acosh_layer, "hyperbolic arccosine");
DEFINE_ENTRYWISE_UNARY_LAYER(asinh_layer, "hyperbolic arcsine");
DEFINE_ENTRYWISE_UNARY_LAYER(atanh_layer, "hyperbolic arctangent");

} // namespace lbann

#undef DEFINE_ENTRYWISE_UNARY_LAYER
#endif // LBANN_LAYERS_MATH_UNARY_HPP_INCLUDED
