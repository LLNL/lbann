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

#ifndef LBANN_LAYER_MATH_BINARY_HPP_INCLUDED
#define LBANN_LAYER_MATH_BINARY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** Base class for binary math layers.
 *  'Name' should be a type such that Name() returns a human-readable
 *  layer name, e.g. an empty struct that can be converted to a
 *  string.
 */
template <data_layout Layout, El::Device Device, typename Name>
class binary_math_layer : public Layer {
public:
  binary_math_layer(lbann_comm *comm) : Layer(comm) {
    m_expected_num_parent_layers = 2;
  }
  binary_math_layer* copy() const override {
    return new binary_math_layer<Layout,Device,Name>(*this);
  }
  std::string get_type() const override { return Name(); }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  
protected:
  
  void setup_dims() override {
    set_output_dims(get_input_dims());
    Layer::setup_dims();

    // Check that input dimensions match
    if (get_input_dims(0) != get_input_dims(1)) {
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has input tensors with different dimensions (";
      for (int i = 0; i < get_num_parents(); ++i) {
        err << (i > 0 ? ", " : "")
            << "layer \"" << m_parent_layers[i]->get_name() << "\" "
            << "outputs ";
        const auto& dims = get_input_dims(i);
        for (size_t j = 0; j < dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << dims[j];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
    
  }
  
  void fp_compute() override;
  void bp_compute() override;
  
};

// Convenience macro to define a binary math layer class
#define LBANN_DEFINE_BINARY_MATH_LAYER(layer_name, layer_string) \
  struct layer_name##_name_struct {                             \
    inline operator std::string() { return layer_string; }      \
  };                                                            \
  template <data_layout Layout, El::Device Device>              \
  using layer_name                                              \
  = binary_math_layer<Layout, Device, layer_name##_name_struct>;

// Arithmetic operations
LBANN_DEFINE_BINARY_MATH_LAYER(add_layer,         "add");
LBANN_DEFINE_BINARY_MATH_LAYER(subtract_layer,    "subtract");
LBANN_DEFINE_BINARY_MATH_LAYER(multiply_layer,    "multiply");
LBANN_DEFINE_BINARY_MATH_LAYER(divide_layer,      "divide");
LBANN_DEFINE_BINARY_MATH_LAYER(mod_layer,         "modulo");
LBANN_DEFINE_BINARY_MATH_LAYER(pow_layer,         "power");
LBANN_DEFINE_BINARY_MATH_LAYER(safe_divide_layer, "safe divide");

// Comparison operations
LBANN_DEFINE_BINARY_MATH_LAYER(max_layer,           "maximum");
LBANN_DEFINE_BINARY_MATH_LAYER(min_layer,           "minimum");
LBANN_DEFINE_BINARY_MATH_LAYER(equal_layer,         "equal");
LBANN_DEFINE_BINARY_MATH_LAYER(not_equal_layer,     "not equal");
LBANN_DEFINE_BINARY_MATH_LAYER(less_layer,          "less than");
LBANN_DEFINE_BINARY_MATH_LAYER(less_equal_layer,    "less than or equal");
LBANN_DEFINE_BINARY_MATH_LAYER(greater_layer,       "greater than");
LBANN_DEFINE_BINARY_MATH_LAYER(greater_equal_layer, "greater than or equal");
  
// Logical operations
LBANN_DEFINE_BINARY_MATH_LAYER(and_layer, "logical and");
LBANN_DEFINE_BINARY_MATH_LAYER(or_layer,  "logical or");
LBANN_DEFINE_BINARY_MATH_LAYER(xor_layer, "logical xor");

} // namespace lbann

#undef LBANN_DEFINE_BINARY_MATH_LAYER 
#endif // LBANN_LAYER_MATH_BINARY_HPP_INCLUDED
