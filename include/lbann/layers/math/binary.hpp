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

#ifndef LBANN_LAYERS_MATH_BINARY_HPP_INCLUDED
#define LBANN_LAYERS_MATH_BINARY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** Templated class for entry-wise binary layers.
 *  'Name' should be a type such that Name() returns a human-readable
 *  layer name, e.g. an empty struct that can be converted to a
 *  string.
 */
template <data_layout Layout, El::Device Device, typename Name>
class entrywise_binary_layer : public Layer {
public:
  entrywise_binary_layer(lbann_comm *comm) : Layer(comm) {
    m_expected_num_parent_layers = 2;
  }
  entrywise_binary_layer* copy() const override {
    return new entrywise_binary_layer<Layout,Device,Name>(*this);
  }
  std::string get_type() const override { return Name(); }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims(get_input_dims());

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

// Convenience macro to define an entry-wise binary layer class
#define DEFINE_ENTRYWISE_BINARY_LAYER(layer_name, layer_string)         \
  struct layer_name##_name_struct {                                     \
    inline operator std::string() { return layer_string; }              \
  };                                                                    \
  template <data_layout Layout, El::Device Device>                      \
  using layer_name                                                      \
  = entrywise_binary_layer<Layout, Device, layer_name##_name_struct>;

// Arithmetic operations
DEFINE_ENTRYWISE_BINARY_LAYER(add_layer,         "add");
DEFINE_ENTRYWISE_BINARY_LAYER(subtract_layer,    "subtract");
DEFINE_ENTRYWISE_BINARY_LAYER(multiply_layer,    "multiply");
DEFINE_ENTRYWISE_BINARY_LAYER(divide_layer,      "divide");
DEFINE_ENTRYWISE_BINARY_LAYER(mod_layer,         "modulo");
DEFINE_ENTRYWISE_BINARY_LAYER(pow_layer,         "power");
DEFINE_ENTRYWISE_BINARY_LAYER(safe_divide_layer, "safe divide");

// Comparison operations
DEFINE_ENTRYWISE_BINARY_LAYER(max_layer,           "maximum");
DEFINE_ENTRYWISE_BINARY_LAYER(min_layer,           "minimum");
DEFINE_ENTRYWISE_BINARY_LAYER(equal_layer,         "equal");
DEFINE_ENTRYWISE_BINARY_LAYER(not_equal_layer,     "not equal");
DEFINE_ENTRYWISE_BINARY_LAYER(less_layer,          "less than");
DEFINE_ENTRYWISE_BINARY_LAYER(less_equal_layer,    "less than or equal");
DEFINE_ENTRYWISE_BINARY_LAYER(greater_layer,       "greater than");
DEFINE_ENTRYWISE_BINARY_LAYER(greater_equal_layer, "greater than or equal");

// Logical operations
DEFINE_ENTRYWISE_BINARY_LAYER(logical_and_layer, "logical and");
DEFINE_ENTRYWISE_BINARY_LAYER(logical_or_layer,  "logical or");
DEFINE_ENTRYWISE_BINARY_LAYER(logical_xor_layer, "logical xor");

} // namespace lbann

#undef DEFINE_ENTRYWISE_BINARY_LAYER
#endif // LBANN_LAYERS_MATH_BINARY_HPP_INCLUDED
