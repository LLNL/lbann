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

#ifndef LBANN_LAYERS_MATH_BINARY_HPP_INCLUDED
#define LBANN_LAYERS_MATH_BINARY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

#define LBANN_DECLARE_ENTRYWISE_BINARY_LAYER(LAYER_NAME, LAYER_STRING)      \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  class LAYER_NAME : public data_type_layer<TensorDataType> {               \
  public:                                                                   \
    LAYER_NAME(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {  \
      this->m_expected_num_parent_layers = 2;                               \
    }                                                                       \
    LAYER_NAME* copy() const override {                                     \
      return new LAYER_NAME<TensorDataType,Layout,Device>(*this);           \
    }                                                                       \
    std::string get_type() const override { return LAYER_STRING; }          \
    data_layout get_data_layout() const override { return Layout; }         \
    El::Device get_device_allocation() const override { return Device; }    \
  protected:                                                                \
    void setup_dims(DataReaderMetaData& dr_metadata) override {                                            \
      data_type_layer<TensorDataType>::setup_dims(dr_metadata);                        \
      this->set_output_dims(this->get_input_dims());                        \
      /* Check that input dimensions match */                               \
      if (this->get_input_dims(0) != this->get_input_dims(1)) {             \
        const auto& parents = this->get_parent_layers();                    \
        std::stringstream err;                                              \
        err << this->get_type() << " layer \"" << this->get_name() << "\" " \
            << "has input tensors with different dimensions (";             \
        for (size_t i = 0; i < this->get_num_parents(); ++i) {              \
          const auto& dims = this->get_input_dims(i);                       \
          err << (i > 0 ? ", " : "")                                        \
              << "layer \"" << parents[i]->get_name() << "\" outputs ";     \
          for (size_t j = 0; j < dims.size(); ++j) {                        \
            err << (j > 0 ? " x " : "") << dims[j];                         \
          }                                                                 \
        }                                                                   \
        err << ")";                                                         \
        LBANN_ERROR(err.str());                                             \
      }                                                                     \
    }                                                                       \
    void fp_compute() override;                                             \
    void bp_compute() override;                                             \
  }

// Convenience macros for ETI decls for binary layers

#ifndef LBANN_BINARY_LAYER_INSTANTIATE
#define BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, DEVICE)                   \
  extern template class LAYER_NAME<T, data_layout::DATA_PARALLEL, DEVICE>; \
  extern template class LAYER_NAME<T, data_layout::MODEL_PARALLEL, DEVICE>
#else
#define BINARY_ETI_DECL_MACRO_DEV(...)
#endif // LBANN_BINARY_LAYER_INSTANTIATE

// Instnatiate both data and model parallel layers
#define BINARY_ETI_INST_MACRO_DEV_DT(LAYER_NAME, T, DEVICE)             \
  template class LAYER_NAME<T, data_layout::DATA_PARALLEL, DEVICE>;  \
  template class LAYER_NAME<T, data_layout::MODEL_PARALLEL, DEVICE>

// Instantiate a DEVICE for each allowed tensor data type
#define BINARY_ETI_INST_MACRO_DEV(LAYER_NAME, DEVICE)      \
  BINARY_ETI_INST_MACRO_DEV_DT(LAYER_NAME, float, DEVICE); \
  BINARY_ETI_INST_MACRO_DEV_DT(LAYER_NAME, double, DEVICE)

#ifdef LBANN_HAS_GPU
#define BINARY_ETI_DECL_MACRO(LAYER_NAME, T)                 \
  BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, El::Device::CPU); \
  BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, El::Device::GPU)
#else
#define BINARY_ETI_DECL_MACRO(LAYER_NAME, T)                 \
  BINARY_ETI_DECL_MACRO_DEV(LAYER_NAME, T, El::Device::CPU)
#endif // LBANN_HAS_GPU

// Convenience macro to define an entry-wise binary layer class
#define DEFINE_ENTRYWISE_BINARY_LAYER(layer_name, layer_string)         \
  LBANN_DECLARE_ENTRYWISE_BINARY_LAYER(layer_name, layer_string);       \
  BINARY_ETI_DECL_MACRO(layer_name, float);                             \
  BINARY_ETI_DECL_MACRO(layer_name, double)

// Arithmetic operations
DEFINE_ENTRYWISE_BINARY_LAYER(add_layer,                "add");
DEFINE_ENTRYWISE_BINARY_LAYER(subtract_layer,           "subtract");
DEFINE_ENTRYWISE_BINARY_LAYER(multiply_layer,           "multiply");
DEFINE_ENTRYWISE_BINARY_LAYER(divide_layer,             "divide");
DEFINE_ENTRYWISE_BINARY_LAYER(mod_layer,                "modulo");
DEFINE_ENTRYWISE_BINARY_LAYER(pow_layer,                "power");
DEFINE_ENTRYWISE_BINARY_LAYER(safe_divide_layer,        "safe divide");
DEFINE_ENTRYWISE_BINARY_LAYER(squared_difference_layer, "squared difference");

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
#undef BINARY_ETI_DECL_MACRO
#undef BINARY_ETI_DECL_MACRO_DEV

#endif // LBANN_LAYERS_MATH_BINARY_HPP_INCLUDED
