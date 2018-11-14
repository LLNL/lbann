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

#ifndef LBANN_LAYER_ACTIVATION_SIGMOID_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_SIGMOID_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/utils/cuda.hpp"

// Output is strictly in (0,1) to avoid numerical issues
#define LBANN_ENABLE_SIGMOID_CUTOFF

namespace lbann {

/** Sigmoid function layer.
 *  \f[ \sigma(x) = \frac{1}{1 + e^{-x}} \f]
 *  See https://en.wikipedia.org/wiki/Sigmoid_function
 */
template <data_layout T_layout, El::Device Dev>
class sigmoid_layer : public Layer {
public:
  sigmoid_layer(lbann_comm *comm) : Layer(comm) {}
  sigmoid_layer* copy() const override { return new sigmoid_layer(*this); }
  std::string get_type() const override { return "sigmoid"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
 protected:
  void fp_compute() override;
  void bp_compute() override;
};
  
} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_SIGMOID_HPP_INCLUDED
