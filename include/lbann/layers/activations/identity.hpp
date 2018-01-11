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

#ifndef IDENTITY_HPP_INCLUDED
#define IDENTITY_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

/** Identity activation function. */
template <data_layout T_layout>
class identity_layer : public activation_layer {
 public:
  /** Constructor. */
  identity_layer(lbann_comm *comm) : activation_layer(comm) {}
  /** Copy function. */
  identity_layer* copy() const override { return new identity_layer(*this); }

  std::string get_type() const override { return "identity"; }
  data_layout get_data_layout() const override { return T_layout; }

  void fp_compute() override {
    El::LockedView(get_activations(), get_prev_activations());
  }

  void bp_compute() override {
    get_error_signals() += get_prev_error_signals();
  }

};

}  // namespace lbann

#endif  // IDENTITY_HPP_INCLUDED
