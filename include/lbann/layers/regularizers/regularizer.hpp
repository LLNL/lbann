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
// lbann_layer .h .cpp - Parent class for all layer types
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include <string>
#include <vector>

namespace lbann {

class regularizer_layer : public Layer {
 public:
  regularizer_layer(int index, 
                    lbann_comm *comm) :
    Layer(index, comm) {
    
  }
  regularizer_layer(const regularizer_layer&) = default;
  regularizer_layer& operator=(const regularizer_layer&) = default;
  virtual ~regularizer_layer() {}

  template<data_layout T_layout> inline void initialize_distributed_matrices() {
    Layer::initialize_distributed_matrices<T_layout>();
  }
};

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_HPP_INCLUDED
