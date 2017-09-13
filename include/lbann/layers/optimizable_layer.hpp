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

/**
 * ABC for layers that have optimizers and want to allow external access to
 * them.
 * This only defines an external interface; layers are free to manage their
 * optimizers internally however they wish.
 */
class optimizable_layer {
 public:
  virtual ~optimizable_layer() {}
  /// Return this layer's optimizer.
  virtual optimizer* get_optimizer() const = 0;
};

}  // namespace lbann

#endif  // LBANN_OPTIMIZABLE_LAYER_HPP_INCLUDED
