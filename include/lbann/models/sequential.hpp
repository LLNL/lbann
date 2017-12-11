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
// lbann_model_sequential .hpp .cpp - Sequential neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_SEQUENTIAL_HPP
#define LBANN_MODEL_SEQUENTIAL_HPP

#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/io/persist.hpp"
#include <vector>
#include <string>

namespace lbann {

class sequential_model : public model {
 public:

  /** Constructor. */
  sequential_model(lbann_comm *comm,
                   int max_mini_batch_size,
                   objective_function *obj_fn,
                   optimizer* default_optimizer = nullptr);
  
  /** Copy constructor. */
  sequential_model(const sequential_model& other) = default;
  /** Copy assignment operator. */
  sequential_model& operator=(const sequential_model& other) = default;
  /** Destructor. */
  ~sequential_model() override = default;
  /** Create copy. */
  sequential_model* copy() const override { return new sequential_model(*this); }

  /** Get model name. */
  std::string name() const override { return "sequential_model"; }

 protected:

  /** Set up topology of layer graph.
   *  Called in setup function. Parent/child relationships are
   *  established between adjacent layers.
   */
  virtual void setup_layer_topology() override;

};

}  // namespace lbann

#endif  // LBANN_MODEL_SEQUENTIAL_HPP
