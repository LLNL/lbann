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
// siamese .hpp .cpp - Siamese neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_SIAMESE_HPP
#define LBANN_MODEL_SIAMESE_HPP

#include "lbann/models/model_dag.hpp"

namespace lbann {

class siamese_model : public dag_model {
 public:

  /// Constructor
  siamese_model(lbann_comm *comm,
                int mini_batch_size,
                objective_function *obj_fn,
                optimizer* default_optimizer,
                int num_heads);

  /** Copy constructor. */
  siamese_model(const siamese_model& other) = default;
  /** Copy assignment operator. */
  siamese_model& operator=(const siamese_model& other) = default;
  /** Destructor. */
  ~siamese_model() override = default;

  /** Create copy. */
  siamese_model* copy() const override { return new siamese_model(*this); }

  /** Setup model. */
  void setup() override;

  /** Get model name. */
  std::string name() const override { return "siamese_model"; }

 protected:

  /** The number of heads in Siamese model. */
  int m_num_heads;

};

}  // namespace lbann

#endif  // LBANN_MODEL_SIAMESE_HPP
