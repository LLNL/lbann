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
// model_dag .hpp .cpp - Directed acyclic graph neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_DAG_HPP
#define LBANN_MODEL_DAG_HPP

#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** Directed acyclic graph neural network model. */
class dag_model : public model {
 public:

  /** Constructor. */
  dag_model(int max_mini_batch_size,
            lbann_comm *comm,
            objective_functions::objective_function *obj_fn,
            optimizer_factory *optimizer_fac);

  /** Copy constructor. */
  dag_model(const dag_model& other) = default;

  /** Copy assignment operator. */
  dag_model& operator=(const dag_model& other) = default;

  /** Destructor. */
  virtual ~dag_model() = default;

  /** Create copy. */
  virtual dag_model* copy() const override { return new dag_model(*this); }

  /** Setup model. */
  virtual void setup() override;

  /** Get model name. */
  virtual std::string name() const override { return "dag_model"; }

 protected:

  /** Apply topological sort to list of layers.
   *  A topologically sorted ordering allows us to traverse a directed
   *  acyclic graph without violating dependencies.
   */
  void topologically_sort_layers();

};

}  // namespace lbann

#endif  // LBANN_MODEL_DAG_HPP
