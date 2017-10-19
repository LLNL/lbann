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
// lbann_model_dnn .hpp .cpp - Deep Neural Networks models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_DNN_HPP
#define LBANN_MODEL_DNN_HPP

#include "lbann/models/model_sequential.hpp"
#include "lbann/layers/layer.hpp"
#include <vector>
#include <string>

namespace lbann {

class deep_neural_network : public sequential_model {
 public:
  /// Constructor
  deep_neural_network(int mini_batch_size,
                      lbann_comm *comm,
                      objective_functions::objective_function *obj_fn,
                      optimizer_factory *_optimizer_fac);
  deep_neural_network(const deep_neural_network&) = default;
  deep_neural_network& operator=(const deep_neural_network&) = default;

  virtual deep_neural_network* copy() const override { return new deep_neural_network(*this); }

  /// Destructor
  ~deep_neural_network();

  virtual std::string name() const override { return "deep neural network"; }

  /// Compute layer summaries
  virtual void summarize_stats(lbann_summary& summarizer) override;
  virtual void summarize_matrices(lbann_summary& summarizer) override;

  /// Train neural network
  virtual void train(int num_epochs) override;
  /// Training step on one mini-batch
  virtual bool train_mini_batch() override;

  /// Evaluate neural network
  virtual void evaluate(execution_mode mode=execution_mode::testing) override;
  /// Evaluation step on one mini-batch
  virtual bool evaluate_mini_batch() override;

};

}  // namespace lbann

#endif  // LBANN_MODEL_DNN_HPP
