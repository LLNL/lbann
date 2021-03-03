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

#ifndef LBANN_BATCH_INFERENCE_ALGORITHM_HPP
#define LBANN_BATCH_INFERENCE_ALGORITHM_HPP

#include "lbann/execution_algorithms/execution_algorithm.hpp"
#include "lbann/models/model.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"

namespace lbann {

/** @brief Class for LBANN batch inference algorithms. */
class batch_inference_algorithm : public execution_algorithm {
public:

  /** Constructor. */
  batch_inference_algorithm() {};
  /** Copy constructor. */
  batch_inference_algorithm(const batch_inference_algorithm& other) = default;
  /** Copy assignment operator. */
  batch_inference_algorithm& operator=(const batch_inference_algorithm& other) = default;
  /** Move constructor. */
  batch_inference_algorithm(batch_inference_algorithm&& other) = default;
  /** Move assignment operator. */
  batch_inference_algorithm& operator=(batch_inference_algorithm&& other) = default;
  /** Destructor. */
  virtual ~batch_inference_algorithm() = default;
  /** Copy training_algorithm. */
  //  virtual batch_inference_algorithm* copy() const = default;

  std::string get_name() const override { return "batch_inference"; }

  // ===========================================
  // Execution
  // ===========================================

  /** Infer on samples from a data coordinator with a given model. */
  void infer(model& model,
             data_coordinator& dc,
             size_t num_batches=0);

  template <typename TensorDataType>
  void infer(model& model,
             El::AbstractDistMatrix<TensorDataType> samples);


protected:
  /** Evaluate model on one step / mini-batch of an SGD forward pass */
  bool infer_mini_batch(model& model, data_coordinator& dc);

  template <typename TensorDataType>
  bool infer_mini_batch(model& model,
                                El::AbstractDistMatrix<TensorDataType> samples);

};

}  // namespace lbann

#endif  // LBANN_BATCH_INFERENCE_ALGORITHM_HPP
