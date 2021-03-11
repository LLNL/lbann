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

#include "lbann/models/model.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"

namespace lbann {

/** @brief Class for LBANN batch inference algorithms. */
class batch_functional_inference_algorithm {
public:

  /** Constructor. */
  batch_functional_inference_algorithm() {};
  /** Copy constructor. */
  batch_functional_inference_algorithm(const batch_functional_inference_algorithm& other) = default;
  /** Copy assignment operator. */
  batch_functional_inference_algorithm& operator=(const batch_functional_inference_algorithm& other) = default;
  /** Move constructor. */
  batch_functional_inference_algorithm(batch_functional_inference_algorithm&& other) = default;
  /** Move assignment operator. */
  batch_functional_inference_algorithm& operator=(batch_functional_inference_algorithm&& other) = default;
  /** Destructor. */
  virtual ~batch_functional_inference_algorithm() = default;
  /** Copy training_algorithm. */
  //  virtual batch_functional_inference_algorithm* copy() const = default;

  std::string get_name() const { return "batch_functional_inference"; }

  // ===========================================
  // Execution
  // ===========================================

  /** Infer on samples from a data coordinator with a given model. */
  template <typename TensorDataType>
  El::AbstractDistMatrix<TensorDataType>
  infer(model& model,
        El::AbstractDistMatrix<TensorDataType> const& samples,
        std::string output_layer,
        size_t mbs=0);


protected:
  /** Infer on one mini batch with a given model. */
  template <typename TensorDataType>
  El::AbstractDistMatrix<TensorDataType>
  infer_mini_batch(model& model,
                   El::AbstractDistMatrix<TensorDataType> const& samples,
                   std::string output_layer);

};

}  // namespace lbann

#endif  // LBANN_BATCH_INFERENCE_ALGORITHM_HPP
