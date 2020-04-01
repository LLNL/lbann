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

#ifndef LBANN_TRAINING_ALGORITHM_HPP
#define LBANN_TRAINING_ALGORITHM_HPP

#include "lbann/base.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"

namespace lbann {

// Forward-declare this.
class execution_context;

/** Base class for LBANN training_algorithms. */
class training_algorithm {
public:

  /** Constructor. */
  training_algorithm() {};
  /** Copy constructor. */
  training_algorithm(const training_algorithm& other) = default;
  /** Copy assignment operator. */
  training_algorithm& operator=(const training_algorithm& other) = default;
  /** Move constructor. */
  training_algorithm(training_algorithm&& other) = default;
  /** Move assignment operator. */
  training_algorithm& operator=(training_algorithm&& other) = default;
  /** Destructor. */
  virtual ~training_algorithm() = default;
  /** Copy training_algorithm. */
  //  virtual training_algorithm* copy() const = default;

  virtual std::string get_name() const = 0;

  virtual void apply(execution_context& context,
                     model& model,
                     data_coordinator& dc,
                     execution_mode mode,
                     termination_criteria const& term_criteria) = 0;

  void setup_models(std::vector<observer_ptr<model>> models, size_t max_mini_batch_size, TargetModeDimMap& data_dimensions_map);

};

}  // namespace lbann

#endif  // LBANN_TRAINING_ALGORITHM_HPP
