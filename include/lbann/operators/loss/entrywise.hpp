////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_LOSS_ENTRYWISE_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_LOSS_ENTRYWISE_HPP_INCLUDED

#include "lbann/operators/declare_stateless_op.hpp"

namespace lbann {

// Cross entropy loss
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(BinaryCrossEntropy,
                                             "binary cross entropy",
                                             true);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(SigmoidBinaryCrossEntropy,
                                             "sigmoid binary cross entropy",
                                             true);

// Boolean loss functions
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(BooleanAccuracy,
                                             "Boolean accuracy",
                                             false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(BooleanFalseNegative,
                                             "Boolean false negative rate",
                                             false);
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(BooleanFalsePositive,
                                             "Boolean false positive rate",
                                             false);

} // namespace lbann

#endif // LBANN_INCLUDE_LBANN_OPERATORS_LOSS_ENTRYWISE_HPP_INCLUDED
