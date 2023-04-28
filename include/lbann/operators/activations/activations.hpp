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

#ifndef LBANN_INCLUDE_LBANN_OPERATORS_ACTIVATIONS_ACTIVATIONS_HPP_INCLUDED
#define LBANN_INCLUDE_LBANN_OPERATORS_ACTIVATIONS_ACTIVATIONS_HPP_INCLUDED

#include <lbann/operators/declare_stateless_op.hpp>

namespace lbann {

/** @class lbann::log_sigmoid_layer
 *  @brief Logarithm of sigmoid function.
 *
 *  @f[ \log(\sigma(x)) = -\log(1 + e^{-x}) @f]
 *  See https://en.wikipedia.org/wiki/Sigmoid_function.
 */
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(LogSigmoid, "log sigmoid", true);

/** @class lbann::selu_layer
 *  @brief Scaled exponential rectified linear unit.
 *
 *  @f[
 *    \text{SELU}(x) =
 *      \begin{cases}
 *        s x                & x > 0 \\
 *        s \alpha (e^x - 1) & x \leq 0
 *      \end{cases}
 *  @f]
 *  with @f$\alpha \approx 1.67@f$ and @f$s \approx 1.05@f$. Note that
 *  SELU is equivalent to @f$ s \, \text{ELU}(x;\alpha) @f$. See:
 *
 *  Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp
 *  Hochreiter. "Self-normalizing neural networks." In Advances in
 *  Neural Information Processing Systems, pp. 971-980. 2017.
 */
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Selu, "SELU", true);

/** @class lbann::sigmoid_layer
 *  @brief Special case of logistic function.
 *
 *  @f[ \sigma(x) = \frac{1}{1 + e^{-x}} @f]
 *  See https://en.wikipedia.org/wiki/Sigmoid_function.
 */
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Sigmoid, "sigmoid", true);
// Sigmoid function output is strictly in (0,1)
// Note: Output is in the range [eps,1-eps], where 'eps' is machine
// epsilon. This avoids denormalized floats and helps mitigate some
// numerical issues.
#define LBANN_ENABLE_SIGMOID_CUTOFF

/** @class lbann::softplus_layer
 *  @brief Smooth approximation to ReLU function.
 *
 *  @f[ \text{softplus}(x) = \log (e^x + 1) @f]
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Softplus, "softplus", true);

/** @class lbann::softsign_layer
 *  @brief Smooth approximation to sign function.
 *
 *  @f[ \text{softsign}(x) = \frac{x}{1 + |x|} @f]
 */
LBANN_DECLARE_STATELESS_ELEMENTWISE_OPERATOR(Softsign, "softsign", true);

} // namespace lbann

#undef DEFINE_ENTRYWISE_UNARY_LAYER
#undef UNARY_ETI_DECL_MACRO
#undef UNARY_ETI_DECL_MACRO_DEV

#endif // LBANN_INCLUDE_LBANN_OPERATORS_ACTIVATIONS_ACTIVATIONS_HPP_INCLUDED
