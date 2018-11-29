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

#ifndef LBANN_LAYERS_ACTIVATIONS_ACTIVATIONS_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_ACTIVATIONS_HPP_INCLUDED

#include "lbann/layers/math/unary.hpp"

namespace lbann {

// Convenience macro to define an entry-wise unary layer class
#define DEFINE_ENTRYWISE_UNARY_LAYER(layer_name, layer_string)          \
  struct layer_name##_name_struct {                                     \
    inline operator std::string() { return layer_string; }              \
  };                                                                    \
  template <data_layout Layout, El::Device Device>                      \
  using layer_name                                                      \
  = entrywise_unary_layer<Layout, Device, layer_name##_name_struct>;

/** Log sigmoid layer.
 *  \f[ \log(\sigma(x)) = -\log(1 + e^{-x}) \f]
 *  See https://en.wikipedia.org/wiki/Sigmoid_function
 */
DEFINE_ENTRYWISE_UNARY_LAYER(log_sigmoid_layer, "log sigmoid")

/** Rectified linear unit layer.
 *  \f[ \text{ReLU}(x) = \text{max}(x, 0) \f]
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
DEFINE_ENTRYWISE_UNARY_LAYER(relu_layer, "ReLU")

/** Scaled exponential rectified linear unit layer.
 *  \f[
 *    \text{SELU}(x) =
 *      \begin{cases}
 *        s x                & x > 0
 *        s \alpha (e^x - 1) & x \leq 0
 *      \end{cases}
 *  \f]
 *  \f$\alpha\f$ and \f$s\f$ are pre-chosen constants. Note that this
 *  is equivalent to \f$ s \, \text{ELU}(x) \f$. See:
 *    Klambauer, Gunter, Thomas Unterthiner, Andreas Mayr, and Sepp
 *    Hochreiter. "Self-normalizing neural networks." In Advances in
 *    Neural Information Processing Systems, pp. 971-980. 2017.
 */
DEFINE_ENTRYWISE_UNARY_LAYER(selu_layer, "SELU")

/** Sigmoid layer.
 *  \f[ \sigma(x) = \frac{1}{1 + e^{-x}} \f]
 *  See https://en.wikipedia.org/wiki/Sigmoid_function
 */
DEFINE_ENTRYWISE_UNARY_LAYER(sigmoid_layer, "sigmoid")
// Sigmoid function output is strictly in (0,1)
// Note: Output is in the range [eps,1-eps], where 'eps' is machine
// epsilon. This avoids denormalized floats and helps mitigate some
// numerical issues.
#define LBANN_ENABLE_SIGMOID_CUTOFF

/** Softplus layer.
 *  \f[ \text{softplus}(x) = \log (e^x + 1) \f]
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
DEFINE_ENTRYWISE_UNARY_LAYER(softplus_layer, "softplus")

/** Softsign layer.
 *  \f[ \text{softsign}(x) = \frac{x}[1 + |x|} \f]
 */
DEFINE_ENTRYWISE_UNARY_LAYER(softsign_layer, "softsign")

} // namespace lbann

#undef DEFINE_ENTRYWISE_UNARY_LAYER
#endif // LBANN_LAYERS_ACTIVATIONS_ACTIVATIONS_HPP_INCLUDED
