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
#ifndef LBANN_WEIGHTS_WEIGHTS_HELPERS_HPP_INCLUDED
#define LBANN_WEIGHTS_WEIGHTS_HELPERS_HPP_INCLUDED

#include "lbann/utils/exception.hpp"
#include "lbann/utils/typename.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/weights/weights.hpp"

/** @file
 *
 *  A hacky utility for dealing with layers that require access to
 *  mutable weights values (e.g. batchnorm layers). A future refactor
 *  should focus on cleaning this up; a suitable metric for at least
 *  some success would be the deletion of this file.
 *
 *  This file mostly exists because the code is common across several
 *  layers, but it is not needed in the header.
 */

namespace lbann {
namespace weights_details {

/** @class SafeWeightsAccessor
 *  @brief Ensure safe access to weights objects' data.
 */
template <typename TensorDataType>
struct SafeWeightsAccessor
{
  using ValuesType = El::AbstractDistMatrix<TensorDataType>;
  using DataTypeWeights = data_type_weights<TensorDataType>;

  static ValuesType& mutable_values(weights& w)
  {
    auto* dtw = dynamic_cast<DataTypeWeights*>(&w);
    if (!dtw)
      LBANN_ERROR("Weights object named \"",
                  w.get_name(),
                  "\" does not have weights of dynamic type \"",
                  TypeName<TensorDataType>(),
                  "\".");
    return dtw->get_values_sharded();
  }
}; // class SafeWeightsAccessor

} // namespace weights_details
} // namespace lbann
#endif // LBANN_WEIGHTS_WEIGHTS_HELPERS_HPP_INCLUDED
