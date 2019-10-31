////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_CALLBACKS_DATA_TYPE_CALLBACK_HPP_INCLUDED
#define LBANN_CALLBACKS_DATA_TYPE_CALLBACK_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/layers/data_type_layer.hpp"

/** @brief A utility macro for easily adding default-constructed sub-class
 *  builders.*/
#define LBANN_ADD_DEFAULT_DATA_TYPE_CALLBACK_BUILDER(Class, FunctionName)      \
  template <typename TensorDataType>                                           \
  inline std::unique_ptr<callback_base> FunctionName(                          \
    const google::protobuf::Message&, std::shared_ptr<lbann_summary> const&) { \
    return lbann::make_unique<Class<TensorDataType>>();                        \
  }

namespace lbann {

/** @class data_type_callback
 *  @brief Parent class for callbacks that operate on a specific TensorDataType
 */
template <typename TensorDataType>
class data_type_callback : public callback_base {
public:
  data_type_callback(int batch_interval = 1) : callback_base(batch_interval) {}
  data_type_callback(const data_type_callback&) = default;
  data_type_callback& operator=(
    const data_type_callback&) = default;

  ///@}
  /** @name Callback hooks */
  ///@{

  /** @brief Called when weights begins optimization. */
  virtual void on_weight_optimize_begin(model *m, weights<TensorDataType> *w) {}
  /** @brief Called when weights ends optimization. */
  virtual void on_weight_optimize_end(model *m, weights<TensorDataType> *w) {}
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_DATA_TYPE_CALLBACK_HPP_INCLUDED
