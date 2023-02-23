////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
// callback .hpp - Base class for LBANN callbacks
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_IMPL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_IMPL_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"

namespace lbann {

  /** @brief Build a standard directory hierarchy including trainer ID.
   */
  inline std::string callback_base::get_multi_trainer_path(const model& m,
                                            const std::string& root_dir) {
    std::string dir = root_dir;
    if (dir.empty()) { dir = "./"; }
    if (dir.back() != '/') { dir += "/"; }

    return build_string(dir,
                        get_const_trainer().get_name(), '/');
  }

  /** @brief Build a standard directory hierachy including trainer,
   * execution context, and model information (in that order).
   */
  inline std::string callback_base::get_multi_trainer_ec_model_path(const model& m,
                                                     const std::string& root_dir) {
    std::string dir = get_multi_trainer_path(m, root_dir);
    const auto& c = static_cast<const SGDExecutionContext&>(m.get_execution_context());
    return build_string(dir,
                        c.get_state_string(), '/',
                        m.get_name(), '/');
  }

  /** @brief Build a standard directory hierachy including trainer,
   * model information in that order.
   */
  inline std::string callback_base::get_multi_trainer_model_path(const model& m,
                                                  const std::string& root_dir) {
    std::string dir = get_multi_trainer_path(m, root_dir);
    return build_string(dir,
                        m.get_name(), '/');
  }

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_IMPL_HPP_INCLUDED