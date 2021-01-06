////////////////////////////////////////////////////////////////////////////////xecu
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

#ifndef LBANN_VISITOR_HOOKS_HPP_INCLUDED
#define LBANN_VISITOR_HOOKS_HPP_INCLUDED

// #include "lbann_config.hpp"

// #include "lbann/Elemental_extensions.hpp"
// #include "lbann/utils/cyg_profile.hpp"
// #include "lbann/utils/file_utils.hpp"
#include "lbann/utils/enum_iterator.hpp"

#include <iostream>
#include <string>

namespace lbann {

/// Neural network execution mode
enum class visitor_hook{setup_begin,
    setup_end,
    train_begin,
    train_end,
    phase_end,
    epoch_begin,
    epoch_end,
    batch_begin,
    batch_end,
    test_begin,
    test_end,
    validation_begin,
    validation_end,
    forward_prop_begin,
    forward_prop_end,
    backward_prop_begin,
    backward_prop_end,
    optimize_begin,
    optimize_end,
    batch_evaluate_begin,
    batch_evaluate_end,
    evaluate_forward_prop_begin,
    evaluate_forward_prop_end,
    invalid};


//enum class execution_mode {training, validation, testing, prediction, invalid};
std::string to_string(visitor_hook hook);
//using visitor_hook_iterator = enum_iterator<visitor_hook, visitor_hook::setup_begin, visitor_hook::invalid>;

/** @brief Convert a string to an execution_mode. */
visitor_hook visitor_hook_from_string(std::string const& str);
/** @brief Extract an execution_mode from a stream. */
std::istream& operator>>(std::istream& os, visitor_hook& e);

} // namespace lbann

#endif // LBANN_VISITOR_HOOKS_HPP_INCLUDED
