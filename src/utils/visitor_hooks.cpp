////////////////////////////////////////////////////////////////////////////////xecu
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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/visitor_hooks.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

bool is_execution_mode_hook(visitor_hook hook)
{
  switch (hook) {
  case visitor_hook::setup_begin:
  case visitor_hook::setup_end:
  case visitor_hook::phase_end:
  case visitor_hook::epoch_begin:
  case visitor_hook::epoch_end:
  case visitor_hook::optimize_begin:
  case visitor_hook::optimize_end:
  case visitor_hook::invalid:
    return false;
  case visitor_hook::execution_mode_begin:
  case visitor_hook::execution_mode_end:
  case visitor_hook::execution_mode_batch_begin:
  case visitor_hook::execution_mode_batch_end:
  case visitor_hook::execution_mode_forward_prop_begin:
  case visitor_hook::execution_mode_forward_prop_end:
  case visitor_hook::execution_mode_backward_prop_begin:
  case visitor_hook::execution_mode_backward_prop_end:
    return true;
  default:
    LBANN_ERROR("Invalid visitor hook specified");
  }
}

std::string to_string(visitor_hook hook)
{
  switch (hook) {
  case visitor_hook::setup_begin:
    return "setup_begin";
  case visitor_hook::setup_end:
    return "setup_end";
  case visitor_hook::phase_end:
    return "phase_end";
  case visitor_hook::epoch_begin:
    return "epoch_begin";
  case visitor_hook::epoch_end:
    return "epoch_end";
  case visitor_hook::optimize_begin:
    return "optimize_begin";
  case visitor_hook::optimize_end:
    return "optimize_end";
  case visitor_hook::invalid:
    return "invalid";
  case visitor_hook::execution_mode_begin:
  case visitor_hook::execution_mode_end:
  case visitor_hook::execution_mode_batch_begin:
  case visitor_hook::execution_mode_batch_end:
  case visitor_hook::execution_mode_forward_prop_begin:
  case visitor_hook::execution_mode_forward_prop_end:
  case visitor_hook::execution_mode_backward_prop_begin:
  case visitor_hook::execution_mode_backward_prop_end:
    LBANN_ERROR("visitor_hook is execution_mode specific");
  default:
    LBANN_ERROR("Invalid visitor hook specified");
  }
}

std::string to_string(visitor_hook hook, execution_mode mode)
{
  switch (hook) {
  case visitor_hook::execution_mode_begin:
    return to_string(mode) + "_begin";
  case visitor_hook::execution_mode_end:
    return to_string(mode) + "_end";
  case visitor_hook::execution_mode_batch_begin:
    return to_string(mode) + "_batch_begin";
  case visitor_hook::execution_mode_batch_end:
    return to_string(mode) + "_batch_end";
  case visitor_hook::execution_mode_forward_prop_begin:
    return to_string(mode) + "_forward_prop_begin";
  case visitor_hook::execution_mode_forward_prop_end:
    return to_string(mode) + "_forward_prop_end";
  case visitor_hook::execution_mode_backward_prop_begin:
    return to_string(mode) + "_backward_prop_begin";
  case visitor_hook::execution_mode_backward_prop_end:
    return to_string(mode) + "_backward_prop_end";
  case visitor_hook::setup_begin:
  case visitor_hook::setup_end:
  case visitor_hook::epoch_begin:
  case visitor_hook::epoch_end:
  case visitor_hook::phase_end:
  case visitor_hook::optimize_begin:
  case visitor_hook::optimize_end:
  case visitor_hook::invalid:
    LBANN_ERROR("visitor_hook is not execution_mode specific");
  default:
    LBANN_ERROR("Invalid visitor hook specified");
  }
}

void visitor_hook_from_string(std::string const& str,
                              visitor_hook& hook,
                              execution_mode& mode)
{
  mode = execution_mode::invalid;

  if (str == "setup_begin") {
    hook = visitor_hook::setup_begin;
    return;
  }
  else if (str == "setup_end") {
    hook = visitor_hook::setup_end;
    return;
  }
  else if (str == "phase_end") {
    hook = visitor_hook::phase_end;
    return;
  }
  else if (str == "epoch_begin") {
    hook = visitor_hook::epoch_begin;
    return;
  }
  else if (str == "epoch_end") {
    hook = visitor_hook::epoch_end;
    return;
  }
  else if (str == "optimize_begin") {
    hook = visitor_hook::optimize_begin;
    return;
  }
  else if (str == "optimize_end") {
    hook = visitor_hook::optimize_end;
    return;
  }
  else if (str == "invalid") {
    hook = visitor_hook::invalid;
    return;
  }
  else {
    std::string delimiter = "_";
    size_t pos = str.find(delimiter);
    if (pos != std::string::npos) {
      std::string mode_token = str.substr(0, pos);
      mode = exec_mode_from_string(mode_token);
      std::string visitor_token = str.substr(pos, str.length());
      if (visitor_token == "_batch_begin") {
        hook = visitor_hook::execution_mode_batch_begin;
        return;
      }
      else if (visitor_token == "_batch_end") {
        hook = visitor_hook::execution_mode_batch_end;
        return;
      }
      else if (visitor_token == "_forward_prop_begin") {
        hook = visitor_hook::execution_mode_forward_prop_begin;
        return;
      }
      else if (visitor_token == "_forward_prop_end") {
        hook = visitor_hook::execution_mode_forward_prop_end;
        return;
      }
      else if (visitor_token == "_backward_prop_begin") {
        hook = visitor_hook::execution_mode_backward_prop_begin;
        return;
      }
      else if (visitor_token == "_backward_prop_end") {
        hook = visitor_hook::execution_mode_backward_prop_end;
        return;
      }
      else if (visitor_token ==
               "_begin") { // Needs to be last to avoid substrings
        hook = visitor_hook::execution_mode_begin;
        return;
      }
      else if (visitor_token ==
               "_end") { // Needs to be last to avoid substrings
        hook = visitor_hook::execution_mode_end;
        return;
      }
      LBANN_ERROR("\"" + str + "\" is not a valid visitor hook.");
    }
    LBANN_ERROR("\"" + str + "\" is not a valid visitor hook.");
  }
  return;
}

std::istream& operator>>(std::istream& is, visitor_hook& hook)
{
  std::string tmp;
  is >> tmp;
  execution_mode mode;
  visitor_hook_from_string(tmp, hook, mode);
  return is;
}

} // namespace lbann
