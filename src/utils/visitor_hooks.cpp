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

#include "lbann/utils/visitor_hooks.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

std::string to_string(visitor_hook hook) {
  switch(hook) {
  case visitor_hook::setup_begin:
    return "setup_begin";
  case visitor_hook::setup_end:
    return "setup_end";
  case visitor_hook::train_begin:
    return "train_begin";
  case visitor_hook::train_end:
    return "train_end";
  case visitor_hook::phase_end:
    return "phase_end";
  case visitor_hook::epoch_begin:
    return "epoch_begin";
  case visitor_hook::epoch_end:
    return "epoch_end";
  case visitor_hook::batch_begin:
    return "batch_begin";
  case visitor_hook::batch_end:
    return "batch_end";
  case visitor_hook::test_begin:
    return "test_begin";
  case visitor_hook::test_end:
    return "test_end";
  case visitor_hook::validation_begin:
    return "validation_begin";
  case visitor_hook::validation_end:
    return "validation_end";
  case visitor_hook::forward_prop_begin:
    return "forward_prop_begin";
  case visitor_hook::forward_prop_end:
    return "forward_prop_end";
  case visitor_hook::backward_prop_begin:
    return "backward_prop_begin";
  case visitor_hook::backward_prop_end:
    return "backward_prop_end";
  case visitor_hook::optimize_begin:
    return "optimize_begin";
  case visitor_hook::optimize_end:
    return "optimize_end";
  case visitor_hook::batch_evaluate_begin:
    return "batch_evaluate_begin";
  case visitor_hook::batch_evaluate_end:
    return "batch_evaluate_end";
  case visitor_hook::evaluate_forward_prop_begin:
    return "evaluate_forward_prop_begin";
  case visitor_hook::evaluate_forward_prop_end:
    return "evaluate_forward_prop_end";
  case visitor_hook::invalid:
    return "invalid";
  default:
      LBANN_ERROR("Invalid visitor hook specified");
  }
}

visitor_hook visitor_hook_from_string(std::string const& str) {
  if(str == "setup_begin") {
    return visitor_hook::setup_begin;
  }else if(str == "setup_end") {
    return visitor_hook::setup_end;
  }else if(str == "train_begin") {
    return visitor_hook::train_begin;
  }else if(str == "train_end") {
    return visitor_hook::train_end;
  }else if(str == "phase_end") {
    return visitor_hook::phase_end;
  }else if(str == "epoch_begin") {
    return visitor_hook::epoch_begin;
  }else if(str == "epoch_end") {
    return  visitor_hook::epoch_end;
  }else if(str == "batch_begin") {
    return visitor_hook::batch_begin;
  }else if(str == "batch_end") {
    return visitor_hook::batch_end;
  }else if(str == "test_begin") {
    return  visitor_hook::test_begin;
  }else if(str == "test_end") {
    return visitor_hook::test_end;
  }else if(str == "validation_begin") {
    return visitor_hook::validation_begin;
  }else if(str == "validation_end") {
    return visitor_hook::validation_end;
  }else if(str == "forward_prop_begin") {
    return visitor_hook::forward_prop_begin;
  }else if(str == "forward_prop_end") {
    return visitor_hook::forward_prop_end;
  }else if(str == "backward_prop_begin") {
    return visitor_hook::backward_prop_begin;
  }else if(str == "backward_prop_end") {
    return visitor_hook::backward_prop_end;
  }else if(str == "optimize_begin") {
    return visitor_hook::optimize_begin;
  }else if(str == "optimize_end") {
    return visitor_hook::optimize_end;
  }else if(str == "batch_evaluate_begin") {
    return visitor_hook::batch_evaluate_begin;
  }else if(str == "batch_evaluate_end") {
    return visitor_hook::batch_evaluate_end;
  }else if(str == "evaluate_forward_prop_begin") {
    return visitor_hook::evaluate_forward_prop_begin;
  }else if(str == "evaluate_forward_prop_end") {
    return visitor_hook::evaluate_forward_prop_end;
  }else if(str == "invalid") {
    return visitor_hook::invalid;
  } else {
    LBANN_ERROR("\"" + str + "\" is not a valid visitor hook.");
  }
}

std::istream& operator>>(std::istream& is, visitor_hook& hook) {
  std::string tmp;
  is >> tmp;
  hook = visitor_hook_from_string(tmp);
  return is;
}

} // namespace lbann
