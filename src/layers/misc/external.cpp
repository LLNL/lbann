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

#include "lbann/comm.hpp"
#include "lbann/layers/misc/external.hpp"
#include "lbann/utils/sync_info_helpers.hpp"

#include <algorithm>
#include <cstdio>
#include <dlfcn.h>

namespace lbann {

external_layer_setup_t load_external_library(const std::string& filename,
                                             const std::string& layer_name)
{
  ////////////////////////////////////////////////////
  // Load the library handles dynamically using dlopen
  void* handle = dlopen(filename.c_str(), RTLD_LAZY);
  if (!handle) {
    LBANN_ERROR(
      "Cannot load library for external layer forward pass (filename: \"",
      filename,
      "\"). Reason: ",
      dlerror());
    return nullptr;
  }

  ////////////////////////////////////////////////////
  // Collect functions from libraries based on device
  if (layer_name.length() == 0) {
    LBANN_ERROR("External layer (filename: \"",
                filename,
                "\") did not define a layer name.");
    return nullptr;
  }

  std::string const setup_funcname = std::string("setup_") + layer_name;
  external_layer_setup_t setup_funcptr =
    (external_layer_setup_t)dlsym(handle, setup_funcname.c_str());
  if (!setup_funcptr) {
    LBANN_ERROR("Malformed external library (filename: \"",
                filename,
                "\"). Reason: Missing function \"",
                setup_funcname,
                "\"");
    return nullptr;
  }

  // TODO(later): Without an object, dlclose is never called.
  return setup_funcptr;
}

} // namespace lbann
