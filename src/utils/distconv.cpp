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

#include "lbann/utils/distconv.hpp"
#include "lbann/utils/cudnn.hpp"
#include <memory>

#ifdef LBANN_HAS_DISTCONV

namespace lbann {
namespace dc {

////////////////////////////////////////////////////////////
// Global Distconv objects
////////////////////////////////////////////////////////////

namespace {

/** Global instance of cuDNN handle. */
std::unique_ptr<Backend> backend_instance;

void initialize() {
  auto &cudnn_h = lbann::cudnn::get_handle();
  cudaStream_t s;
  CHECK_CUDNN(cudnnGetStream(cudnn_h, &s));
  backend_instance.reset(
      new Backend(cudnn_h, s));
}

void destroy() {
  backend_instance.reset();
}

} // namespace

Backend &get_backend() {
  if (!backend_instance) { initialize(); }
  return *backend_instance;
}

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
