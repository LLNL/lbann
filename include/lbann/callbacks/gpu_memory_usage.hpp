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
//
// callback_gpu_memory_usage .hpp .cpp - Callbacks for printing GPU memory usage
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_GPU_MEMORY_USAGE_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_GPU_MEMORY_USAGE_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/** Callback hooks for printing GPU memory usage. */
class gpu_memory_usage : public callback_base {
 public:

  /** Constructor.
   */
  gpu_memory_usage() = default;
  gpu_memory_usage(const gpu_memory_usage&) = default;
  gpu_memory_usage& operator=(const gpu_memory_usage&) = default;
  gpu_memory_usage* copy() const override { return new gpu_memory_usage(*this); }
  void on_epoch_begin(model *m) override;
  std::string name() const override { return "GPU memory usage"; }
};

// Builder function
LBANN_ADD_DEFAULT_CALLBACK_BUILDER(
  gpu_memory_usage, build_gpu_memory_usage_callback_from_pbuf);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_GPU_MEMORY_USAGE_HPP_INCLUDED
