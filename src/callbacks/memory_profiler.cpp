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
//
// memory_profiler .hpp .cpp - Itemized memory usage profiling.
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/memory_profiler.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/weights.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>
#include <string>

namespace lbann {
namespace callback {

memory_profiler::memory_profiler() : callback_base() {}

memory_profiler::~memory_profiler() {}

template <class Archive>
void memory_profiler::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)));
}

void memory_profiler::write_specific_proto(lbann_data::Callback& proto) const
{
  // auto* msg = proto.mutable_memory_profiler();
}

void memory_profiler::on_setup_begin(model* m) {}
void memory_profiler::on_setup_end(model* m) {}
void memory_profiler::on_forward_prop_begin(model* m) {}
void memory_profiler::on_forward_prop_begin(model* m, Layer* l) {}
void memory_profiler::on_forward_prop_end(model* m, Layer* l) {}
void memory_profiler::on_backward_prop_begin(model* m, Layer* l) {}
void memory_profiler::on_backward_prop_end(model* m, Layer* l) {}

std::unique_ptr<callback_base> build_memory_profiler_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  /*const auto& params =
    dynamic_cast<const
    lbann_data::Callback::CallbackMemoryProfiler&>(proto_msg);*/
  return std::make_unique<memory_profiler>();
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::memory_profiler
#define LBANN_CLASS_LIBNAME callback_memory_profiler
#include <lbann/macros/register_class_with_cereal.hpp>
