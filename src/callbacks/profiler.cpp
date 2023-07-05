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
// timer .hpp .cpp - Callback hooks to time training
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/profiler.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/weights.hpp"

#include "lbann/proto/callbacks.pb.h"

#ifdef LBANN_NVPROF
#include "cuda_runtime.h"
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#endif

#include <algorithm>
#include <string>

namespace lbann {
namespace callback {

profiler::profiler(bool sync, bool skip_init)
  : callback_base(), m_sync(sync), m_skip_init(skip_init)
{
#ifdef LBANN_NVPROF
  nvtxNameCudaStreamA(hydrogen::cuda::GetDefaultStream(), "Hydrogen");
#endif
  if (!m_skip_init) {
    prof_start();
  }
}

template <class Archive>
void profiler::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_sync),
     CEREAL_NVP(m_skip_init));
}

void profiler::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_profiler();
  msg->set_sync(m_sync);
  msg->set_skip_init(m_skip_init);
}

void profiler::on_epoch_begin(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  // Skip the first epoch
  if (m_skip_init && c.get_epoch() == 1) {
    prof_start();
  }
  prof_region_begin(("epoch " + std::to_string(c.get_epoch())).c_str(),
                    prof_colors[0],
                    m_sync);
}

void profiler::on_epoch_end(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  prof_region_end(("epoch " + std::to_string(c.get_epoch())).c_str(), m_sync);
}

void profiler::on_validation_begin(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  prof_region_begin(("val " + std::to_string(c.get_epoch())).c_str(),
                    prof_colors[0],
                    m_sync);
}

void profiler::on_validation_end(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  prof_region_end(("val " + std::to_string(c.get_epoch())).c_str(), m_sync);
}

void profiler::on_test_begin(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  prof_region_begin(("test " + std::to_string(c.get_epoch())).c_str(),
                    prof_colors[0],
                    m_sync);
}

void profiler::on_test_end(model* m)
{
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  prof_region_end(("test " + std::to_string(c.get_epoch())).c_str(), m_sync);
}

void profiler::on_batch_begin(model* m)
{
  const auto& c = m->get_execution_context();
  prof_region_begin(("batch " + std::to_string(c.get_step())).c_str(),
                    prof_colors[1],
                    m_sync);
}

void profiler::on_batch_end(model* m)
{
  const auto& c = m->get_execution_context();
  prof_region_end(("batch " + std::to_string(c.get_step())).c_str(), m_sync);
}

void profiler::on_batch_evaluate_begin(model* m)
{
  const auto& c = m->get_execution_context();
  prof_region_begin(("batch eval " + std::to_string(c.get_step())).c_str(),
                    prof_colors[1],
                    m_sync);
}

void profiler::on_batch_evaluate_end(model* m)
{
  const auto& c = m->get_execution_context();
  prof_region_end(("batch eval " + std::to_string(c.get_step())).c_str(),
                  m_sync);
}

void profiler::on_forward_prop_begin(model* m)
{
  prof_region_begin("forward", prof_colors[2], m_sync);
}

void profiler::on_forward_prop_end(model* m)
{
  prof_region_end("forward", m_sync);
}

void profiler::on_evaluate_forward_prop_begin(model* m)
{
  prof_region_begin("forward", prof_colors[2], m_sync);
}

void profiler::on_evaluate_forward_prop_end(model* m)
{
  prof_region_end("forward", m_sync);
}

void profiler::on_backward_prop_begin(model* m)
{
  prof_region_begin("backward", prof_colors[3], m_sync);
}

void profiler::on_backward_prop_end(model* m)
{
  prof_region_end("backward", m_sync);
}

void profiler::on_optimize_begin(model* m)
{
  prof_region_begin("optimize", prof_colors[4], m_sync);
}

void profiler::on_optimize_end(model* m)
{
  prof_region_end("optimize", m_sync);
}

int profiler::get_color(Layer* l)
{
  const std::string& lname = l->get_type();
  int idx = 5;
  if (lname == "fully connected") {
    idx = 6;
  }
  else if (lname == "convolution") {
    idx = 7;
  }
  else if (lname == "pooling") {
    idx = 8;
  }
  else if (lname == "input:partitioned") {
    idx = 9;
  }
  else if (lname == "input:distributed") {
    idx = 9;
  }
  else if (lname == "batch normalization") {
    idx = 10;
  }
  else if (lname == "softmax") {
    idx = 11;
  }
  else if (lname == "ReLU") {
    idx = 12;
  }
  else if (lname == "split") {
    idx = 13;
  }
  else if (lname == "sum") {
    idx = 13;
  }
  return prof_colors[idx % num_prof_colors];
}

void profiler::on_forward_prop_begin(model* m, Layer* l)
{
  prof_region_begin(("fw " + l->get_name()).c_str(), get_color(l), m_sync);
}

void profiler::on_forward_prop_end(model* m, Layer* l)
{
  prof_region_end(("fw " + l->get_name()).c_str(), m_sync);
}

void profiler::on_evaluate_forward_prop_begin(model* m, Layer* l)
{
  prof_region_begin(("fw " + l->get_name()).c_str(), get_color(l), m_sync);
}

void profiler::on_evaluate_forward_prop_end(model* m, Layer* l)
{
  prof_region_end(("fw " + l->get_name()).c_str(), m_sync);
}

void profiler::on_backward_prop_begin(model* m, Layer* l)
{
  prof_region_begin(("bw " + l->get_name()).c_str(), get_color(l), m_sync);
}

void profiler::on_backward_prop_end(model* m, Layer* l)
{
  prof_region_end(("bw " + l->get_name()).c_str(), m_sync);
}

void profiler::on_optimize_begin(model* m, weights* w)
{
  prof_region_begin(("opt " + w->get_name()).c_str(), prof_colors[5], m_sync);
}

void profiler::on_optimize_end(model* m, weights* w)
{
  prof_region_end(("opt " + w->get_name()).c_str(), m_sync);
}

std::unique_ptr<callback_base>
build_profiler_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                                  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackProfiler&>(proto_msg);
  return std::make_unique<profiler>(params.sync(), params.skip_init());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::profiler
#define LBANN_CLASS_LIBNAME callback_profiler
#include <lbann/macros/register_class_with_cereal.hpp>
