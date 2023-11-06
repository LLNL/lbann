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
// dump_minibatch_sample_indices .hpp .cpp - Callbacks
// to dump the list of indices per minibatch
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/dump_minibatch_sample_indices.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/data_ingestion/coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"
#include <vector>

#include "lbann/proto/callbacks.pb.h"

#include <cstdlib>
#include <iomanip>

namespace lbann {
namespace callback {

dump_minibatch_sample_indices::dump_minibatch_sample_indices()
  : dump_minibatch_sample_indices("", 0)
{}

template <class Archive>
void dump_minibatch_sample_indices::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_basename));
}

void dump_minibatch_sample_indices::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_dump_mb_indices();
  msg->set_basename(m_basename);
  msg->set_interval(m_batch_interval);
}

void dump_minibatch_sample_indices::dump_to_file(model* m,
                                                 Layer* l,
                                                 int64_t step)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  // Print minibatch sample indices of the data coordinator
  data_coordinator& dc = get_trainer().get_data_coordinator();
  El::Matrix<El::Int>* indices =
    dc.get_sample_indices_per_mb(c.get_execution_mode());
  if (indices == nullptr || indices->Height() == 0 || indices->Width() == 0) {
    return;
  }

  const std::string path = get_multi_trainer_path(*m, m_basename);
  makedir(path.c_str());
  const std::string file =
    (path + to_string(c.get_execution_mode()) + "_e" +
     std::to_string(c.get_epoch()) + "_s" + std::to_string(c.get_step()) +
     "_r" + std::to_string(get_trainer().get_comm()->get_rank_in_trainer()) +
     "-MB_Sample_Indices");
  El::Write(*indices, file, El::ASCII);
}

void dump_minibatch_sample_indices::on_forward_prop_end(model* m, Layer* l)
{
  const auto& c = m->get_execution_context();
  dump_to_file(m, l, c.get_step());
}

void dump_minibatch_sample_indices::on_evaluate_forward_prop_end(model* m,
                                                                 Layer* l)
{
  const auto& c = m->get_execution_context();
  dump_to_file(m, l, c.get_step());
}

std::unique_ptr<callback_base> build_dump_mb_indices_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDumpMBIndices&>(proto_msg);
  return std::make_unique<dump_minibatch_sample_indices>(params.basename(),
                                                         params.interval());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::dump_minibatch_sample_indices
#define LBANN_CLASS_LIBNAME callback_dump_minibatch_sample_indices
#include <lbann/macros/register_class_with_cereal.hpp>
