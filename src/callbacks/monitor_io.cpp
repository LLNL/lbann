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
// monitor_io .hpp .cpp - Callback hooks for I/O monitoring
////////////////////////////////////////////////////////////////////////////////

#include <utility>

#include "lbann/callbacks/monitor_io.hpp"
#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

namespace lbann {
namespace callback {

template <class Archive>
void monitor_io::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_layers));
}

void monitor_io::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_disp_io_stats();
  msg->set_layers(protobuf::to_space_sep_string(m_layers));
}

void monitor_io::on_epoch_end(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  const data_coordinator& dc = get_const_trainer().get_data_coordinator();
  lbann_comm* comm = m->get_comm();
  std::cout << "Rank " << comm->get_trainer_rank() << "."
            << comm->get_rank_in_trainer() << " processed "
            << dc.get_num_samples(execution_mode::training)
            << " training samples of "
            << dc.get_total_num_samples(execution_mode::training) << " ("
            << dc.get_num_samples(execution_mode::training) / c.get_epoch()
            << " per epoch)" << std::endl;
}

void monitor_io::on_test_end(model* m)
{
  const auto& c =
    static_cast<const SGDExecutionContext&>(m->get_execution_context());
  const data_coordinator& dc = get_const_trainer().get_data_coordinator();
  lbann_comm* comm = m->get_comm();
  std::cout << "Rank " << comm->get_trainer_rank() << "."
            << comm->get_rank_in_trainer() << " processed "
            << dc.get_num_samples(execution_mode::testing)
            << " test samples of "
            << dc.get_total_num_samples(execution_mode::testing) << " ("
            << dc.get_num_samples(execution_mode::testing) / c.get_epoch()
            << " per epoch)" << std::endl;
}

std::unique_ptr<callback_base>
build_monitor_io_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                                    const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDispIOStats&>(proto_msg);
  return std::make_unique<monitor_io>(parse_list<std::string>(params.layers()));
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::monitor_io
#define LBANN_CLASS_LIBNAME callback_monitor_io
#include <lbann/macros/register_class_with_cereal.hpp>
