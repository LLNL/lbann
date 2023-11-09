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

#include "lbann/callbacks/callback.hpp"
#include "lbann/data_coordinator/buffered_data_coordinator.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/trainers/trainer.hpp"

#include "lbann/proto/trainer.pb.h"

namespace lbann {
namespace proto {

std::unique_ptr<trainer>
construct_trainer(lbann_comm* comm, const lbann_data::Trainer& proto_trainer)
{

  auto proto_datatype = resolve_default_datatype(
    proto_trainer.data_coordinator().datatype());
  std::unique_ptr<data_coordinator> dc;
#define TEMPLATE_INSTANTIATION(TensorDataType)                                 \
  do {                                                                         \
    if (proto_datatype == TypeToProtoDataType<TensorDataType>::value) {        \
      dc = std::make_unique<buffered_data_coordinator<TensorDataType>>(comm);  \
    }                                                                          \
  } while (0)

#define PROTO(T) TEMPLATE_INSTANTIATION(T)

#include "lbann/macros/instantiate.hpp"

#undef PROTO
#undef TEMPLATE_INSTANTIATION

  if (dc == nullptr) {
    LBANN_ERROR("Could not construct data coordinator");
  }

  // Instantiate trainer
  auto t = std::make_unique<trainer>(
    comm,
    std::move(dc),
    proto_trainer.mini_batch_size(),
    (proto_trainer.has_training_algorithm()
       ? make_abstract<TrainingAlgorithm>(proto_trainer.training_algorithm())
       : nullptr));
  const auto& name = proto_trainer.name();
  if (!name.empty()) {
    t->set_name(name);
  }

  // Construct callbacks
  for (int i = 0; i < proto_trainer.callback_size(); i++) {
    t->add_callback(construct_callback(proto_trainer.callback(i)));
  }

  return t;
}

} // namespace proto
} // namespace lbann
