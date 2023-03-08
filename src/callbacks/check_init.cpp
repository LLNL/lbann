////////////////////////////////////////////////////////////////////////////////
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
//
// check_init .hpp .cpp - Check multi-model init
////////////////////////////////////////////////////////////////////////////////

#include "lbann/comm_impl.hpp"
#include "lbann/callbacks/check_init.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "lbann/proto/callbacks.pb.h"

namespace lbann {
namespace callback {
namespace {
template <typename TensorDataType>
bool check_equal(const El::AbstractMatrix<TensorDataType>& x,
                 const El::AbstractMatrix<TensorDataType>& y) {
  const El::Int height = x.Height();
  const El::Int width = x.Width();
  if (height != y.Height() || width != y.Width() || x.LDim() != y.LDim()) {
    return false;
  }
  const TensorDataType *x_buf = x.LockedBuffer();
  const TensorDataType *y_buf = y.LockedBuffer();
  for (El::Int i = 0; i < height * width; ++i) {
    if (x_buf[i] != y_buf[i]) {
      return false;
    }
  }
  return true;
}
}// namespace <anon>

template <class Archive>
void check_init::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)));
}

void check_init::write_specific_proto(lbann_data::Callback& proto) const
{
  proto.mutable_init();
}

void check_init::on_train_begin(model *m) {
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  // Skip after the first epoch.
  if (c.get_epoch() != 0) {
    return;
  }
  lbann_comm *comm = m->get_comm();
  if (comm->am_world_master()) {
    std::cout << "Checking all model initial weights match..." << std::endl;
  }
  if (comm->get_num_trainers() == 1) {
    return;
  }

  for (const auto* w : m->get_weights()) {
    if (comm->am_world_master()) {
      std::cout << "Checking " << w->get_name() << std::endl;
    }
    // Model 0 holds the master copy, it gathers the values from other models
    // and compares them.
    auto const& dtw = dynamic_cast<data_type_weights<DataType> const&>(*w);
    const auto& local_matrix = dtw.get_values().LockedMatrix();
    CPUMat remote_matrix(local_matrix.Height(), local_matrix.Width());
    for (int model = 1; model < comm->get_num_trainers(); ++model) {
      comm->global_barrier();
      if (comm->get_trainer_rank() == 0) {
        comm->recv(remote_matrix, model);
        if (!check_equal(local_matrix, remote_matrix)) {
          LBANN_ERROR("check_init: "
                      "model ", model, " "
                      "rank in model ", comm->get_rank_in_trainer(), " "
                      "does not match model 0");
        }
      } else if (comm->get_trainer_rank() == model) {
        comm->send(local_matrix, 0);
      }
    }
  }
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::check_init
#define LBANN_CLASS_LIBNAME callback_check_init
#include <lbann/macros/register_class_with_cereal.hpp>
