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

#include "lbann/callbacks/check_small.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

namespace lbann {
namespace callback {
namespace {
template <typename TensorDataType>
bool is_good(const El::AbstractDistMatrix<TensorDataType>& m)
{
  static const TensorDataType threshold =
    El::Sqrt(std::numeric_limits<TensorDataType>::min());

  const auto& local_mat = m.LockedMatrix();
  const El::Int height = local_mat.Height();
  const El::Int width = local_mat.Width();
  for (El::Int col = 0; col < width; ++col) {
    for (El::Int row = 0; row < height; ++row) {
      const auto val = std::abs(local_mat(row, col));
      if (val > TensorDataType(0) && val <= threshold) {
        std::cout << "Found small value " << val << " at (" << row << "," << col
                  << ")!" << std::endl;
        return false;
      }
    }
  }
  return true;
}
} // namespace

template <class Archive>
void check_small::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)));
}

void check_small::write_specific_proto(lbann_data::Callback& proto) const
{
  proto.mutable_check_small();
}

void check_small::on_forward_prop_end(model* m, Layer* l)
{
  const auto& c = m->get_execution_context();
  auto& dtl = dynamic_cast<data_type_layer<DataType>&>(*l);
  const auto& acts = dtl.get_activations();
  if (!is_good(acts)) {
    LBANN_ERROR(name(),
                ": "
                "[",
                std::to_string(m->get_comm()->get_rank_in_world()),
                "]: "
                "error in activations of ",
                l->get_name(),
                " "
                "(step=",
                std::to_string(c.get_step()),
                ")");
  }
}

void check_small::on_backward_prop_end(model* m)
{
  const auto& c = m->get_execution_context();
  for (weights* w : m->get_weights()) {
    auto& dtw = dynamic_cast<data_type_weights<DataType>&>(*w);
    auto* opt = dtw.get_optimizer();
    if (opt != nullptr && !is_good(opt->get_gradient_sharded())) {
      LBANN_ERROR(name(),
                  ": "
                  "[",
                  std::to_string(m->get_comm()->get_rank_in_world()),
                  "]: "
                  "error in weights gradient of ",
                  dtw.get_name(),
                  " "
                  "(step=",
                  std::to_string(c.get_step()),
                  ")");
    }
  }
}

void check_small::on_batch_end(model* m)
{
  const auto& c = m->get_execution_context();
  for (weights* w : m->get_weights()) {
    auto& dtw = dynamic_cast<data_type_weights<DataType>&>(*w);
    if (!is_good(dtw.get_values())) {
      LBANN_ERROR(name(),
                  ": "
                  "[",
                  std::to_string(m->get_comm()->get_rank_in_world()),
                  "]: "
                  "error in weights of ",
                  w->get_name(),
                  " "
                  "(step=",
                  std::to_string(c.get_step() - 1),
                  ")");
    }
  }
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::check_small
#define LBANN_CLASS_LIBNAME callback_check_small
#include <lbann/macros/register_class_with_cereal.hpp>
