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

#include "lbann/callbacks/mixup.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/beta.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/serialize.hpp"
#include <algorithm>

#include "lbann/proto/callbacks.pb.h"

#include <unordered_set>

namespace lbann {
namespace callback {

mixup::mixup(std::unordered_set<std::string> layers, float alpha)
  : callback_base(), m_layers(layers), m_alpha(alpha)
{
  if (alpha < 0.0f) {
    LBANN_ERROR("Mixup alpha must be non-negative.");
  }
}

mixup::mixup() : mixup({}, 0) {}

template <class Archive>
void mixup::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_layers),
     CEREAL_NVP(m_alpha));
}

void mixup::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_mixup();
  msg->set_layers(protobuf::to_space_sep_string(m_layers));
  msg->set_alpha(m_alpha);
}

void mixup::on_forward_prop_end(model* m, Layer* l)
{
  LBANN_CALIPER_MARK_SCOPE("mixup::on_forward_prop_end");
  if (!m_layers.count(l->get_name())) {
    return;
  }
  const auto& c =
    dynamic_cast<const SGDExecutionContext&>(m->get_execution_context());
  if (c.get_execution_mode() != execution_mode::training) {
    return; // No mixup outside of training.
  }

  auto* dtl = dynamic_cast<data_type_layer<DataType>*>(l);
  auto& samples_orig = dtl->get_local_activations(0);
  auto& labels_orig = dtl->get_local_activations(1);
  if (samples_orig.GetDevice() != El::Device::CPU ||
      labels_orig.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("Mixup requires CPU data.");
  }
  // Copy samples.
  // Assumes data are on CPU.
  CPUMat samples, labels;
  El::Copy(samples_orig, samples);
  El::Copy(labels_orig, labels);
  El::Int mbsize = samples.Width();
  const El::Int samples_height = samples.Height();
  const El::Int labels_height = labels.Height();
  beta_distribution<float> dist(m_alpha, m_alpha);

  // For now, data must be on the CPU.
  if (samples.GetDevice() != El::Device::CPU ||
      labels.GetDevice() != El::Device::CPU) {
    LBANN_ERROR("mixup only works with CPU data");
  }

  // Decide how to mix the mini-batch.
  std::vector<El::Int> shuffled_indices(mbsize);
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
  std::shuffle(shuffled_indices.begin(),
               shuffled_indices.end(),
               get_fast_generator());

  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < mbsize; ++i) {
    const El::Int j = shuffled_indices[i];
    if (i == j) {
      continue;
    }
    float lambda = dist(get_OMP_fast_generator());
    lambda = std::max(lambda, 1.0f - lambda);
    const float lambda_sub = 1.0f - lambda;
    const DataType* __restrict__ x1_buf = samples.LockedBuffer(0, i);
    const DataType* __restrict__ x2_buf = samples.LockedBuffer(0, j);
    DataType* __restrict__ x = samples_orig.Buffer(0, i);
    const DataType* __restrict__ y1_buf = labels.LockedBuffer(0, i);
    const DataType* __restrict__ y2_buf = labels.LockedBuffer(0, j);
    DataType* __restrict__ y = labels_orig.Buffer(0, i);
    for (El::Int k = 0; k < samples_height; ++k) {
      x[k] = lambda * x1_buf[k] + lambda_sub * x2_buf[k];
    }
    for (El::Int k = 0; k < labels_height; ++k) {
      y[k] = lambda * y1_buf[k] + lambda_sub * y2_buf[k];
    }
  }
}

std::unique_ptr<callback_base>
build_mixup_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                               const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackMixup&>(proto_msg);
  const auto& layers_list = parse_list<std::string>(params.layers());
  std::unordered_set<std::string> layers(layers_list.begin(),
                                         layers_list.end());
  return std::make_unique<mixup>(layers, params.alpha());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::mixup
#define LBANN_CLASS_LIBNAME callback_mixup
#include <lbann/macros/register_class_with_cereal.hpp>
