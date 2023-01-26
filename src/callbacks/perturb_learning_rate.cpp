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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/comm_impl.hpp"
#include "lbann/callbacks/perturb_learning_rate.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/random_number_generators.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/protobuf.hpp"

#include <callbacks.pb.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <sstream>
#include <utility>

namespace lbann {
namespace callback {

perturb_learning_rate::perturb_learning_rate(DataType learning_rate_factor,
                           bool perturb_during_training,
                           El::Int batch_interval,
                           std::set<std::string> weights_names)
  : callback_base(batch_interval),
    m_learning_rate_factor(learning_rate_factor),
    m_perturb_during_training(perturb_during_training),
    m_weights_names(std::move(weights_names)) {}

perturb_learning_rate::perturb_learning_rate()
  : perturb_learning_rate(0, false, 0)
{}

template <class Archive>
void perturb_learning_rate::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_learning_rate_factor),
     CEREAL_NVP(m_perturb_during_training),
     CEREAL_NVP(m_weights_names));
}

void perturb_learning_rate::write_specific_proto(
  lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_perturb_learning_rate();
  msg->set_learning_rate_factor(m_learning_rate_factor);
  msg->set_perturb_during_training(m_perturb_during_training);
  msg->set_batch_interval(m_batch_interval);
  msg->set_weights(protobuf::to_space_sep_string(m_weights_names));
}

void perturb_learning_rate::setup(model* m) {
  perturb(*m);
}

void perturb_learning_rate::on_batch_begin(model* m) {
  const auto& c = m->get_execution_context();
  if (m_perturb_during_training &&
      c.get_step() % m_batch_interval == 0 &&
      c.get_step() > 0) {
    perturb(*m);
  }
}

void perturb_learning_rate::perturb(model& m) const {
  auto* comm = m.get_comm();
  for (auto* w : m.get_weights()) {
    if (w == nullptr) {
      LBANN_ERROR("callback \"", name(), "\" "
                  "got a weights pointer that is a null pointer");
    }
    if (m_weights_names.empty()
        || m_weights_names.count(w->get_name()) > 0) {

      // Check if weights has optimizer
      auto *opt = dynamic_cast<data_type_optimizer<DataType>*>(w->get_optimizer());
      if (!m_weights_names.empty() && opt == nullptr) {
        auto* opt_ = w->get_optimizer();
        LBANN_ERROR(
          "callback \"", name(), "\" "
          "expected weights \"", w->get_name(), "\" "
          "to have an optimizer, but found ",
          (opt_ ? opt_->get_type() : "no optimizer"));
      }

      // Perturb optimizer learning rate
      if (opt) {
        perturb(*comm, *opt);
      }

    }
  }
}

void perturb_learning_rate::perturb(lbann_comm& comm, data_type_optimizer<DataType>& opt) const {

  // Perturb learning rate on trainer master processes
  DataType new_lr;
  if (comm.am_trainer_master()) {

    // Useful constants
    constexpr DataType zero = 0;
    constexpr DataType one = 1;
    constexpr DataType min_val = std::numeric_limits<DataType>::min();

    // RNG
    auto& gen = get_generator();
    std::normal_distribution<DataType> dist(zero, one);

    // Perturb log(learning_rate)
    DataType learning_rate = opt.get_learning_rate();
    if (m_learning_rate_factor != zero && learning_rate >= zero) {
      auto log_val = std::log(std::max(learning_rate, min_val));
      log_val += m_learning_rate_factor * dist(gen);
      learning_rate = std::exp(log_val);
    }
    new_lr = learning_rate;

  }

  // Communicate new lr  from trainer master processes
  comm.trainer_broadcast(comm.get_trainer_master(),new_lr);

  // Workers update new lr
  opt.set_learning_rate(new_lr);

}

std::unique_ptr<callback_base>
build_perturb_learning_rate_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackPerturbLearningRate&>(proto_msg);
  return std::make_unique<perturb_learning_rate>(
    params.learning_rate_factor(),
    params.perturb_during_training(),
    params.batch_interval(),
    parse_set<std::string>(params.weights()));
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::perturb_learning_rate
#define LBANN_CLASS_LIBNAME callback_perturb_learning_rate
#include <lbann/macros/register_class_with_cereal.hpp>
