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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/proto/factories.hpp"

namespace lbann {
namespace proto {

namespace {

/** Construct a weights initialization specified with prototext. */
weights_initializer* construct_initializer(const lbann_data::Weights& proto_weights) {

  // Constant initialization
  if (proto_weights.has_constant_initializer()) {
    const auto& params = proto_weights.constant_initializer();
    return new constant_initializer(params.value());
  }

  // Value initialization
  if (proto_weights.has_value_initializer()) {
    const auto& params = proto_weights.value_initializer();
    return new value_initializer(parse_list<DataType>(params.values()));
  }

  // Random initialization
  if (proto_weights.has_uniform_initializer()) {
    const auto& params = proto_weights.uniform_initializer();
    const auto& min = params.min();
    const auto& max = params.max();
    if (min != 0.0 || max != 0.0) {
      return new uniform_initializer(min, max);
    } else {
      return new uniform_initializer();
    }
  }
  if (proto_weights.has_normal_initializer()) {
    const auto& params = proto_weights.normal_initializer();
    const auto& mean = params.mean();
    const auto& standard_deviation = params.standard_deviation();
    if (mean != 0.0 || standard_deviation != 0.0) {
      return new normal_initializer(mean, standard_deviation);
    } else {
      return new normal_initializer();
    }
  }

  // Variance scaling initialization
  if (proto_weights.has_glorot_normal_initializer()) {
    return new glorot_initializer(probability_distribution::gaussian);
  }
  if (proto_weights.has_glorot_uniform_initializer()) {
    return new glorot_initializer(probability_distribution::uniform);
  }
  if (proto_weights.has_he_normal_initializer()) {
    return new he_initializer(probability_distribution::gaussian);
  }
  if (proto_weights.has_he_uniform_initializer()) {
    return new he_initializer(probability_distribution::uniform);
  }

  return nullptr;
}

} // namespace

weights* construct_weights(lbann_comm* comm,
                           const lbann_data::Optimizer& proto_opt,
                           const lbann_data::Weights& proto_weights) {
  std::stringstream err;

  // Instantiate weights
  weights* w = new weights(comm);

  // Set weights name if provided
  const auto& name = proto_weights.name();
  const auto& parsed_name = parse_list<std::string>(name);
  if (!name.empty()) {
    if (parsed_name.empty() || parsed_name.front() != name) {
      err << "weights name \"" << name << "\" is invalid since it "
          << "contains whitespace";
      LBANN_ERROR(err.str());
    }
    w->set_name(name);
  }

  // Set weights initializer and optimizer
  std::unique_ptr<weights_initializer> init(construct_initializer(proto_weights));
  std::unique_ptr<optimizer> opt;
  if (proto_weights.has_optimizer()) {
    opt.reset(construct_optimizer(comm, proto_weights.optimizer()));
  } else {
    opt.reset(construct_optimizer(comm, proto_opt));
  }
  w->set_initializer(init);
  w->set_optimizer(opt);

  return w;

}

} // namespace proto
} // namespace lbann
