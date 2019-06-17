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

optimizer* construct_optimizer(lbann_comm* comm,
                               const lbann_data::Optimizer& proto_opt) {

  // Stochastic gradient descent
  if (proto_opt.has_sgd()) {
    const auto& params = proto_opt.sgd();
    return new sgd(comm,
                   params.learn_rate(),
                   params.momentum(),
                   params.nesterov());
  }

  // AdaGrad
  if (proto_opt.has_adagrad()) {
    const auto& params = proto_opt.adagrad();
    return new adagrad(comm, params.learn_rate(), params.eps());
  }

  // RMSProp
  if (proto_opt.has_rmsprop()) {
    const auto& params = proto_opt.rmsprop();
    return new rmsprop(comm,
                       params.learn_rate(),
                       params.decay_rate(),
                       params.eps());
  }

  // Adam
  if (proto_opt.has_adam()) {
    const auto& params = proto_opt.adam();
    return new adam(comm,
                    params.learn_rate(),
                    params.beta1(),
                    params.beta2(),
                    params.eps());
  }

  // Hypergradient Adam
  if (proto_opt.has_hypergradient_adam()) {
    const auto& params = proto_opt.hypergradient_adam();
    return new hypergradient_adam(comm,
                                  params.init_learning_rate(),
                                  params.hyper_learning_rate(),
                                  params.beta1(),
                                  params.beta2(),
                                  params.eps());
  }

  // Return null pointer if no optimizer is specified
  return nullptr;

}

} // namespace proto
} // namespace lbann
