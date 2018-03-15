////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
    const auto& proto_sgd = proto_opt.sgd();
    const auto& learn_rate = proto_sgd.learn_rate();
    const auto& momentum = proto_sgd.momentum();
    const auto& nesterov = proto_sgd.nesterov();
    return new sgd(comm, learn_rate, momentum, nesterov);
  }

  // AdaGrad
  if (proto_opt.has_adagrad()) {
    const auto& proto_adagrad = proto_opt.adagrad();
    const auto& learn_rate = proto_adagrad.learn_rate();
    const auto& eps = proto_adagrad.eps();
    return new adagrad(comm, learn_rate, eps);
  }

  // RMSProp
  if (proto_opt.has_rmsprop()) {
    const auto& proto_rmsprop = proto_opt.rmsprop();
    const auto& learn_rate = proto_rmsprop.learn_rate();
    const auto& decay_rate = proto_rmsprop.decay_rate();
    const auto& eps = proto_rmsprop.eps();
    return new rmsprop(comm, learn_rate, decay_rate, eps);
  }

  // Adam
  if (proto_opt.has_adam()) {
    const auto& proto_adam = proto_opt.adam();
    const auto& learn_rate = proto_adam.learn_rate();
    const auto& beta1 = proto_adam.beta1();
    const auto& beta2 = proto_adam.beta2();
    const auto& eps = proto_adam.eps();
    return new adam(comm, learn_rate, beta1, beta2, eps);
  }

  // Hypergradient Adam
  if (proto_opt.has_hypergradient_adam()) {
    const auto& proto_hypergradient_adam = proto_opt.hypergradient_adam();
    const auto& init_lr = proto_hypergradient_adam.init_learning_rate();
    const auto& hyper_lr = proto_hypergradient_adam.hyper_learning_rate();
    const auto& beta1 = proto_hypergradient_adam.beta1();
    const auto& beta2 = proto_hypergradient_adam.beta2();
    const auto& eps = proto_hypergradient_adam.eps();
    return new hypergradient_adam(comm, init_lr, hyper_lr, beta1, beta2, eps);
  }

  // Return null pointer if no optimizer is specified
  return nullptr;

}

} // namespace proto
} // namespace lbann
