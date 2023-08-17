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
#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"

#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

#include "checkpoint_common.hpp"

namespace {
bool have_same_optimizer_type(lbann::lbann_comm const& c,
                              lbann::optimizer const& opt,
                              El::Int partner_trainer)
{
  std::size_t const my_type_hash = typeid(opt).hash_code();
  std::size_t other_type_hash = -1;
  c.sendrecv(&my_type_hash,
             1,
             partner_trainer,
             0,
             &other_type_hash,
             1,
             partner_trainer,
             0,
             El::SyncInfo<El::Device::CPU>{});
  return my_type_hash == other_type_hash;
}
} // namespace

namespace lbann {
namespace ltfb {

SendRecvWeights::SendRecvWeights(std::set<std::string> const& weights_names,
                                 bool exchange_hyperparameters)
  : BaseType(weights_names), exchange_hyperparams_{exchange_hyperparameters}
{}

SendRecvWeights::SendRecvWeights(std::set<std::string>&& weights_names,
                                 bool exchange_hyperparameters)
  : BaseType(std::move(weights_names)),
    exchange_hyperparams_{exchange_hyperparameters}
{}

std::unique_ptr<model>
SendRecvWeights::get_partner_model(model const& m,
                                   El::Int partner_trainer,
                                   size_t /*step*/)
{
  auto& comm = *m.get_comm();

  // Start by copying this model, then do the exchange.
  auto partner_model_ptr = std::make_unique<model>(m);
  model& partner_model = *partner_model_ptr;

  // Get partner process
  const El::Int rank_in_trainer = comm.get_rank_in_trainer();
  const El::Int procs_per_trainer = comm.get_procs_per_trainer();

  const bool subgrid = m.get_comm()->get_grid_type() != GridType::NO_GRID;
  const El::Int partner_rank_in_world =
    (partner_trainer * procs_per_trainer * (subgrid ? 2 : 1) + rank_in_trainer);
  comm.intertrainer_barrier();

  // Exchange weights with partner
  for (auto&& w_ptr : partner_model.get_weights()) {
    // Skip weights if name isn't in list
    auto const& weights_names = this->weights_names();
    if (!weights_names.empty() &&
        (weights_names.find(w_ptr->get_name()) == weights_names.cend())) {
      continue;
    }

    // Exchange weights values
    using TensorDataType = DataType;
    using WeightsType = data_type_weights<TensorDataType>;
    auto& recv_weights = dynamic_cast<WeightsType&>(*w_ptr);
    auto send_weights = recv_weights;
    El::SendRecv(send_weights.get_values().LockedMatrix(),
                 recv_weights.get_values().Matrix(),
                 comm.get_world_comm(),
                 partner_rank_in_world,
                 partner_rank_in_world);

    // If the two weights objects use different optimizers across
    // the set of trainers, we need to be careful about how we
    // exchange the data.

    // Skip if there is no optimizer.
    // FIXME (trb 04/14/2021): Could we hit a situation in which the
    // weights in Trainer I has an optimizer and the corresponding
    // weights for Trainer J does not?
    optimizer* send_opt = send_weights.get_optimizer();
    if (!send_opt)
      continue;

    bool const do_binary_exchange =
      exchange_hyperparams_ ||
      !have_same_optimizer_type(comm, *send_opt, partner_trainer);

    if (do_binary_exchange) {
      // Since we cannot get at the unique pointer directly, we make
      // a copy:
      auto opt_up = send_opt->clone();
      exchange(comm, opt_up, partner_trainer);
      opt_up->setup(&recv_weights);
      recv_weights.set_optimizer(std::move(opt_up));
    }
    else {
      // Exchange SGD optimizer state
      using SGDType = sgd<TensorDataType>;
      auto* send_sgd = dynamic_cast<SGDType*>(send_weights.get_optimizer());
      auto* recv_sgd = dynamic_cast<SGDType*>(recv_weights.get_optimizer());
      if (send_sgd != nullptr && recv_sgd != nullptr) {
        El::SendRecv(send_sgd->get_velocity().LockedMatrix(),
                     recv_sgd->get_velocity().Matrix(),
                     comm.get_world_comm(),
                     partner_rank_in_world,
                     partner_rank_in_world);
        continue;
      }

      // Exchange Adam optimizer state
      using AdamType = adam<TensorDataType>;
      auto* send_adam = dynamic_cast<AdamType*>(send_weights.get_optimizer());
      auto* recv_adam = dynamic_cast<AdamType*>(recv_weights.get_optimizer());
      if (send_adam != nullptr && recv_adam != nullptr) {
        El::SendRecv(send_adam->get_moment1().LockedMatrix(),
                     recv_adam->get_moment1().Matrix(),
                     comm.get_world_comm(),
                     partner_rank_in_world,
                     partner_rank_in_world);
        El::SendRecv(send_adam->get_moment2().LockedMatrix(),
                     recv_adam->get_moment2().Matrix(),
                     comm.get_world_comm(),
                     partner_rank_in_world,
                     partner_rank_in_world);
        continue;
      }
      LBANN_WARNING("Unknown optimizer type. NO EXCHANGE.");
    }
  }
  return partner_model_ptr;
}

} // namespace ltfb

} // namespace lbann
