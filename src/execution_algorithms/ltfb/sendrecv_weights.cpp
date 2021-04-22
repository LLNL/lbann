#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"

#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/optimizers/sgd.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/weights/data_type_weights_impl.hpp"

namespace lbann {

namespace ltfb {

SendRecvWeights::SendRecvWeights(std::set<std::string> const& weights_names,
                                 bool exchange_hyperparameters)
  : BaseType(weights_names), exchange_hyperparams_{exchange_hyperparameters}
{}

SendRecvWeights::SendRecvWeights(std::set<std::string>&& weights_names,
                                 bool exchange_hyperparameters)
  : BaseType(std::move(weights_names)), exchange_hyperparams_{
                                          exchange_hyperparameters}
{}

std::unique_ptr<model>
SendRecvWeights::get_partner_model(model const& m, El::Int partner_trainer, size_t /*step*/)
{
  auto&& comm = *m.get_comm();

  // Get partner process
  const El::Int rank_in_trainer = comm.get_rank_in_trainer();
  const El::Int procs_per_trainer = comm.get_procs_per_trainer();
  const El::Int partner_rank_in_world =
    (partner_trainer * procs_per_trainer + rank_in_trainer);

  // Exchange weights with partner
  for (auto&& w_ptr : m.get_weights()) {

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

    // Exchange SGD optimizer state
    using SGDType = sgd<TensorDataType>;
    auto* send_sgd = dynamic_cast<SGDType*>(send_weights.get_optimizer());
    auto* recv_sgd = dynamic_cast<SGDType*>(recv_weights.get_optimizer());
    if (send_sgd != nullptr && recv_sgd != nullptr) {
      if (exchange_hyperparams_) {
        using hyperparameters_type =
          std::tuple<TensorDataType, TensorDataType, bool>;
        hyperparameters_type hyperparameters(send_sgd->get_learning_rate(),
                                             send_sgd->get_momentum(),
                                             send_sgd->using_nesterov());
        El::mpi::SendRecv(reinterpret_cast<El::byte*>(&hyperparameters),
                          sizeof(hyperparameters_type),
                          partner_rank_in_world,
                          partner_rank_in_world,
                          comm.get_world_comm(),
                          El::SyncInfo<El::Device::CPU>{});
        recv_sgd->set_learning_rate(std::get<0>(hyperparameters));
        recv_sgd->set_momentum(std::get<1>(hyperparameters));
        recv_sgd->set_nesterov(std::get<2>(hyperparameters));
      }
      El::SendRecv(send_sgd->get_velocity().LockedMatrix(),
                   recv_sgd->get_velocity().Matrix(),
                   comm.get_world_comm(),
                   partner_rank_in_world,
                   partner_rank_in_world);
    }

    // Exchange Adam optimizer state
    using AdamType = adam<TensorDataType>;
    auto* send_adam = dynamic_cast<AdamType*>(send_weights.get_optimizer());
    auto* recv_adam = dynamic_cast<AdamType*>(recv_weights.get_optimizer());
    if (send_adam != nullptr && recv_adam != nullptr) {
      if (exchange_hyperparams_) {
        using hyperparameters_type = std::tuple<TensorDataType,
                                                TensorDataType,
                                                TensorDataType,
                                                TensorDataType,
                                                TensorDataType,
                                                TensorDataType>;
        hyperparameters_type hyperparameters(send_adam->get_learning_rate(),
                                             send_adam->get_beta1(),
                                             send_adam->get_beta2(),
                                             send_adam->get_eps(),
                                             send_adam->get_current_beta1(),
                                             send_adam->get_current_beta2());
        El::mpi::SendRecv(reinterpret_cast<El::byte*>(&hyperparameters),
                          sizeof(hyperparameters_type),
                          partner_rank_in_world,
                          partner_rank_in_world,
                          comm.get_world_comm(),
                          El::SyncInfo<El::Device::CPU>{});
        recv_adam->set_learning_rate(std::get<0>(hyperparameters));
        recv_adam->set_beta1(std::get<1>(hyperparameters));
        recv_adam->set_beta2(std::get<2>(hyperparameters));
        recv_adam->set_eps(std::get<3>(hyperparameters));
        recv_adam->set_current_beta1(std::get<4>(hyperparameters));
        recv_adam->set_current_beta2(std::get<5>(hyperparameters));
      }
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
    }
  }
  return nullptr;
}

} // namespace ltfb

} // namespace lbann
