
#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"

#include "lbann/comm_impl.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/models/model.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include <memory>
#include <training_algorithm.pb.h>

namespace lbann {
namespace ltfb {
namespace {
// FIXME (trb 03/18/21): Move these out of here.
#define LBANN_LOG_WORLD_MASTER(comm_ref, ...)                                  \
  do {                                                                         \
    if (comm_ref.am_world_master())                                            \
      Output(std::clog, __VA_ARGS__);                                          \
  } while (0)

#define LBANN_LOG_TRAINER_MASTER(comm_ref, ...)                                \
  do {                                                                         \
    if (comm_ref.am_trainer_master())                                          \
      Output(std::clog, __VA_ARGS__);                                          \
  } while (0)

template <typename... Args> void Output(std::ostream& os, Args&&... args)
{
  (os << ... << args) << "\n";
}
} // namespace

// RandomPairwiseExchange implementation

RandomPairwiseExchange::RandomPairwiseExchange(
  std::string metric_name,
  metric_strategy winner_strategy,
  std::unique_ptr<ExchangeStrategy> comm_algo)
  : m_metric_strategy{winner_strategy}, m_metric_name{std::move(metric_name)},
    m_comm_algo{std::move(comm_algo)}
{}

RandomPairwiseExchange::RandomPairwiseExchange(
  RandomPairwiseExchange const& other)
  : m_metric_strategy{other.m_metric_strategy},
    m_metric_name{other.m_metric_name}, m_comm_algo{other.m_comm_algo->clone()}
{}

EvalType RandomPairwiseExchange::evaluate_model(model& m,
                                                ExecutionContext& ctxt,
                                                data_coordinator& dc) const
{
  // Make sure data readers finish asynchronous work
  const auto original_mode = ctxt.get_execution_mode();
  dc.collect_background_data_fetch(original_mode);

  if (!dc.is_execution_mode_valid(execution_mode::tournament)) {
    LBANN_ERROR("LTFB requires ",
                to_string(execution_mode::tournament),
                " execution mode");
  }
  // Mark the data store as loading - Note that this is a temporary fix
  // for the current use of the tournament
  m.mark_data_store_explicitly_loading(execution_mode::tournament);

  // Evaluate model on validation set
  get_trainer().evaluate(&m, execution_mode::tournament);

  // Get metric value
  bool found_metric = false;
  EvalType metric_value = 0;
  for (const auto& met : m.get_metrics()) {
    if (met->name() == m_metric_name) {
      found_metric = true;
      metric_value = met->get_mean_value(execution_mode::tournament);
      break;
    }
  }
  if (!found_metric) {
    LBANN_ERROR("could not find metric \"",
                m_metric_name,
                "\" ",
                "in model \"",
                m.get_name(),
                "\"");
  }

  // Mark the data store as loaded - Note that this is a temporary fix
  // for the current use of the tournament
  m.make_data_store_preloaded(execution_mode::tournament);

  // Clean up and return metric value
  m.reset_mode(ctxt, original_mode);
  dc.reset_mode(ctxt);
  return metric_value;
}

El::Int RandomPairwiseExchange::get_partner_trainer(
  lbann_comm const& comm) const noexcept
{
  // Assign partner trainers
  // Note: The first trainer in 'trainers' is paired with the
  // second, the third with the fourth, and so on. If there are an
  // odd number of trainers, the last one is partnered with itself.
  const El::Int num_trainers = comm.get_num_trainers();
  std::vector<El::Int> trainers(num_trainers);
  std::iota(trainers.begin(), trainers.end(), 0);
  // Everyone use a special RNG that is only for LTFB so that they
  // can all communicate the same pairs without communication
  std::shuffle(trainers.begin(), trainers.end(), get_ltfb_generator());

  if (comm.am_world_master()) { // Root process
    // Print partner assignments to standard output
    std::ostringstream msg;
    msg << "tournament partners -";
    for (El::Int i = 0; i < num_trainers; i += 2) {
      msg << (i > 0 ? "," : "") << " {" << trainers[i];
      if (i + 1 < num_trainers) {
        msg << "," << trainers[i + 1];
      }
      msg << "}";
    }
    msg << "\n";
    LBANN_LOG_WORLD_MASTER(comm, msg.str());
  }

  // Setup partner assignments for all processes
  std::vector<El::Int> send_buffer(num_trainers);
  for (El::Int i = 0; i < num_trainers; i += 2) {
    const auto& trainer1 = trainers[i];
    const auto& trainer2 = (i + 1 < num_trainers) ? trainers[i + 1] : trainer1;
    send_buffer[trainer1] = trainer2;
    send_buffer[trainer2] = trainer1;
  }
  return send_buffer[comm.get_trainer_rank()];
}

bool RandomPairwiseExchange::local_is_better(EvalType local,
                                             EvalType remote) const noexcept
{
  switch (m_metric_strategy) {
  case metric_strategy::LOWER_IS_BETTER:
    return (local <= remote);
  case metric_strategy::HIGHER_IS_BETTER:
    return (local >= remote);
  }
}

void RandomPairwiseExchange::select_next(model& m,
                                         ltfb::ExecutionContext& ctxt,
                                         data_coordinator& dc) const
{
  auto const& comm = *(m.get_comm());
  auto const step = ctxt.get_step();
  const std::string message_prefix =
    (comm.am_world_master() ? build_string("LTFB (model \"",
                                           m.get_name(),
                                           "\", "
                                           "step ",
                                           step,
                                           "): ")
                            : "");

  LBANN_LOG_WORLD_MASTER(comm, message_prefix, "starting tournament...");

  El::Int const local_trainer = comm.get_trainer_rank();
  El::Int const partner_trainer = get_partner_trainer(comm);

  LBANN_LOG_WORLD_MASTER(comm, message_prefix, "evaluating local model...");

  auto const local_score = evaluate_model(m, ctxt, dc);

  LBANN_LOG_WORLD_MASTER(comm, message_prefix, "exchanging model data...");

  // The "local_model" is passed in here to accommodate the
  // "sendrecv_weights" strategy; other than that, I don't think it
  // should be necessary.
  auto partner_model =
    m_comm_algo->get_partner_model(m, partner_trainer, ctxt.get_step());

  LBANN_LOG_WORLD_MASTER(comm, message_prefix, "evaluating partner model...");

  auto const partner_score = evaluate_model(*partner_model, ctxt, dc);

  // If we win, we do nothing. The input model is the winner, so no
  // further action is required. Otherwise, swap models.
  El::Int const tournament_winner =
    (local_is_better(local_score, partner_score) ? local_trainer
                                                 : partner_trainer);

  if (tournament_winner == partner_trainer) {
    // FIXME (trb 03/18/21): This is ... not great. We need to
    // unravel the "fake" polymorphism in the model non-hierarchy
    // soon.
    using DAGModel = directed_acyclic_graph_model;
    auto& local_model = dynamic_cast<DAGModel&>(m);
    auto& partner_dag_model = dynamic_cast<DAGModel&>(*partner_model);
    local_model = std::move(partner_dag_model);
    auto& trainer = get_trainer();
    auto&& metadata = trainer.get_data_coordinator().get_dr_metadata();
    m.setup(trainer.get_max_mini_batch_size(),
            metadata,
            /*force=*/true);
  }

  LBANN_LOG_TRAINER_MASTER(comm,
                           "trainer ",
                           local_trainer,
                           " selected model from trainer ",
                           tournament_winner,
                           " (Trainer ",
                           local_trainer,
                           " score = ",
                           local_score,
                           ", trainer ",
                           partner_trainer,
                           " score = ",
                           partner_score,
                           ")");
}

} // namespace ltfb
} // namespace lbann

namespace {

using ExchangeStrategyFactory = lbann::generic_factory<
  lbann::ltfb::RandomPairwiseExchange::ExchangeStrategy,
  std::string,
  lbann::proto::generate_builder_type<
    lbann::ltfb::RandomPairwiseExchange::ExchangeStrategy,
    std::set<std::string>,
    google::protobuf::Message const&>>;

std::unique_ptr<lbann::ltfb::CheckpointBinary>
make_checkpoint_binary(std::set<std::string> weights_names,
                       google::protobuf::Message const& msg)
{
  using CkptBinary =
    lbann_data::RandomPairwiseExchange::ExchangeStrategy::CheckpointBinary;
  LBANN_ASSERT(dynamic_cast<CkptBinary const*>(&msg));
  return std::make_unique<lbann::ltfb::CheckpointBinary>(
    std::move(weights_names));
}

std::unique_ptr<lbann::ltfb::CheckpointFile>
make_checkpoint_file(std::set<std::string> weights_names,
                     google::protobuf::Message const& msg)
{
  using CkptFile =
    lbann_data::RandomPairwiseExchange::ExchangeStrategy::CheckpointFile;
  auto const& params = dynamic_cast<CkptFile const&>(msg);
  return std::make_unique<lbann::ltfb::CheckpointFile>(std::move(weights_names),
                                                       params.base_dir());
}

std::unique_ptr<lbann::ltfb::SendRecvWeights>
make_sendrecv_weights(std::set<std::string> weights_names,
                      google::protobuf::Message const& msg)
{
  using SendRecvWeights =
    lbann_data::RandomPairwiseExchange::ExchangeStrategy::SendRecvWeights;
  auto const& params = dynamic_cast<SendRecvWeights const&>(msg);
  return std::make_unique<lbann::ltfb::SendRecvWeights>(
    std::move(weights_names),
    params.exchange_hyperparameters());
}

lbann::ltfb::RandomPairwiseExchange::metric_strategy
to_lbann(lbann_data::RandomPairwiseExchange::MetricStrategy strategy)
{
  using LBANNEnumType = lbann::ltfb::RandomPairwiseExchange::metric_strategy;
  using ProtoEnumType = lbann_data::RandomPairwiseExchange::MetricStrategy;
  switch (strategy) {
  case ProtoEnumType::RandomPairwiseExchange_MetricStrategy_LOWER_IS_BETTER:
    return LBANNEnumType::LOWER_IS_BETTER;
  case ProtoEnumType::RandomPairwiseExchange_MetricStrategy_HIGHER_IS_BETTER:
    return LBANNEnumType::HIGHER_IS_BETTER;
  default:
    LBANN_ERROR("Unknown enum value: ", static_cast<int>(strategy));
  }
  return LBANNEnumType::LOWER_IS_BETTER;
}

ExchangeStrategyFactory build_default_exchange_factory()
{
  ExchangeStrategyFactory factory;
  factory.register_builder("CheckpointBinary", make_checkpoint_binary);
  factory.register_builder("CheckpointFile", make_checkpoint_file);
  factory.register_builder("SendRecvWeights", make_sendrecv_weights);
  return factory;
}

ExchangeStrategyFactory& get_exchange_factory()
{
  static ExchangeStrategyFactory factory = build_default_exchange_factory();
  return factory;
}

} // namespace

template <>
std::unique_ptr<lbann::ltfb::RandomPairwiseExchange::ExchangeStrategy>
lbann::make_abstract<lbann::ltfb::RandomPairwiseExchange::ExchangeStrategy>(
  const google::protobuf::Message& msg)
{
  using ProtoStrategy = lbann_data::RandomPairwiseExchange::ExchangeStrategy;
  auto const& params = dynamic_cast<ProtoStrategy const&>(msg);

  std::set<std::string> weights_names;
  auto const num_weights = params.weights_name_size();
  for (int id = 0; id < num_weights; ++id)
    weights_names.emplace(params.weights_name(id));

  auto const& exchange_params =
    proto::helpers::get_oneof_message(params, "strategy");
  return get_exchange_factory().create_object(
    proto::helpers::message_type(exchange_params),
    std::move(weights_names),
    exchange_params);
}

template <>
std::unique_ptr<lbann::ltfb::RandomPairwiseExchange>
lbann::make<lbann::ltfb::RandomPairwiseExchange>(
  google::protobuf::Message const& msg_in)
{
  auto const& params = dynamic_cast<google::protobuf::Any const&>(msg_in);
  lbann_data::RandomPairwiseExchange msg;
  LBANN_ASSERT(params.UnpackTo(&msg));

  using ExchangeStrategyType =
    lbann::ltfb::RandomPairwiseExchange::ExchangeStrategy;
  return make_unique<lbann::ltfb::RandomPairwiseExchange>(
    msg.metric_name(),
    to_lbann(msg.metric_strategy()),
    make_abstract<ExchangeStrategyType>(msg.exchange_strategy()));
}
