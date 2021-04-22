#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/ltfb.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <google/protobuf/message.h>
#include <memory>
#include <training_algorithm.pb.h>

namespace {

lbann::TrainingAlgorithmFactory build_default_factory()
{
  lbann::TrainingAlgorithmFactory fact;
  fact.register_builder("SGD", lbann::make<lbann::sgd_training_algorithm>);
  fact.register_builder("LTFB", lbann::make<lbann::LTFB>);
  return fact;
}

lbann::TrainingAlgorithmFactory& get_factory()
{
  static lbann::TrainingAlgorithmFactory fact = build_default_factory();
  return fact;
}

} // namespace

void lbann::register_new_training_algorithm(TrainingAlgorithmKey key,
                                            TrainingAlgorithmBuilder builder)
{
  get_factory().register_builder(std::move(key), std::move(builder));
}

template <>
std::unique_ptr<lbann::training_algorithm>
lbann::make_abstract<lbann::training_algorithm>(
  google::protobuf::Message const& params)
{
  auto const& algo_params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(params);
  return get_factory().create_object(
    proto::helpers::message_type(algo_params.parameters()),
    params);
}
