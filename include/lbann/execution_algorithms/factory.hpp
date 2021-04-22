#ifndef LBANN_EXECUTION_ALGORITHMS_FACTORY_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_FACTORY_HPP_INCLUDED

#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/factory_error_policies.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <h2/meta/typelist/TypeList.hpp>

#include <google/protobuf/message.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace lbann {

/** @brief Factory for constructing training algorithms from protobuf
 *         messages.
 */
using TrainingAlgorithmFactory = generic_factory<
  training_algorithm,
  std::string,
  proto::generate_builder_type<training_algorithm,
                               google::protobuf::Message const&>>;

/** @brief The builder type used to create a new training algorithm.
 */
using TrainingAlgorithmBuilder =
  typename TrainingAlgorithmFactory::builder_type;

/** @brief The trainining algorithm factory key. */
using TrainingAlgorithmKey = typename TrainingAlgorithmFactory::id_type;

/** @brief Register a new training algorithm with the default factory.
 *  @param[in] key The identifier for the training algorithm.
 *  @param[in] builder The builder for the training algorithm.
 */
void register_new_training_algorithm(TrainingAlgorithmKey key,
                                     TrainingAlgorithmBuilder builder);

} // namespace lbann

/** @brief Get the factory for a given key from the default factory
 *         factory.
 *  @param[in] key The identifier for the training algorithm.
 *  @return The abstract factory that can build components of the
 *          requested training algorithm.
 */
template <>
std::unique_ptr<lbann::training_algorithm>
lbann::make_abstract<lbann::training_algorithm>(
  google::protobuf::Message const& params);

#endif // LBANN_EXECUTION_ALGORITHMS_FACTORY_HPP_INCLUDED
