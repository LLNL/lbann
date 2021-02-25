#include <catch2/catch.hpp>
#include <lbann/optimizers/hypergradient_adam.hpp>

#include "optimizer_common.hpp"

#include <sstream>

// See test_sgd.cpp for a detailed, annotated test case.

namespace
{

template <typename TensorDataType>
struct HypergradientAdamBuilder
{
  static lbann::hypergradient_adam<TensorDataType> Stateful() {
    return lbann::hypergradient_adam<TensorDataType>(
      /*init_learning_rate=*/TensorDataType(0.0123f),
      /*hyper_learning_rate=*/TensorDataType(0.0321f),
      /*beta1=*/TensorDataType(1.234f),
      /*beta2=*/TensorDataType(4.321f),
      /*eps=*/TensorDataType(0.0234f));
  }

  static lbann::hypergradient_adam<TensorDataType> Default() {
    return lbann::hypergradient_adam<TensorDataType>(
      /*init_learning_rate=*/TensorDataType(0.0f),
      /*hyper_learning_rate=*/TensorDataType(0.0f),
      /*beta1=*/TensorDataType(0.0f),
      /*beta2=*/TensorDataType(0.0f),
      /*eps=*/TensorDataType(0.0f));
  }
};// struct Hypergradient_AdamBuilder

}// namespace <anon>

TEMPLATE_LIST_TEST_CASE(
  "Hypergradient Adam Optimizer serialization",
  "[optimizer][serialize]",
  AllArchiveTypes)
{
  using ValueType = tlist::Car<TestType>;

  using ArchiveTypes = tlist::Cdr<TestType>;
  using OutputArchiveType = tlist::Car<ArchiveTypes>;
  using InputArchiveType = tlist::Cadr<ArchiveTypes>;

  using OptimizerType = lbann::hypergradient_adam<ValueType>;
  using BuilderType = HypergradientAdamBuilder<ValueType>;

  std::stringstream ss;

  OptimizerType opt = BuilderType::Stateful();
  OptimizerType opt_restore = BuilderType::Default();

  // Verify that the optimizers differ in the first place.
  CHECK_FALSE(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK_FALSE(desc_string(opt) == desc_string(opt_restore));

  {
    OutputArchiveType oarchive(ss);
    CHECK_NOTHROW(oarchive(opt));
  }

  {
    InputArchiveType iarchive(ss);
    CHECK_NOTHROW(iarchive(opt_restore));
  }

  CHECK(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK(desc_string(opt) == desc_string(opt_restore));
}
