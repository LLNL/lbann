#include <catch2/catch.hpp>
#include <lbann/optimizers/rmsprop.hpp>

#include "optimizer_common.hpp"

#include <sstream>

// See test_sgd.cpp for a detailed, annotated test case.

namespace
{

template <typename TensorDataType>
struct RmspropBuilder
{
  static lbann::rmsprop<TensorDataType> Stateful() {
    return lbann::rmsprop<TensorDataType>(
      /*learning_rate=*/TensorDataType(1.f),
      /*decay_rate=*/TensorDataType(3.f),
      /*eps=*/TensorDataType(2.f));
  }

  static lbann::rmsprop<TensorDataType> Default() {
    return lbann::rmsprop<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*decay_rate=*/TensorDataType(0.0f),
      /*eps=*/TensorDataType(0.0f));
  }
};// struct RmspropBuilder

}// namespace <anon>

TEMPLATE_LIST_TEST_CASE(
  "RMSProp Optimizer serialization",
  "[optimizer][serialize]",
  AllArchiveTypes)
{
  using ValueType = tlist::Car<TestType>;

  using ArchiveTypes = tlist::Cdr<TestType>;
  using OutputArchiveType = tlist::Car<ArchiveTypes>;
  using InputArchiveType = tlist::Cadr<ArchiveTypes>;

  using OptimizerType = lbann::rmsprop<ValueType>;
  using BuilderType = RmspropBuilder<ValueType>;

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
