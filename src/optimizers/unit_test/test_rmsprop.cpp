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

template <typename DataType, typename ArchiveTypes>
struct TestRmsprop : TestOptimizer<lbann::rmsprop<DataType>,
                                RmspropBuilder<DataType>,
                                ArchiveTypes>
{};

}// namespace <anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Optimizer serialization",
  "[optimizer][serialize]",
  TestRmsprop,
  TEMPLATE_ARG_LIST)
{
  using TypePack = TestType;
  using OptimizerType = GetOptimizerType<TypePack>;
  using BuilderType = GetBuilderType<TypePack>;
  using OutputArchiveType = GetOutputArchiveType<TypePack>;
  using InputArchiveType = GetInputArchiveType<TypePack>;

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
