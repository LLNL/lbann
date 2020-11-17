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

template <typename DataType, typename ArchiveTypes>
struct TestHypergradAdam
  : TestOptimizer<lbann::hypergradient_adam<DataType>,
                  HypergradientAdamBuilder<DataType>,
                  ArchiveTypes>
{};

}// namespace <anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Optimizer serialization",
  "[optimizer][serialize]",
  TestHypergradAdam,
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
