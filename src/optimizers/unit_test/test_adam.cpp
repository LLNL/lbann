#include <catch2/catch.hpp>
#include <lbann/optimizers/adam.hpp>

#include "optimizer_common.hpp"

#include <sstream>

// See test_sgd.cpp for a detailed, annotated test case.

namespace
{

template <typename TensorDataType>
struct AdamBuilder
{
  static lbann::adam<TensorDataType> Stateful() {
    return lbann::adam<TensorDataType>(
      /*learning_rate=*/TensorDataType(3.f),
      /*beta1=*/TensorDataType(1.f),
      /*beta2=*/TensorDataType(4.f),
      /*eps=*/TensorDataType(2.f));
  }

  static lbann::adam<TensorDataType> Default() {
    return lbann::adam<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*beta1=*/TensorDataType(0.0f),
      /*beta2=*/TensorDataType(0.0f),
      /*eps=*/TensorDataType(0.0f));
  }
};// struct AdamBuilder

template <typename TensorDataType>
bool CompareMetadata(
  lbann::adam<TensorDataType> const& original,
  lbann::adam<TensorDataType> const& restored)
{
  return ((original.get_learning_rate() == restored.get_learning_rate())
          && (original.get_beta1() == restored.get_beta1())
          && (original.get_beta2() == restored.get_beta2())
          && (original.get_current_beta1() == restored.get_current_beta1())
          && (original.get_current_beta2() == restored.get_current_beta2())
          && (original.get_eps() == restored.get_eps()));
}

template <typename DataType, typename ArchiveTypes>
struct TestAdam : TestOptimizer<lbann::adam<DataType>,
                                AdamBuilder<DataType>,
                                ArchiveTypes>
{};

}// namespace <anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Adam Optimizer serialization",
  "[optimizer][serialize]",
  TestAdam,
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
  CHECK_FALSE(CompareMetadata(opt, opt_restore));

  {
    OutputArchiveType oarchive(ss);
    CHECK_NOTHROW(oarchive(opt));
  }

  {
    InputArchiveType iarchive(ss);
    CHECK_NOTHROW(iarchive(opt_restore));
  }

  CHECK(CompareMetadata(opt, opt_restore));
}
