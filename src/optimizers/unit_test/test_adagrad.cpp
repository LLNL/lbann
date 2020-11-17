#include <catch2/catch.hpp>
#include <lbann/optimizers/adagrad.hpp>

#include "optimizer_common.hpp"

#include <sstream>

// See test_sgd.cpp for a detailed, annotated test case.

namespace
{

template <typename TensorDataType>
struct AdagradBuilder
{
  static lbann::adagrad<TensorDataType> Stateful()
  {
    return lbann::adagrad<TensorDataType>(
      /*learning_rate=*/TensorDataType(1.f),
      /*eps=*/TensorDataType(2.f));
  }

  static lbann::adagrad<TensorDataType> Default() {
    return lbann::adagrad<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*eps=*/TensorDataType(0.0f));
  }
};// struct AdagradBuilder

template <typename DataType, typename ArchiveTypes>
struct TestAdagrad : TestOptimizer<lbann::adagrad<DataType>,
                                   AdagradBuilder<DataType>,
                                   ArchiveTypes>
{};

}// namespace <anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Optimizer serialization",
  "[optimizer][serialize]",
  TestAdagrad,
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
