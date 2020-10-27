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

template <typename TensorDataType>
bool CompareMetadata(
  lbann::adagrad<TensorDataType> const& original,
  lbann::adagrad<TensorDataType> const& restored)
{
  std::ostringstream original_desc, restored_desc;
  original_desc << original.get_description();
  restored_desc << restored.get_description();
  return ((original.get_learning_rate() == restored.get_learning_rate())
          && (original_desc.str() == restored_desc.str()));
}

template <typename DataType, typename ArchiveTypes>
struct TestAdagrad : TestOptimizer<lbann::adagrad<DataType>,
                                   AdagradBuilder<DataType>,
                                   ArchiveTypes>
{};

}// namespace <anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Adagrad Optimizer serialization",
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
