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
    lbann::adam<TensorDataType> ret(
      /*learning_rate=*/TensorDataType(3.f),
      /*beta1=*/TensorDataType(1.f),
      /*beta2=*/TensorDataType(4.f),
      /*eps=*/TensorDataType(2.f));

    // These probably shouldn't be set here, but let's pretend
    // something's happened to perturb the state.
    ret.set_current_beta1(TensorDataType(5.f));
    ret.set_current_beta2(TensorDataType(6.f));
    return ret;
  }

  static lbann::adam<TensorDataType> Default() {
    return lbann::adam<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*beta1=*/TensorDataType(0.0f),
      /*beta2=*/TensorDataType(0.0f),
      /*eps=*/TensorDataType(0.0f));
  }
};// struct AdamBuilder

template <typename DataType, typename ArchiveTypes>
struct TestAdam : TestOptimizer<lbann::adam<DataType>,
                                AdamBuilder<DataType>,
                                ArchiveTypes>
{};

}// namespace <anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Optimizer serialization",
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
  CHECK_FALSE(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK_FALSE(opt.get_beta1() == opt_restore.get_beta1());
  CHECK_FALSE(opt.get_beta2() == opt_restore.get_beta2());
  CHECK_FALSE(opt.get_current_beta1() == opt_restore.get_current_beta1());
  CHECK_FALSE(opt.get_current_beta2() == opt_restore.get_current_beta2());
  CHECK_FALSE(opt.get_eps() == opt_restore.get_eps());

  {
    OutputArchiveType oarchive(ss);
    CHECK_NOTHROW(oarchive(opt));
  }

  {
    InputArchiveType iarchive(ss);
    CHECK_NOTHROW(iarchive(opt_restore));
  }

  CHECK(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK(opt.get_beta1() == opt_restore.get_beta1());
  CHECK(opt.get_beta2() == opt_restore.get_beta2());
  CHECK(opt.get_current_beta1() == opt_restore.get_current_beta1());
  CHECK(opt.get_current_beta2() == opt_restore.get_current_beta2());
  CHECK(opt.get_eps() == opt_restore.get_eps());
}
