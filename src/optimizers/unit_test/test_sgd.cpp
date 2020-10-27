// Must include this for all the Catch2 machinery
#include <catch2/catch.hpp>

// Some common infrastructure for testing optimizers
#include "optimizer_common.hpp"

// Classes under test
#include <lbann/optimizers/sgd.hpp>

// Needed for doing diskless testing.
#include <sstream>

// This test file is highly annotated and is meant to serve as the
// exemplar of the serialization tests. Subsequent testing will be
// more compact.

namespace
{

// We need a way to construct objects. One must at least pretend
// to have meaningful state, and the other must have some "default"
// state (really, it just has to be different from the "meaningful
// state" so that we can detect a change in the restored value).
//
// We accomplish this using little builder functor templates to handle
// the TensorDataType. Since there are no optimzer classes with common
// construction arguments, we do this on an optimizer-specific basis.
//
// If it would be better to save/load from
// "unique/shared-pointer-to-base" instead, these functors could
// easily be modified to accommodate that instead.

template <typename TensorDataType>
struct SGDBuilder
{
  static lbann::sgd<TensorDataType> Stateful() {
    return lbann::sgd<TensorDataType>(
      /*learning_rate=*/TensorDataType(2.f),
      /*momentum=*/TensorDataType(3.f),
      /*nesterov=*/true);
  }

  static lbann::sgd<TensorDataType> Default() {
    return lbann::sgd<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*momentum=*/TensorDataType(0.0f),
      /*nesterov=*/false);
  }
};// struct SGDBuilder

// Now that we can build optimizer objects, we need a way to compare
// two such objects. For this example, we just compare the metadata,
// and name our overloaded function suitably. This could be a simple
// template, were operator==(OptT const&, OptT const&) defined for
// every optimizer.

template <typename TensorDataType>
bool CompareMetadata(
  lbann::sgd<TensorDataType> const& original,
  lbann::sgd<TensorDataType> const& restored)
{
  // Compare objects
  int original_nesterov = original.using_nesterov();
  int restored_nesterov = restored.using_nesterov();

  return ((original.get_learning_rate() == restored.get_learning_rate())
          && (original.get_momentum() == restored.get_momentum())
          && (original_nesterov == restored_nesterov));
}

template <typename DataType, typename ArchiveTypes>
struct TestSGD : TestOptimizer<lbann::sgd<DataType>,
                               SGDBuilder<DataType>,
                               ArchiveTypes>
{};

}// namespace <anon>

// By way of documentation, this is what all the metaprogramming and
// "common" code have been building up to. The driving goal was to
// make things look "nice" and be meaningful to developers on the
// command line. The result is that the test case names will be
// something like:
//
//   "Optimizer serialization - TestSGD<float, XMLArchiveTypes>"
//
// This gives the developer three important bits of information: the
// optimizer class under test, the TensorDataType being used, and the
// Cereal archive type being used. This doesn't really matter when the
// test is passing, but it will be very useful specificity when/if it
// fails. It is also useful for narrowing down the list of tests to
// run by regexp match, which is supported in both CTest and the
// Catch2 CLI.
//
// The test case itself is implemented as though it were a template
// function, taking a subclass of TestOptimizer<...> as its template
// parameter. All of the data for the test is generated internally to
// the test, and the archive is written to a string stream, so the
// test is also disk-free. The test asserts that serializing and
// deserializing the archive is exception-free and that the
// deserialized optimizer object matches the serialized optimizer
// object, for whatever definition of "matching" the test implementor
// has decided upon.

TEMPLATE_PRODUCT_TEST_CASE(
  "SGD Optimizer serialization",
  "[optimizer][serialize]",
  TestSGD,
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
