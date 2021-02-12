#include <catch2/catch.hpp>

#include <lbann/base.hpp> // half stuff is here.
#include <lbann/utils/serialize.hpp>

#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#include <sstream>

using namespace h2::meta;

// (NOTE trb 04/06/2020): There seems to be an issue with Catch2 where
// this *must* be a parameter pack. This only appears to be true for
// templated type aliases, not actual classes. I haven't looked into
// it, but I don't care that much since this works around well.

template <typename... ValueType>
using BinaryArchiveTypes = TL<ValueType...,
                              cereal::BinaryOutputArchive,
                              cereal::BinaryInputArchive>;
template <typename... ValueType>
using XMLArchiveTypes = TL<ValueType...,
                           cereal::XMLOutputArchive,
                           cereal::XMLInputArchive>;

// This is not really elegant, but preprocessing macros inside
// preprocessor blocks is "undefined behavior" so we duplicate the
// whole thing.
#ifdef LBANN_HAS_GPU_FP16
TEMPLATE_PRODUCT_TEST_CASE(
  "Serialization of half types",
  "[utilities][half][serialize]",
  (BinaryArchiveTypes, XMLArchiveTypes),
  (lbann::cpu_fp16, lbann::fp16))
#else
TEMPLATE_PRODUCT_TEST_CASE(
  "Serialization of half types",
  "[utilities][half][serialize]",
  (BinaryArchiveTypes, XMLArchiveTypes),
  (lbann::cpu_fp16))
#endif
{
  using ValueType = tlist::Car<TestType>;
  using ArchiveTypes = tlist::Cdr<TestType>;
  using OutputArchiveT = tlist::Car<ArchiveTypes>; // First entry
  using InputArchiveT = tlist::Cadr<ArchiveTypes>; // Second entry

  std::stringstream ss;
  ValueType val(1.23f), val_restore(0.f);

  // Save
  {
    OutputArchiveT oarchive(ss);

    CHECK_NOTHROW(oarchive(val));
  }

  // Restore
  {
    InputArchiveT iarchive(ss);
    CHECK_NOTHROW(iarchive(val_restore));
  }

  CHECK(val == val_restore);
}
