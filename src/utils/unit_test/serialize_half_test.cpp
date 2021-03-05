#include <catch2/catch.hpp>

#include <lbann/base.hpp> // half stuff is here.
#include <lbann/utils/serialize.hpp>

#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#include <sstream>

using namespace h2::meta;

using Fp16Types = TL<lbann::cpu_fp16
#ifdef LBANN_HAS_GPU_FP16
                     , lbann::fp16
#endif // LBANN_HAS_GPU_FP16
                     >;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
template <typename T>
using BinaryArchiveTypeBundle = TL<T,
                                   cereal::BinaryOutputArchive,
                                   cereal::BinaryInputArchive>;
using BinaryArchiveTypes = tlist::ExpandTL<BinaryArchiveTypeBundle, Fp16Types>;
#else
using BinaryArchiveTypes = tlist::Empty;
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
template <typename T>
using XMLArchiveTypeBundle = TL<T,
                                cereal::XMLOutputArchive,
                                cereal::XMLInputArchive>;
using XMLArchiveTypes = tlist::ExpandTL<XMLArchiveTypeBundle, Fp16Types>;
#else
using XMLArchiveTypes = tlist::Empty;
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

using AllArchiveTypes = tlist::Append<BinaryArchiveTypes,
                                      XMLArchiveTypes>;

TEMPLATE_LIST_TEST_CASE(
  "Serialization of half types",
  "[utilities][half][serialize]",
  AllArchiveTypes)
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
