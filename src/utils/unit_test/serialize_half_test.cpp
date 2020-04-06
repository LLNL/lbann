#include <catch2/catch.hpp>

#include <lbann/base.hpp> // half stuff is here.
#include <lbann/utils/serialization.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

#include <lbann/utils/h2_tmp.hpp>

using namespace h2::meta;
using BinaryArchiveTypes = TL<cereal::BinaryOutputArchive,
                              cereal::BinaryInputArchive>;
using JSONArchiveTypes = TL<cereal::JSONOutputArchive,
                            cereal::JSONInputArchive>;
using XMLArchiveTypes = TL<cereal::XMLOutputArchive,
                           cereal::XMLInputArchive>;
TEMPLATE_TEST_CASE("Serialization of half types",
                   "[utilities][half][serialize]",
                   BinaryArchiveTypes,
                   JSONArchiveTypes,
                   XMLArchiveTypes)
{
  using ArchiveTypes = TestType;
  using OutputArchiveT = tlist::Car<ArchiveTypes>; // First entry
  using InputArchiveT = tlist::Cadr<ArchiveTypes>; // Second entry

  std::stringstream ss;
  lbann::cpu_fp16 val(4.13f), val_restore(0.f);

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
