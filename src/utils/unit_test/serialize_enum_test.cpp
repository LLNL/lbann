#include <catch2/catch.hpp>

#include "TestHelpers.hpp"
#include "MPITestHelpers.hpp"

#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

namespace
{
enum class myenum
{
  DEFAULT,
  NOT_THE_DEFAULT,
};
}

TEST_CASE("Serializing enums", "[serialize][utils][enum]")
{
  auto& comm = ::unit_test::utilities::current_world_comm();
  auto const& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;
  myenum a = myenum::NOT_THE_DEFAULT;
  myenum b = myenum::DEFAULT;

  SECTION("XML Archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
  SECTION("Binary Archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
  SECTION("Rooted Binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }

}
