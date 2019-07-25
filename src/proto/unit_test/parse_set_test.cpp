#include <catch2/catch.hpp>

#include <lbann/base.hpp>
#include <lbann/proto/proto_common.hpp>

#include <string>
#include <set>

TEST_CASE("Testing parse_set", "[proto][utilities]")
{
  SECTION("execution_mode")
  {
    const std::set<lbann::execution_mode> expected =
      { lbann::execution_mode::training,
        lbann::execution_mode::validation,
        lbann::execution_mode::testing };

    auto const answer =
      lbann::parse_set<lbann::execution_mode>("train validate train test test");
    CHECK(answer == expected);
    CHECK(
      lbann::parse_set<lbann::execution_mode>("")
      == std::set<lbann::execution_mode>{});
    CHECK(
      lbann::parse_set<lbann::execution_mode>(" ")
      == std::set<lbann::execution_mode>{});
  }

  SECTION("std::string")
  {
    const std::set<std::string> expected = { "this", "is", "a", "test" };
    auto const answer =
      lbann::parse_set<std::string>("this is a test");
    CHECK(answer == expected);
    CHECK(lbann::parse_set<std::string>("") == std::set<std::string>{});
  }

  SECTION("int")
  {
    const std::set<int> expected = { 1, 2, 3, 4, 5 };
    auto const answer =
      lbann::parse_set<int>("1 1 2 1 3 4 3 3 5 2");
    CHECK(answer == expected);
    CHECK(lbann::parse_set<int>("") == std::set<int>{});
    CHECK(lbann::parse_set<int>(" ") == std::set<int>{});
  }
}
