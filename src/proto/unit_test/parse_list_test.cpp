#include <catch2/catch.hpp>

#include <lbann/base.hpp>
#include <lbann/proto/proto_common.hpp>

#include <string>
#include <vector>

TEST_CASE("Testing parse_list", "[proto][utilities]")
{
  SECTION("execution_mode")
  {
    const std::vector<lbann::execution_mode> expected =
      { lbann::execution_mode::training,
        lbann::execution_mode::validation,
        lbann::execution_mode::testing };

    auto const answer =
      lbann::parse_list<lbann::execution_mode>("train validate test");
    CHECK(answer == expected);
    CHECK(
      lbann::parse_list<lbann::execution_mode>("")
      == std::vector<lbann::execution_mode>{});
    CHECK(
      lbann::parse_list<lbann::execution_mode>(" ")
      == std::vector<lbann::execution_mode>{});

    CHECK_THROWS(
      lbann::parse_list<lbann::execution_mode>("banana tuna salad"));
  }

  SECTION("std::string")
  {
    const std::vector<std::string> expected = { "this", "is", "a", "test" };
    auto const answer =
      lbann::parse_list<std::string>("this is a test");
    CHECK(answer == expected);
    CHECK(
      lbann::parse_list<std::string>("") == std::vector<std::string>{});

  }

  SECTION("int")
  {
    const std::vector<int> expected = { 1, 2, 3, 4, 5 };
    auto const answer =
      lbann::parse_list<int>("1 2 3 4 5");
    CHECK(answer == expected);
    CHECK(lbann::parse_list<int>("") == std::vector<int>{});
    CHECK(lbann::parse_list<int>(" ") == std::vector<int>{});
  }
}
