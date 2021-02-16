// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/file_utils.hpp>

#include <memory>
#include <numeric>
#include <vector>

TEST_CASE ("Testing \"file_join\" function", "[seq][file][utilities]")
{
  REQUIRE(lbann::file::join_path("a") == "a");
  REQUIRE(lbann::file::join_path("a", "b") == "a/b");
  REQUIRE(lbann::file::join_path("a/", "b") == "a//b");
  REQUIRE(lbann::file::join_path(
            "/a", "b", std::string("c"), "d") == "/a/b/c/d");
}
