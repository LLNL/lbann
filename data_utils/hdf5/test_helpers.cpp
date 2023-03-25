////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////
#ifdef LBANN_USE_CATCH2_V3
#include <catch2/catch_all.hpp>
#else
#include <catch2/catch.hpp>
#endif // LBANN_USE_CATCH2_V3

#include <conduit/conduit_error.hpp>
#include <regex>

#include "helpers.hpp"

TEST_CASE("Normalizing paths", "[data_utils][path][string]")
{
  CHECK(data_utils::normalize_path("/") == "/");
  CHECK(data_utils::normalize_path(".") == ".");
  CHECK(data_utils::normalize_path("./") == ".");
  CHECK(data_utils::normalize_path("a/b") == "a/b");
  CHECK(data_utils::normalize_path("/a/b") == "/a/b");
  CHECK(data_utils::normalize_path("/a/b/") == "/a/b");
  CHECK(data_utils::normalize_path("/////////////////////////") == "/");
  CHECK(data_utils::normalize_path("a////b") == "a/b");
  CHECK(data_utils::normalize_path("/a//b///c/d////e//////////f////") ==
        "/a/b/c/d/e/f");
}

TEST_CASE("Longest prefix", "[data_utils][string]")
{
  SECTION("emptly list")
  {
    CHECK(data_utils::get_longest_common_prefix({}) == "");
  }

  SECTION("single file, single token")
  {
    CHECK(data_utils::get_longest_common_prefix({"a"}) == ".");
  }

  SECTION("single path, multiple tokens")
  {
    CHECK(data_utils::get_longest_common_prefix({"a/b"}) == "a");
    CHECK(data_utils::get_longest_common_prefix({"a/b/c"}) == "a/b");
    CHECK(data_utils::get_longest_common_prefix({"a/b/c", "a/b/c/d"}) == "a/b");
  }

  SECTION("multiple paths, multiple tokens")
  {
    CHECK(data_utils::get_longest_common_prefix({"a/b/c", "a/b/d"}) == "a/b");
    CHECK(data_utils::get_longest_common_prefix({"a/b/c/d", "a/b/e/f"}) ==
          "a/b");
    CHECK(data_utils::get_longest_common_prefix(
            {"foo/experiment001/a/b/c.whatever",
             "foo/experiment002/a/b/c.whatever",
             "foo/experiment003/a/b/c.whatever"}) == "foo");
    CHECK(data_utils::get_longest_common_prefix({"/a/b/c", "/a/b/d"}) ==
          "/a/b");
    CHECK(data_utils::get_longest_common_prefix({"/a/b/c", "/b/c/d"}) == "/");
  }

  SECTION("multiple paths, multiple tokens, no common prefix")
  {
    CHECK(data_utils::get_longest_common_prefix({"a/b/c/d", "b/c/d/e"}) == ".");
  }

  SECTION("error modes") {
    CHECK_THROWS(data_utils::get_longest_common_prefix({"/a/b/c", "ab/c/d"}));
    CHECK_THROWS(
      data_utils::get_longest_common_prefix({"a/b/c", "/a/b/c"}));
  }
}

using Tokens = std::vector<std::string>;
TEST_CASE("Conduit node extraction", "[data_utils][conduit]")
{
  conduit::Node node;
  node["foo/bar/a"] = 2;
  node["foo/bar/b"] = 7.0;
  node["foo/bar/c/d"] = 12;

  node["foo/baz/a"] = 4;
  node["foo/baz/b"] = 13.0;
  node["foo/baz/c/d"] = 7;

  node["foo/bat/baz/a"] = 88;
  node["foo/bat/baz/b"] = 173.0;
  node["foo/bat/baz/c/d"] = 321;

  auto const& proto_schema =
    data_utils::get_prototype_sample(node, "foo/bar").schema();
  auto const paths = data_utils::get_matching_node_paths(node, proto_schema);
  CHECK(paths == Tokens{"foo/bar", "foo/baz", "foo/bat/baz"});
}

TEST_CASE("Splitting a string", "[data_utils][string]")
{
  SECTION("empty string")
  {
    auto const tokens = data_utils::split("");
    CHECK(tokens.empty());
  }

  SECTION("delimiters only")
  {
    auto const tokens = data_utils::split("////");
    CHECK(tokens.empty());
  }

  SECTION("single token string")
  {
    std::string const input = "bananas";
    auto const tokens = data_utils::split(input);
    Tokens const correct = {"bananas"};
    CHECK(tokens == correct);
  }

  SECTION("two token string")
  {
    std::string const input = "bananas/apples";
    auto const tokens = data_utils::split(input);
    Tokens const correct = {"bananas", "apples"};
    CHECK(tokens == correct);
  }

  SECTION("two token string with leading delimiter")
  {
    std::string const input = "/bananas/apples";
    auto const tokens = data_utils::split(input);
    Tokens const correct = {"bananas", "apples"};
    CHECK(tokens == correct);
  }

  SECTION("two token string with trailing delimiter")
  {
    std::string const input = "/bananas/apples/";
    auto const tokens = data_utils::split(input);
    Tokens const correct = {"bananas", "apples"};
    CHECK(tokens == correct);
  }

  SECTION("two token string with extra delimiters")
  {
    std::string const input = "//bananas////apples//";
    auto const tokens = data_utils::split(input);
    Tokens const correct = {"bananas", "apples"};
    CHECK(tokens == correct);
  }

  SECTION("four token string with extra delimiters")
  {
    std::string const input = "/lemons//bananas///apples////cauliflower/////";
    auto const tokens = data_utils::split(input);
    Tokens const correct = {"lemons", "bananas", "apples", "cauliflower"};
    CHECK(tokens == correct);
  }
}
