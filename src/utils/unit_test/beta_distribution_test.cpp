// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/beta.hpp>

#include <cmath>

constexpr size_t num_tests = 1000;

template <typename RealType, typename Generator>
void test_dist(Generator& g, RealType a, RealType b) {
  lbann::beta_distribution<RealType> dist(a, b);
  for (size_t i = 0; i < num_tests; ++i) {
    RealType val = dist(g);
    REQUIRE(std::isfinite(val));
    REQUIRE(val >= RealType(0));
    REQUIRE(val <= RealType(1));
  }
}

TEST_CASE("Testing beta_distribution", "[random][utilities]") {
  std::mt19937 gen;
  SECTION("float") {
    SECTION("a=0.5 b=0.5") {
      test_dist<float>(gen, 0.5f, 0.5f);
    }
    SECTION("a=0.001 b=0.001") {
      test_dist<float>(gen, 0.001f, 0.001f);
    }
    SECTION("a=1.5 b=1.5") {
      test_dist<float>(gen, 1.5f, 1.5f);
    }
  }
  SECTION("double") {
    SECTION("a=0.5 b=0.5") {
      test_dist<double>(gen, 0.5, 0.5);
    }
    SECTION("a=0.001 b=0.001") {
      test_dist<double>(gen, 0.001, 0.001);
    }
    SECTION("a=1.5 b=1.5") {
      test_dist<double>(gen, 1.5, 1.5);
    }
  }
}
