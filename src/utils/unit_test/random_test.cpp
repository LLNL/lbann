// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/random.hpp>

constexpr size_t num_tests = 1000;

TEST_CASE("Testing fast_random_uniform", "[random][utilities]") {
  SECTION("32-bit generator") {
    std::mt19937 gen;
    SECTION("floats") {
      for (size_t i = 0; i < num_tests; ++i) {
        float val = lbann::fast_random_uniform<float>(gen);
        REQUIRE(val >= 0.0f);
        REQUIRE(val < 1.0f);
      }
    }

    SECTION("doubles") {
      for (size_t i = 0; i < num_tests; ++i) {
        double val = lbann::fast_random_uniform<double>(gen);
        REQUIRE(val >= 0.0);
        REQUIRE(val < 1.0);
      }
    }
  }
  SECTION("64-bit generator") {
    std::mt19937_64 gen;
    SECTION("floats") {
      for (size_t i = 0; i < num_tests; ++i) {
        float val = lbann::fast_random_uniform<float>(gen);
        REQUIRE(val >= 0.0f);
        REQUIRE(val < 1.0f);
      }
    }

    SECTION("doubles") {
      for (size_t i = 0; i < num_tests; ++i) {
        double val = lbann::fast_random_uniform<double>(gen);
        REQUIRE(val >= 0.0);
        REQUIRE(val < 1.0);
      }
    }
  }
}
