// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/colorize.hpp>
#include "helper.hpp"

TEST_CASE("Testing colorize preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    identity(mat.template get<uint8_t>(), 3, 3, 1);
    std::vector<size_t> dims = {1, 3, 3};
    auto gs = lbann::transform::colorize();

    SECTION("applying grayscape") {
      REQUIRE_NOTHROW(gs.apply(mat, dims));

      SECTION("colorize changes dims correctly") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("colorize does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("colorize does not change values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 3,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (row == col) { REQUIRE(x == 1); }
            else { REQUIRE(x == 0); }
          });
      }
    }
  }

  SECTION("matrix with three channels") {
    identity(mat.template get<uint8_t>(), 3, 3, 3);
    std::vector<size_t> dims = {3, 3, 3};
    auto gs = lbann::transform::colorize();

    SECTION("applying colorize") {
      REQUIRE_NOTHROW(gs.apply(mat, dims));

      SECTION("colorize does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("colorize does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("colorize produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 3,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (row == col) { REQUIRE(x == 1); }
            else { REQUIRE(x == 0); }
          });
      }
    }
  }
}
