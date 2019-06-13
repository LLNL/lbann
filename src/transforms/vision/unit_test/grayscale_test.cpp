// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/grayscale.hpp>
#include "helper.hpp"

TEST_CASE("Testing grayscale preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    identity(mat.template get<uint8_t>(), 3, 3, 1);
    std::vector<size_t> dims = {1, 3, 3};
    auto gs = lbann::transform::grayscale();

    SECTION("applying grayscape") {
      REQUIRE_NOTHROW(gs.apply(mat, dims));

      SECTION("grayscale does not change dims") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("grayscale does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("grayscale does not change values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 1,
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
    auto gs = lbann::transform::grayscale();

    SECTION("applying grayscale") {
      REQUIRE_NOTHROW(gs.apply(mat, dims));

      SECTION("grayscale changes dims correctly") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("grayscale does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("grayscale produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 1,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (row == col) { REQUIRE(x == 1); }
            else { REQUIRE(x == 0); }
          });
      }
    }
  }
}
