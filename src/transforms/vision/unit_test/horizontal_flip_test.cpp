// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/horizontal_flip.hpp>
#include "helper.hpp"

TEST_CASE("Testing horizontal flip preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    zeros(mat.template get<uint8_t>(), 3, 3, 1);
    apply_elementwise(mat.template get<uint8_t>(), 3, 3, 1,
                      [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                        if (col == 0) { x = 1; }
                      });
    std::vector<size_t> dims = {1, 3, 3};
    auto flipper = lbann::transform::horizontal_flip(1.0);

    SECTION("applying the flip") {
      REQUIRE_NOTHROW(flipper.apply(mat, dims));

      SECTION("flipping does not change dims") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("flipping does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("flipping produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 1,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (col == 2) {
              REQUIRE(x == 1);
            } else {
              REQUIRE(x == 0);
            }
          });
      }
    }
  }

  SECTION("matrix with three channels") {
    zeros(mat.template get<uint8_t>(), 3, 3, 3);
    apply_elementwise(mat.template get<uint8_t>(), 3, 3, 3,
                      [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                        if (col == 0) { x = 1; }
                      });
    std::vector<size_t> dims = {3, 3, 3};
    auto flipper = lbann::transform::horizontal_flip(1.0);

    SECTION("applying the flip") {
      REQUIRE_NOTHROW(flipper.apply(mat, dims));

      SECTION("flipping does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("flipping does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("flipping produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 3,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (col == 2) {
              REQUIRE(x == 1);
            } else {
              REQUIRE(x == 0);
            }
          });
      }
    }
  }
}
