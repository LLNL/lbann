// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/random_crop.hpp>
#include "helper.hpp"

TEST_CASE("Testing random crop preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    ones(mat.template get<uint8_t>(), 5, 5, 1);
    std::vector<size_t> dims = {1, 5, 5};
    auto cropper = lbann::transform::random_crop(3, 3);

    SECTION("applying the crop") {
      REQUIRE_NOTHROW(cropper.apply(mat, dims));

      SECTION("cropping changes dims correctly") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("cropping does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("cropping produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 1,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            REQUIRE(x == 1);
          });
      }
    }
  }

  SECTION("matrix with three channels") {
    ones(mat.template get<uint8_t>(), 5, 5, 3);
    std::vector<size_t> dims = {3, 5, 5};
    auto cropper = lbann::transform::random_crop(3, 3);

    SECTION("applying the crop") {
      REQUIRE_NOTHROW(cropper.apply(mat, dims));

      SECTION("cropping changes dims correctly") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("cropping does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("cropping produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 3,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            REQUIRE(x == 1);
          });
      }
    }
  }
}
