// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/to_lbann_layout.hpp>
#include "helper.hpp"

TEST_CASE("Testing to LBANN layout", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    zeros(mat.template get<uint8_t>(), 3, 3, 1);
    apply_elementwise(mat.template get<uint8_t>(), 3, 3, 1,
                      [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                        if (row == 0) { x = 1; }
                      });
    std::vector<size_t> dims = {1, 3, 3};
    auto tll = lbann::transform::to_lbann_layout();

    SECTION("converting the matrix") {
      REQUIRE_NOTHROW(tll.apply(mat, dims));

      SECTION("converting does not change dims") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("converting changes matrix type") {
        REQUIRE_THROWS(mat.template get<uint8_t>());
        REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
      }
      SECTION("converting produces correct values") {
        auto& real_mat = mat.template get<lbann::DataType>();
        const lbann::DataType* buf = real_mat.LockedBuffer();
        for (size_t col = 0; col < 3; ++col) {
          for (size_t row = 0; row < 3; ++row) {
            const lbann::DataType val = buf[row + col*3];
            if (row == 0) { REQUIRE(val == 1.0f / 255.0f); }
            else { REQUIRE(val == 0.0f); }
          }
        }
      }
    }
  }

  SECTION("matrix with three channels") {
    zeros(mat.template get<uint8_t>(), 3, 3, 3);
    apply_elementwise(mat.template get<uint8_t>(), 3, 3, 3,
                      [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                        if (row == 0) { x = 1; }
                      });
    std::vector<size_t> dims = {3, 3, 3};
    auto tll = lbann::transform::to_lbann_layout();

    SECTION("converting the matrix") {
      REQUIRE_NOTHROW(tll.apply(mat, dims));

      SECTION("converting does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("converting changes matrix type") {
        REQUIRE_THROWS(mat.template get<uint8_t>());
        REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
      }
      SECTION("converting produces correct values") {
        auto& real_mat = mat.template get<lbann::DataType>();
        const lbann::DataType* buf = real_mat.LockedBuffer();
        for (size_t channel = 0; channel < 3; ++channel) {
          for (size_t col = 0; col < 3; ++col) {
            for (size_t row = 0; row < 3; ++row) {
              const lbann::DataType val = buf[3*3*channel + row + col*3];
              if (row == 0) { REQUIRE(val == 1.0f / 255.0f); }
              else { REQUIRE(val == 0.0f); }
            }
          }
        }
      }
    }
  }
}
