// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/resize.hpp>
#include "helper.hpp"

TEST_CASE("Testing resize preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    ones(mat.template get<uint8_t>(), 3, 3, 1);
    std::vector<size_t> dims = {1, 3, 3};

    SECTION("resizing larger") {
      auto resizer = lbann::transform::resize(5, 5);

      SECTION("applying the resize") {
        REQUIRE_NOTHROW(resizer.apply(mat, dims));

        SECTION("resizing changes dims correctly") {
          REQUIRE(dims[0] == 1);
          REQUIRE(dims[1] == 5);
          REQUIRE(dims[2] == 5);
        }
        SECTION("resizing does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 5, 5, 1,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }
      }
    }
    SECTION("resizing smaller") {
      auto resizer = lbann::transform::resize(2, 2);

      SECTION("applying the resize") {
        REQUIRE_NOTHROW(resizer.apply(mat, dims));

        SECTION("resizing changes dims correctly") {
          REQUIRE(dims[0] == 1);
          REQUIRE(dims[1] == 2);
          REQUIRE(dims[2] == 2);
        }
        SECTION("resizing does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 2, 2, 1,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }
      }
    }
  }

  SECTION("matrix with three channels") {
    ones(mat.template get<uint8_t>(), 3, 3, 3);
    std::vector<size_t> dims = {3, 3, 3};

    SECTION("resizing larger") {
      auto resizer = lbann::transform::resize(5, 5);

      SECTION("applying the resize") {
        REQUIRE_NOTHROW(resizer.apply(mat, dims));

        SECTION("resizing changes dims correctly") {
          REQUIRE(dims[0] == 3);
          REQUIRE(dims[1] == 5);
          REQUIRE(dims[2] == 5);
        }
        SECTION("resizing does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 5, 5, 3,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }
      }
    }
    SECTION("resizing smaller") {
      auto resizer = lbann::transform::resize(2, 2);

      SECTION("applying the resize") {
        REQUIRE_NOTHROW(resizer.apply(mat, dims));

        SECTION("resizing changes dims correctly") {
          REQUIRE(dims[0] == 3);
          REQUIRE(dims[1] == 2);
          REQUIRE(dims[2] == 2);
        }
        SECTION("resizing does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 2, 2, 3,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }
      }
    }
  }
}
