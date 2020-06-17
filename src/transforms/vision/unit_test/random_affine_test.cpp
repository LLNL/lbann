// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/random_affine.hpp>
#include "helper.hpp"

// Note: This is *random* so we only do basic checks.
TEST_CASE("Testing random affine preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
  // For simplicity, we'll only use a 3-channel matrix here.
  identity(mat.template get<uint8_t>(), 10, 10, 3);
  std::vector<size_t> dims = {3, 10, 10};

  SECTION("rotation") {
    auto affiner = lbann::transform::random_affine(0.0, 90.0, 0, 0, 0, 0, 0, 0);

    SECTION("applying the transform") {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("translate") {
    auto affiner = lbann::transform::random_affine(0, 0, 0.1, 0.1, 0, 0, 0, 0);

    SECTION("applying the transform") {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("scale") {
    auto affiner = lbann::transform::random_affine(0, 0, 0, 0, 0.0, 2.0, 0, 0);

    SECTION("applying the transform") {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("shear") {
    auto affiner = lbann::transform::random_affine(0, 0, 0, 0, 0, 0, 0.0, 45.0);

    SECTION("applying the transform") {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("all") {
    auto affiner = lbann::transform::random_affine(
      0.0, 90.0, 0.1, 0.1, 0.0, 2.0, 0.0, 45.0);

    SECTION("applying the transform") {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }
}
