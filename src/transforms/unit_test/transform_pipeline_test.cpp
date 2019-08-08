// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/transform_pipeline.hpp>
#include <lbann/transforms/scale.hpp>
#include <lbann/transforms/sample_normalize.hpp>
#include <lbann/utils/memory.hpp>

TEST_CASE("Testing transform pipeline", "[preproc]") {
  lbann::transform::transform_pipeline p;
  p.add_transform(lbann::make_unique<lbann::transform::scale>(2.0f));
  p.add_transform(lbann::make_unique<lbann::transform::sample_normalize>());
  lbann::CPUMat mat;
  El::Identity(mat, 3, 3);
  std::vector<size_t> dims = {3, 3};

  SECTION("applying the pipeline") {
    REQUIRE_NOTHROW(p.apply(mat, dims));

    SECTION("pipeline does not change dims") {
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 3);
    }

    SECTION("pipeline produces correct values") {
      for (El::Int col = 0; col < 3; ++col) {
        for (El::Int row = 0; row < 3; ++row) {
          if (row == col) {
            REQUIRE(mat(row, col) == Approx(1.41421356));
          } else {
            REQUIRE(mat(row, col) == Approx(-0.70710678));
          }
        }
      }
    }
  }
}
