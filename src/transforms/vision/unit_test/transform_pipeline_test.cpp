// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/transform_pipeline.hpp>
#include <lbann/transforms/vision/resized_center_crop.hpp>
#include <lbann/transforms/vision/to_lbann_layout.hpp>
#include <lbann/transforms/scale.hpp>
#include <lbann/transforms/normalize.hpp>
#include <lbann/utils/memory.hpp>
#include "helper.hpp"

TEST_CASE("Testing vision transform pipeline", "[preproc]") {
  lbann::transform::transform_pipeline p;
  p.add_transform(
    lbann::make_unique<lbann::transform::resized_center_crop>(7, 7, 3, 3));
  p.add_transform(lbann::make_unique<lbann::transform::to_lbann_layout>());
  p.add_transform(lbann::make_unique<lbann::transform::scale>(2.0f));
  p.add_transform(lbann::make_unique<lbann::transform::normalize>(
                    std::vector<float>({0.5f, 0.5f, 0.5f}),
                    std::vector<float>({2.0f, 2.0f, 2.0f})));
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
  ones(mat.template get<uint8_t>(), 5, 5, 3);
  std::vector<size_t> dims = {3, 5, 5};

  SECTION("applying the pipeline") {
    REQUIRE_NOTHROW(p.apply(mat, dims));

    SECTION("pipeline produces correct dims") {
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 3);
      REQUIRE(dims[2] == 3);
    }
    SECTION("pipeline produces correct type") {
      REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
    }
    SECTION("pipeline produces correct values") {
      auto& real_mat = mat.template get<lbann::DataType>();
      const lbann::DataType* buf = real_mat.LockedBuffer();
      for (size_t i = 0; i < 3*3*3; ++i) {
        REQUIRE(buf[i] == Approx(-0.24607843));
      }
    }
  }
}
