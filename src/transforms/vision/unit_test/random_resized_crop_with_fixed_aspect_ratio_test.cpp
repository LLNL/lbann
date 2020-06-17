// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/vision/random_resized_crop_with_fixed_aspect_ratio.hpp>
#include <lbann/transforms/vision/resize.hpp>
#include <lbann/transforms/vision/random_crop.hpp>
#include "helper.hpp"

TEST_CASE("Testing random resized crop with fixed aspect ratio preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel") {
    ones(mat.template get<uint8_t>(), 5, 5, 1);
    std::vector<size_t> dims = {1, 5, 5};

    SECTION("resizing larger and cropping") {
      auto resize_cropper = lbann::transform::random_resized_crop_with_fixed_aspect_ratio(7, 7, 3, 3);

      SECTION("applying the resize/crop") {
        REQUIRE_NOTHROW(resize_cropper.apply(mat, dims));

        SECTION("resizing/cropping changes dims correctly") {
          REQUIRE(dims[0] == 1);
          REQUIRE(dims[1] == 3);
          REQUIRE(dims[2] == 3);
        }
        SECTION("resizing/cropping does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing/cropping produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 3, 3, 1,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }

        SECTION("compare with resize then crop") {
          lbann::utils::type_erased_matrix mat2 =
            lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
          ones(mat2.template get<uint8_t>(), 5, 5, 1);
          std::vector<size_t> dims2 = {1, 5, 5};
          auto resizer = lbann::transform::resize(7, 7);
          auto cropper = lbann::transform::random_crop(3, 3);
          REQUIRE_NOTHROW(resizer.apply(mat2, dims2));
          REQUIRE_NOTHROW(cropper.apply(mat2, dims2));
          REQUIRE(dims == dims2);
          const uint8_t* buf = mat.template get<uint8_t>().LockedBuffer();
          const uint8_t* buf2 = mat2.template get<uint8_t>().LockedBuffer();
          for (size_t i = 0; i < dims2[1]*dims2[2]; ++i) {
            REQUIRE(buf[i] == buf2[i]);
          }
        }
      }
    }
    SECTION("resizing smaller and cropping") {
      auto resize_cropper = lbann::transform::random_resized_crop_with_fixed_aspect_ratio(3, 3, 1, 1);

      SECTION("applying the resize/crop") {
        REQUIRE_NOTHROW(resize_cropper.apply(mat, dims));

        SECTION("resizing/cropping changes dims correctly") {
          REQUIRE(dims[0] == 1);
          REQUIRE(dims[1] == 1);
          REQUIRE(dims[2] == 1);
        }
        SECTION("resizing/cropping does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing/cropping produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 1, 1, 1,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }

        SECTION("compare with resize then crop") {
          lbann::utils::type_erased_matrix mat2 =
            lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
          ones(mat2.template get<uint8_t>(), 5, 5, 1);
          std::vector<size_t> dims2 = {1, 5, 5};
          auto resizer = lbann::transform::resize(3, 3);
          auto cropper = lbann::transform::random_crop(1, 1);
          REQUIRE_NOTHROW(resizer.apply(mat2, dims2));
          REQUIRE_NOTHROW(cropper.apply(mat2, dims2));
          REQUIRE(dims == dims2);
          const uint8_t* buf = mat.template get<uint8_t>().LockedBuffer();
          const uint8_t* buf2 = mat2.template get<uint8_t>().LockedBuffer();
          for (size_t i = 0; i < dims2[1]*dims2[2]; ++i) {
            REQUIRE(buf[i] == buf2[i]);
          }
        }
      }
    }
  }

  SECTION("matrix with three channels") {
    ones(mat.template get<uint8_t>(), 5, 5, 3);
    std::vector<size_t> dims = {3, 5, 5};

    SECTION("resizing larger and cropping") {
      auto resize_cropper = lbann::transform::random_resized_crop_with_fixed_aspect_ratio(7, 7, 3, 3);

      SECTION("applying the resize/crop") {
        REQUIRE_NOTHROW(resize_cropper.apply(mat, dims));

        SECTION("resizing/cropping changes dims correctly") {
          REQUIRE(dims[0] == 3);
          REQUIRE(dims[1] == 3);
          REQUIRE(dims[2] == 3);
        }
        SECTION("resizing/cropping does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing/cropping produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 3, 3, 3,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }

        SECTION("compare with resize then crop") {
          lbann::utils::type_erased_matrix mat2 =
            lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
          ones(mat2.template get<uint8_t>(), 5, 5, 3);
          std::vector<size_t> dims2 = {3, 5, 5};
          auto resizer = lbann::transform::resize(7, 7);
          auto cropper = lbann::transform::random_crop(3, 3);
          REQUIRE_NOTHROW(resizer.apply(mat2, dims2));
          REQUIRE_NOTHROW(cropper.apply(mat2, dims2));
          REQUIRE(dims == dims2);
          const uint8_t* buf = mat.template get<uint8_t>().LockedBuffer();
          const uint8_t* buf2 = mat2.template get<uint8_t>().LockedBuffer();
          for (size_t i = 0; i < dims2[1]*dims2[2]; ++i) {
            REQUIRE(buf[i] == buf2[i]);
          }
        }
      }
    }
    SECTION("resizing smaller and cropping") {
      auto resize_cropper = lbann::transform::random_resized_crop_with_fixed_aspect_ratio(3, 3, 1, 1);

      SECTION("applying the resize/crop") {
        REQUIRE_NOTHROW(resize_cropper.apply(mat, dims));

        SECTION("resizing/cropping changes dims correctly") {
          REQUIRE(dims[0] == 3);
          REQUIRE(dims[1] == 1);
          REQUIRE(dims[2] == 1);
        }
        SECTION("resizing/cropping does not change matrix type") {
          REQUIRE_NOTHROW(mat.template get<uint8_t>());
        }
        SECTION("resizing/cropping produces correct values") {
          auto& real_mat = mat.template get<uint8_t>();
          apply_elementwise(
            real_mat, 1, 1, 3,
            [](uint8_t& x, El::Int row, El::Int col, El::Int) {
              REQUIRE(x == 1);
            });
        }

        SECTION("compare with resize then crop") {
          lbann::utils::type_erased_matrix mat2 =
            lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
          ones(mat2.template get<uint8_t>(), 5, 5, 3);
          std::vector<size_t> dims2 = {3, 5, 5};
          auto resizer = lbann::transform::resize(3, 3);
          auto cropper = lbann::transform::random_crop(1, 1);
          REQUIRE_NOTHROW(resizer.apply(mat2, dims2));
          REQUIRE_NOTHROW(cropper.apply(mat2, dims2));
          REQUIRE(dims == dims2);
          const uint8_t* buf = mat.template get<uint8_t>().LockedBuffer();
          const uint8_t* buf2 = mat2.template get<uint8_t>().LockedBuffer();
          for (size_t i = 0; i < dims2[1]*dims2[2]; ++i) {
            REQUIRE(buf[i] == buf2[i]);
          }
        }
      }
    }
  }
}
