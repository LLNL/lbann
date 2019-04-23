// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/normalize.hpp>

TEST_CASE("Testing normalize preprocessing", "[preproc]") {
  SECTION("matrix with no channels") {
    lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(lbann::CPUMat());
    El::Ones(mat.template get<lbann::DataType>(), 3, 3);
    El::Scale(2.0f, mat.template get<lbann::DataType>());
    std::vector<size_t> dims = {3, 3};
    auto normalizer = lbann::transform::normalize({0.5}, {2.0});
    SECTION("applying the normalizer") {
      REQUIRE_NOTHROW(normalizer.apply(mat, dims));

      SECTION("normalizing does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
      }
      SECTION("normalizing does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
      }
      SECTION("normalizing produces correct values") {
        auto& real_mat = mat.template get<lbann::DataType>();
        for (El::Int col = 0; col < 3; ++col) {
          for (El::Int row = 0; row < 3; ++row) {
            REQUIRE(real_mat(row, col) == Approx(0.75));
          }
        }
      }
    }
  }

  SECTION("matrix with one channel") {
    lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(lbann::CPUMat());
    El::Ones(mat.template get<lbann::DataType>(), 3, 3);
    El::Scale(2.0f, mat.template get<lbann::DataType>());
    std::vector<size_t> dims = {1, 3, 3};
    auto normalizer = lbann::transform::normalize({0.5}, {2.0});
    SECTION("applying the normalizer") {
      REQUIRE_NOTHROW(normalizer.apply(mat, dims));

      SECTION("normalizing does not change dims") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("normalizing does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
      }
      SECTION("normalizing produces correct values") {
        auto& real_mat = mat.template get<lbann::DataType>();
        for (El::Int col = 0; col < 3; ++col) {
          for (El::Int row = 0; row < 3; ++row) {
            REQUIRE(real_mat(row, col) == Approx(0.75));
          }
        }
      }
    }
  }

  SECTION("matrix with three channels") {
    lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(lbann::CPUMat());
    El::Ones(mat.template get<lbann::DataType>(), 27, 1);
    El::Scale(2.0f, mat.template get<lbann::DataType>());
    std::vector<size_t> dims = {3, 3, 3};
    auto normalizer = lbann::transform::normalize({0.75, 0.5, 0.25},
                                                  {1.0, 2.0, 4.0});
    SECTION("applying the normalizer") {
      REQUIRE_NOTHROW(normalizer.apply(mat, dims));

      SECTION("normalizing does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("normalizing does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
      }
      SECTION("normalizing produces correct values") {
        auto& real_mat = mat.template get<lbann::DataType>();
        const lbann::DataType* buf = real_mat.Buffer();
        for (size_t i = 0; i < 9; ++i) {
          REQUIRE(buf[i] == Approx(1.25));
        }
        for (size_t i = 9; i < 18; ++i) {
          REQUIRE(buf[i] == Approx(0.75));
        }
        for (size_t i = 18; i < 27; ++i) {
          REQUIRE(buf[i] == Approx(0.4375));
        }
      }
    }
  }
}
