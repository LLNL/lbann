// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/sample_normalize.hpp>

TEST_CASE("Testing sample normalize preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(lbann::CPUMat());
  El::Identity(mat.template get<lbann::DataType>(), 3, 3);
  El::Scale(2.0, mat.template get<lbann::DataType>());
  std::vector<size_t> dims = {3, 3};
  auto normalizer = lbann::transform::sample_normalize();
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
          if (row == col) {
            REQUIRE(real_mat(row, col) == Approx(1.41421356));
          } else {
            REQUIRE(real_mat(row, col) == Approx(-0.70710678));
          }
        }
      }
    }
  }
}
