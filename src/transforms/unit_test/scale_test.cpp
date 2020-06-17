// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/transforms/scale.hpp>

TEST_CASE("Testing scale preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(lbann::CPUMat());
  El::Ones(mat.template get<lbann::DataType>(), 3, 3);
  std::vector<size_t> dims = {3, 3};
  auto scaler = lbann::transform::scale(2.0);

  SECTION("applying the scaler") {
    REQUIRE_NOTHROW(scaler.apply(mat, dims));

    SECTION("scaling does not change dims") {
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 3);
    }
    SECTION("scaling does not change matrix type") {
      REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
    }
    SECTION("scaling changes matrix values") {
      auto& real_mat = mat.template get<lbann::DataType>();
      for (El::Int col = 0; col < 3; ++col) {
        for (El::Int row = 0; row < 3; ++row) {
          REQUIRE(real_mat(row, col) == 2.0);
        }
      }
    }
  }
}
