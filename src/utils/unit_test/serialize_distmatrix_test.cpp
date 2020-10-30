#include <catch2/catch.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>

#include <El.hpp>

TEST_CASE("DistMatrix serialization",
          "[inprogress][serialize][utils][distmatrix][mpi]")
{
  using DistMatType =
    El::DistMatrix<float, El::MC, El::MR, El::ELEMENT, El::Device::CPU>;

  std::stringstream ss;
  DistMatType mat(12,16), mat_restore;

  SECTION("JSON archive")
  {
    {
      cereal::JSONOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));
    }

    std::cout << ss.str() << std::endl;

    {
      cereal::JSONInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
  }

  SECTION("Binary archive")
  {
    El::MakeUniform(mat);

    {
      cereal::BinaryOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));

    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    REQUIRE(mat.Height() == mat_restore.Height());
    REQUIRE(mat.Width() == mat_restore.Width());
    for (El::Int col = 0; col < mat.LocalWidth(); ++col)
    {
      for (El::Int row = 0; row < mat.LocalHeight(); ++row)
      {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat.GetLocal(row, col) == mat_restore.GetLocal(row, col));
      }
    }
  }

}
