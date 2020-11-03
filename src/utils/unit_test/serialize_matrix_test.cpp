#include <catch2/catch.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>
#include <El.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/polymorphic.hpp>

#include <lbann/utils/exception.hpp>
#include <stdexcept>

TEST_CASE("Matrix serialization",
          "[serialize][utils][matrix][seq]")
{
  std::stringstream ss;
  El::Matrix<float, El::Device::CPU> mat(5,7), mat_restore;

  SECTION("JSON archive")
  {
    {
      cereal::JSONOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));
    }

    {
      cereal::JSONInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
  }

  // The binary test case is slightly different since we serialize the
  // data, too. In this case, the buffer has to have real data.
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

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
    for (El::Int col = 0; col < mat.Width(); ++col)
    {
      for (El::Int row = 0; row < mat.Height(); ++row)
      {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat.CRef(row, col) == mat_restore.CRef(row, col));
      }
    }
  }

  SECTION("Binary archive, noncontiguous")
  {
    El::Matrix<float, El::Device::CPU> mat_noncontig(5, 7, 12);
    El::MakeUniform(mat_noncontig);

    {
      cereal::BinaryOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat_noncontig));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat_noncontig.Height() == mat_restore.Height());
    CHECK(mat_noncontig.Width() == mat_restore.Width());
    for (El::Int col = 0; col < mat_noncontig.Width(); ++col)
    {
      for (El::Int row = 0; row < mat_noncontig.Height(); ++row)
      {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat_noncontig.CRef(row, col) == mat_restore.CRef(row, col));
      }
    }

  }

  SECTION("Views are not serializable")
  {
    auto mat_view = El::View(mat);
    {
      cereal::JSONOutputArchive oarchive(ss);
      CHECK_THROWS(oarchive(mat_view));
    }

    {
      cereal::BinaryOutputArchive oarchive(ss);
      CHECK_THROWS(oarchive(mat_view));
    }
  }
}

using check_valid_ptr = bool;

TEST_CASE("Matrix smart-pointer-to-concrete serialization",
          "[serialize][utils][matrix][seq][smartptr]")
{
  using MatrixType = El::Matrix<float, El::Device::CPU>;

  std::stringstream ss;
  std::unique_ptr<MatrixType> mat, mat_restore;
  mat = std::make_unique<MatrixType>(16, 12);

  SECTION("JSON Archive")
  {
    {
      cereal::JSONOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));
    }
    {
      cereal::JSONInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    REQUIRE((check_valid_ptr) mat_restore);
    CHECK(mat->Height() == mat_restore->Height());
    CHECK(mat->Width() == mat_restore->Width());
  }

  SECTION("Binary archive")
  {
    El::MakeUniform(*mat);

    {
      cereal::BinaryOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat->Height() == mat_restore->Height());
    CHECK(mat->Width() == mat_restore->Width());
    for (El::Int col = 0; col < mat->Width(); ++col)
    {
      for (El::Int row = 0; row < mat->Height(); ++row)
      {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat->CRef(row, col) == mat_restore->CRef(row, col));
      }
    }
  }

}
