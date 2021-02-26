#include <catch2/catch.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

using MatrixTypes =
  h2::meta::TL<
  El::Matrix<float, El::Device::CPU>
#ifdef LBANN_HAS_GPU
  , El::Matrix<float, El::Device::GPU>
#endif
  >;

TEMPLATE_LIST_TEST_CASE("Matrix serialization",
                        "[serialize][utils][matrix][seq]",
                        MatrixTypes)
{
  using MatrixType = TestType;
  std::stringstream ss;
  MatrixType mat(13,17), mat_restore;

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));
    }

    {
      cereal::XMLInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

  // The binary test case is slightly different since we serialize the
  // data, too. In this case, the buffer has to have real data.
#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
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
        CHECK(mat.Get(row, col) == mat_restore.Get(row, col));
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
        CHECK(mat_noncontig.Get(row, col) == mat_restore.Get(row, col));
      }
    }

  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

  SECTION("Views are not serializable")
  {
    auto mat_view = El::View(mat);
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
    {
      cereal::XMLOutputArchive oarchive(ss);
      CHECK_THROWS(oarchive(mat_view));
    }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
    {
      cereal::BinaryOutputArchive oarchive(ss);
      CHECK_THROWS(oarchive(mat_view));
    }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
  }
}

using check_valid_ptr = bool;

TEMPLATE_LIST_TEST_CASE("Matrix smart-pointer-to-concrete serialization",
                        "[serialize][utils][matrix][seq][smartptr]",
                        MatrixTypes)
{
  using MatrixType = TestType;

  std::stringstream ss;
  std::unique_ptr<MatrixType> mat, mat_restore;
  mat = std::make_unique<MatrixType>(16, 12);

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML Archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      CHECK_NOTHROW(oarchive(mat));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    REQUIRE((check_valid_ptr) mat_restore);
    CHECK(mat->Height() == mat_restore->Height());
    CHECK(mat->Width() == mat_restore->Width());
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
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
        CHECK(mat->Get(row, col) == mat_restore->Get(row, col));
      }
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
}
