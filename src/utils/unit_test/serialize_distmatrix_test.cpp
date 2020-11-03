#include <catch2/catch.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>

#include <El.hpp>

// Enumerate all DistMatrix types. Start by getting all the
// distributions.
template <typename T, El::Device D>
using DistMatrixTypesWithDevice = h2::meta::TL<
  El::DistMatrix<T, El::CIRC, El::CIRC, El::ELEMENT, D>,
  El::DistMatrix<T, El::MC  , El::MR  , El::ELEMENT, D>,
  El::DistMatrix<T, El::MC  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MD  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MR  , El::MC  , El::ELEMENT, D>,
  El::DistMatrix<T, El::MR  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MC  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MD  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MR  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::VC  , El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::VR  , El::ELEMENT, D>,
  El::DistMatrix<T, El::VC  , El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::VR  , El::STAR, El::ELEMENT, D>>;

// Now get all the devices. For now, just do CPU testing.
template <typename T>
using DistMatrixTypes = DistMatrixTypesWithDevice<T, El::Device::CPU>;

// Finally, enumerate all data types.
using AllDistMatrixTypes = h2::meta::tlist::Append<
  DistMatrixTypes<float>,
  DistMatrixTypes<double>>;

TEMPLATE_LIST_TEST_CASE("DistMatrix serialization",
                        "[serialize][utils][distmatrix][mpi]",
                        AllDistMatrixTypes)
{
  using DistMatType = TestType;

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

// Just a bit of sugar to make the output clearer when testing for
// null pointers.
using check_valid_ptr = bool;

TEMPLATE_LIST_TEST_CASE(
  "DistMatrix serialization with smart pointers",
  "[serialize][utils][distmatrix][mpi][smartptr]",
  AllDistMatrixTypes)
{
  using DistMatType = TestType;
  using AbsDistMatType = typename TestType::absType;

  std::stringstream ss;
  std::unique_ptr<AbsDistMatType> mat, mat_restore;
  mat = std::make_unique<DistMatType>(12, 16);

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

    REQUIRE(mat->Height() == mat_restore->Height());
    REQUIRE(mat->Width() == mat_restore->Width());
    for (El::Int col = 0; col < mat->LocalWidth(); ++col)
    {
      for (El::Int row = 0; row < mat->LocalHeight(); ++row)
      {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat->GetLocal(row, col) == mat_restore->GetLocal(row, col));
      }
    }
  }
}
