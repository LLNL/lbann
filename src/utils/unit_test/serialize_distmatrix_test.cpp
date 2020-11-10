#include <catch2/catch.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/h2_tmp.hpp>
#include <lbann/utils/serialize.hpp>

#include "MPITestHelpers.hpp"

// Enumerate all DistMatrix types. Start by getting all the
// distributions.
template <typename T, El::Device D>
using DistMatrixTypesWithDevice = h2::meta::TL<
  // There is currently a known bug where copying
  // (CIRC,CIRC,CPU)->(CIRC,CIRC,GPU) results in an infinite recursion
  // in Hydrogen. Since we don't actually use this in our code, ignore
  // this case for now.
  //
  // El::DistMatrix<T, El::CIRC, El::CIRC, El::ELEMENT, D>,
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

template <typename T>
using DistMatrixTypes =
#if defined LBANN_HAS_GPU
  h2::meta::tlist::Append<
  DistMatrixTypesWithDevice<T, El::Device::CPU>,
  DistMatrixTypesWithDevice<T, El::Device::GPU>>;
#else
  DistMatrixTypesWithDevice<T, El::Device::CPU>;
#endif // defined LBANN_HAS_GPU

// Finally, enumerate all data types.
using AllDistMatrixTypes = h2::meta::tlist::Append<
  DistMatrixTypes<float>,
  DistMatrixTypes<double>>;

TEMPLATE_LIST_TEST_CASE("DistMatrix serialization",
                        "[serialize][utils][distmatrix][mpi]",
                        AllDistMatrixTypes)
{
  using DistMatType = TestType;

  // Setup the grid stack
  auto& comm = ::unit_test::utilities::current_world_comm();
  lbann::utils::grid_manager mgr(comm.get_trainer_grid());

  std::stringstream ss;
  DistMatType mat(12,16, lbann::utils::get_current_grid()),
    mat_restore(lbann::utils::get_current_grid());

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

  // Setup the grid stack
  auto& comm = ::unit_test::utilities::current_world_comm();
  lbann::utils::grid_manager mgr(comm.get_trainer_grid());

  std::stringstream ss;
  std::unique_ptr<AbsDistMatType> mat, mat_restore;
  mat = std::make_unique<DistMatType>(12, 16, lbann::utils::get_current_grid());

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

    REQUIRE((check_valid_ptr) mat_restore);
    CHECK(mat->Height() == mat_restore->Height());
    CHECK(mat->Width() == mat_restore->Width());
  }

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
