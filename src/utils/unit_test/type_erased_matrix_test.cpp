// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/type_erased_matrix.hpp>

// Other includes
#include <El.hpp>

namespace
{
template <typename SrcT, typename TgtT>
struct TypePair
{
  using src_type = SrcT;
  using tgt_type = TgtT;
};
}// namespace <Anon>

TEMPLATE_PRODUCT_TEST_CASE(
  "Testing type-erase Matrix","[type-erase][la][utilities]",
  (TypePair),
  ((int, float), (int, double),
   (float, int), (float, double),
   (double, int), (double,float)))
{
  using src_type = typename TestType::src_type;
  using tgt_type = typename TestType::tgt_type;

  GIVEN("A type-erased matrix")
  {
    auto x = lbann::utils::create_type_erased_matrix<src_type>();

    THEN ("the internal matrix has the correct storage type")
    {
      REQUIRE_NOTHROW(x->template get<src_type>());
      REQUIRE_THROWS_AS(x->template get<tgt_type>(),
                        lbann::utils::bad_any_cast);

      auto&& internal_mat = x->template get<src_type>();
      REQUIRE(internal_mat.Height() == 0);
      REQUIRE(internal_mat.Width() == 0);
    }

    WHEN ("The matrix is resized")
    {
      REQUIRE_NOTHROW(x->template get<src_type>().Resize(10,12));

      THEN ("The change is reflected in the internal matrix.")
      {
        auto&& internal_mat = x->template get<src_type>();
        REQUIRE(internal_mat.Height() == 10);
        REQUIRE(internal_mat.Width() == 12);
      }
      AND_WHEN ("The matrix is converted")
      {
        REQUIRE_NOTHROW(x->template convert<src_type, tgt_type>());

        THEN ("The internal matrix has the right type and size")
        {
          REQUIRE_NOTHROW(x->template get<tgt_type>());
          REQUIRE_THROWS_AS(x->template get<src_type>(),
                            lbann::utils::bad_any_cast);

          REQUIRE(x->template get<tgt_type>().Height() == 10);
          REQUIRE(x->template get<tgt_type>().Width() == 12);
        }
      }
    }
  }
}
