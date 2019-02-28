#ifndef LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED
#define LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED

#include <lbann/utils/any.hpp>
#include <lbann/utils/memory.hpp>

#include <El.hpp>

namespace lbann
{
namespace utils
{

class type_erased_matrix
{
public:

  template <typename Field>
  type_erased_matrix(El::Matrix<Field> const& in_matrix)
    : m_matrix{in_matrix}
  {}

  template <typename Field>
  type_erased_matrix(El::Matrix<Field>&& in_matrix)
    : m_matrix{std::move(in_matrix)}
  {}

  template <typename Field>
  El::Matrix<Field>& get()
  {
    return const_cast<El::Matrix<Field>&>(
        static_cast<type_erasted_matrix const&>(*this)
        .template get<Field>());
  }

  template <typename Field>
  El::Matrix<Field> const& get()
  {
    return any_cast<El::Matrix<Field> const&>(m_matrix);
  }

private:
  any m_matrix;
};// class type_erased_matrix

// Helper function for what will probably be the usual construction
// process.
template <typename Field>
std::unique_ptr<type_erased_matrix>
create_type_erased_matrx()
{
  return make_unique<type_erased_matrix>(El::Matrix<Field>{});
}

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED
