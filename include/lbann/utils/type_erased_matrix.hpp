#ifndef LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED
#define LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED

#include <lbann/utils/any.hpp>
#include <lbann/utils/memory.hpp>

#include <El.hpp>

namespace lbann
{
namespace utils
{

/** @class type_erased_matrix
 *  @brief A type-erased wrapper around an @c El::Matrix<T,Device::CPU>
 *
 *  @warning This class is an implementation detail of the
 *      preprocessing pipeline and should not be used in general
 *      LBANN code.
 */
class type_erased_matrix
{
public:

  /** @brief Construct from a copy of a given matrix.
   *
   *  Deep-copy the input matrix into the held matrix.
   *
   *  @tparam Field The data type of the input matrix
   *
   *  @param in_matrix The input matrix.
   *
   *  @warning This performs a deep copy of the matrix.
   */
  template <typename Field>
  type_erased_matrix(El::Matrix<Field> const& in_matrix)
  {
    El::Matrix<Field> held;
    El::Copy(in_matrix, held);
    m_matrix.emplace<El::Matrix<Field>>(std::move(held));
  }

  /** @brief Construct by moving the given matrix into type-erased
   *      storage.
   *
   *  Move the input matrix into the held matrix.
   *
   *  @tparam Field The data type of the input matrix
   *
   *  @param in_matrix The input matrix.
   */
  template <typename Field>
  type_erased_matrix(El::Matrix<Field>&& in_matrix)
    : m_matrix{std::move(in_matrix)}
  {}

  /** @brief Access the underlying matrix.
   *
   *  Provides read/write access to the underlying matrix if the input
   *  @c Field matches the data type of the held matrix.
   *
   *  @tparam Field The data type of the held matrix
   *
   *  @throws bad_any_cast If the datatype of the held matrix does not
   *      match the input @c Field.
   */
  template <typename Field>
  El::Matrix<Field>& get()
  {
    return const_cast<El::Matrix<Field>&>(
        static_cast<type_erased_matrix const&>(*this)
        .template get<Field>());
  }

  /** @brief Access the underlying matrix.
   *
   *  Provides read-only access to the underlying matrix if the input
   *  @c Field matches the data type of the held matrix.
   *
   *  @tparam Field The data type of the held matrix
   *
   *  @return Reference to the underlying matrix
   *
   *  @throws bad_any_cast If the datatype of the held matrix does not
   *      match the input @c Field.
   */
  template <typename Field>
  El::Matrix<Field> const& get() const
  {
    return any_cast<El::Matrix<Field> const&>(m_matrix);
  }

  /** @brief Access the underlying matrix.
   *
   *  Converts (copies) the internal matrix into a matrix of a new
   *  type, which is then held in place of the original matrix.
   *
   *  @tparam OldField The data type of the originally held matrix
   *  @tparam NewField The data type of the newly held matrix
   *
   *  @return @c const reference to the underlying matrix
   *
   *  @throws bad_any_cast If the datatype of the held matrix does not
   *      match the input @c OldField.
   */
  template <typename OldField, typename NewField>
  El::Matrix<NewField>& convert()
  {
    El::Matrix<NewField> new_mat;
    El::Copy(this->template get<OldField>(), new_mat);
    m_matrix.template emplace<El::Matrix<NewField>>(std::move(new_mat));
    return this->template get<NewField>();
  }

private:
  /** @brief Type-erased matrix storage */
  any m_matrix;
};// class type_erased_matrix

/** @brief Create an empty type-erased matrix with given underlying
 *      data type.
 *
 *  @tparam Field The type of the underlying matrix.
 *
 *  @return A pointer to an empty type-erased matrix with data type @c
 *      Field.
 */
template <typename Field>
std::unique_ptr<type_erased_matrix>
create_type_erased_matrix()
{
  return make_unique<type_erased_matrix>(El::Matrix<Field>{});
}

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED
