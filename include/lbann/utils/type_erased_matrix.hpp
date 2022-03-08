////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED
#define LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED

#include <lbann/utils/memory.hpp>

#include <El.hpp>

#include <any>

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
   *  @throws std::bad_any_cast If the datatype of the held matrix does not
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
   *  @throws std::bad_any_cast If the datatype of the held matrix does not
   *      match the input @c Field.
   */
  template <typename Field>
  El::Matrix<Field> const& get() const
  {
    return std::any_cast<El::Matrix<Field> const&>(m_matrix);
  }

  /** @brief Replace the held matrix with a new one constructed
   *      in-place from the arguments.
   *
   *  @tparam Field The data type of the newly held matrix
   *
   *  @param args The arguments with which to construct the new matrix.
   *
   *  @return Reference to the new underlying matrix
   */
  template <typename Field, typename... Args>
  El::Matrix<Field>& emplace(Args&&... args)
  {
    return m_matrix.emplace<El::Matrix<Field>>(std::forward<Args>(args)...);
  }

private:
  /** @brief Type-erased matrix storage */
  std::any m_matrix;
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
  return std::make_unique<type_erased_matrix>(El::Matrix<Field>{});
}

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_TYPE_ERASED_MATRIX_HPP_INCLUDED
