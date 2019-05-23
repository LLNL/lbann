////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef _LBANN_CNPY_UTILS_HPP_
#define _LBANN_CNPY_UTILS_HPP_

#include "cnpy.h"
#include <string>
#include <vector>
#include "lbann/utils/exception.hpp"

namespace lbann {
namespace cnpy_utils {

/**
 * Return the offset to the element (in terms of the number of elements from
 * the beginning of the array) of a loaded numpy array na specified by indices
 * If the number of indices is less than the dimension of na array, the indices
 * vector is appended with zeros to match the dimension.
 */
size_t compute_cnpy_array_offset(const cnpy::NpyArray& na, std::vector<size_t> indices);

/**
 * If the type T of the numpy array element is something larger than 1 byte in
 * size, the word_size of the numpy array must be the same as sizeof(T).
 * In such a case, offset to add to a type T pointer is computed as the number
 * of elements up to the position pointed by the indices.
 * If sizeof(T) is 1 byte, then the array may be of char string, or the pointer
 * may be cast to a byte-long type. In such a case, the offset is computed as
 * the number of elements scaled by the word_size of the array.
 * cnpy treats an array of strings as a 1D array, for which the word_size is
 * equal to the length of the largest string.
 */
template<typename T>
inline size_t ptr_offset(const cnpy::NpyArray& na, std::vector<size_t> indices) {
  if ((sizeof(T) != na.word_size) && (sizeof(T) != 1u)) {
    throw lbann_exception(std::string("cnpy_utils::ptr_offset() :") +
           "The data type is not consistent with the word size of the array.");
  }
  return (compute_cnpy_array_offset(na, indices)
           * ((sizeof(T) == 1u)? na.word_size : 1u));
}


/**
 * Allow the access to the data element identified by the indices and the
 * word_size of the array na, but present it as a type T element at the address.
 */
template<typename T>
inline T& data(const cnpy::NpyArray& na, const std::vector<size_t> indices) {
  return *(reinterpret_cast<T*>(&(* na.data_holder)[0]) + ptr_offset<T>(na, indices));
}


/**
 * Return the address of the data element identified by the indices and the
 * word_size of the array na, but present it as the address of a type T element
 */
template<typename T>
inline T* data_ptr(const cnpy::NpyArray& na, const std::vector<size_t> indices) {
  return (reinterpret_cast<T*>(&(* na.data_holder)[0]) + ptr_offset<T>(na, indices));
}

template<>
inline void* data_ptr<void>(const cnpy::NpyArray& na, const std::vector<size_t> indices) {
  return data_ptr<uint8_t>(na, indices);
}


/**
 * Shrink the first dimension of cnpy::NpyArray to the given size.
 * This is used to choose only first sz samples in data.
 */
void shrink_to_fit(cnpy::NpyArray& na, size_t sz);


/// Show the dimensions of loaded data
std::string show_shape(const cnpy::NpyArray& na);

} // end of namespace cnpy_utils
} // end of namespace lbann

#endif // _LBANN_CNPY_UTILS_HPP_
