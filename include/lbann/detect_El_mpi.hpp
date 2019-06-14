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
//
// detect_El_mpi .hpp .cpp - detect the existence of the instantiations
//               of the overloaded El::mpi wrappers for a data type
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DETECT_EL_MPI_HPP_INCLUDED
#define LBANN_DETECT_EL_MPI_HPP_INCLUDED

#include "base.hpp"
#include <type_traits>

namespace lbann {

template<typename... Ts>
struct make_void {
  using type = void;
};

/// Alternative to c++17 std::void_t for older compilers.
template<typename... Ts>
using void_t = typename make_void<Ts...>::type;


/// By default, assume no instantiation for the type T in El::mpi.
template<typename T, typename = void>
struct is_instantiated_El_mpi_type : std::false_type {};

/**
 * Detect instantiated types in El::mpi. This should match the instantiations
 * of El::mpi::Types<> in hydrogen/src/src/core/mpi_register.cpp.
 */
template<typename T>
struct is_instantiated_El_mpi_type<
         T,
         void_t<typename std::enable_if< std::is_same<T, El::byte>::value ||
                                         std::is_same<T, short>::value ||
                                         std::is_same<T, int>::value ||
                                         std::is_same<T, unsigned>::value ||
                                         std::is_same<T, long int>::value ||
                                         std::is_same<T, unsigned long>::value ||
#ifdef EL_HAVE_MPI_LONG_LONG
                                         std::is_same<T, long long int>::value ||
                                         std::is_same<T, unsigned long long>::value ||
#endif
                                         std::is_same<T, float>::value ||
                                         std::is_same<T, double>::value ||
                                         std::is_same<T, El::Complex<float>>::value ||
                                         std::is_same<T, El::Complex<double>>::value
                                       >::type
               >
       >
  : std::true_type {};


/**
 * Set to use El::byte as type except if the first template argument B is true
 * and the second argument T is non-void, in which case the non-void type is used.
 * The first template argument B should indicate if Elemental has instantiated
 * MPI wrappers for the type T.
 */
template<bool B, class T = void>
struct interpret_as_byte_if_needed {
  using type = El::byte;
};

/// Use type T as is if Elemental has instantiated MPI wrappers for type T.
template<class T>
struct interpret_as_byte_if_needed<true, T> {
  using type = T;
};

/// For void pointers
template<>
struct interpret_as_byte_if_needed<true, void> {
  using type = El::byte;
};

} // namespace lbann

#endif // LBANN_DETECT_EL_MPI_HPP_INCLUDED
