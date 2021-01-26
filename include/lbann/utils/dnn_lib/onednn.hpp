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

#ifndef LBANN_UTILS_DNN_LIB_ONEDNN_HPP
#define LBANN_UTILS_DNN_LIB_ONEDNN_HPP

#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/h2_tmp.hpp"

#ifdef LBANN_HAS_ONEDNN

#include <oneapi/dnnl/dnnl.hpp>

// Error utility macros
#define CHECK_ONEDNN(onednn_call)                               \
  do {                                                          \
    try {                                                       \
      (onednn_call);                                            \
    }                                                           \
    catch (::dnnl::error const& e)                              \
    {                                                           \
      LBANN_ERROR("Detected oneDNN error. e.what(): \n\n",      \
                  e.what());                                    \
    }                                                           \
  } while (0)

#define CHECK_ONEDNN_DTOR(onednn_call)                                  \
  try {                                                                 \
    (onednn_call);                                                      \
  }                                                                     \
  catch (std::exception const& e) {                                     \
    std::cerr << "Caught exception:\n\n    what(): "                    \
              << e.what() << "\n\nCalling std::terminate() now."        \
              <<  std::endl;                                            \
    std::terminate();                                                   \
  }                                                                     \
  catch (...) {                                                         \
    std::cerr << "Caught something that isn't an std::exception.\n\n"   \
              << "Calling std::terminate() now." << std::endl;          \
    std::terminate();                                                   \
  }


namespace lbann {

// Forward declaration
class Layer;

namespace onednn {
namespace details {

/** @class TypeMapT
 *  @brief Map C++ types to OneDNN enum values.
 */
template <typename T>
struct TypeMapT;

/** @class IsSupportedTypeT
 *  @brief Predicate indicating if a type is supported by oneDNN.
 */
template <typename T>
struct IsSupportedTypeT : std::false_type {};

#define ADD_ONEDNN_TYPE_MAP(CPPTYPE, EVAL)                      \
  template <>                                                   \
  struct TypeMapT<CPPTYPE>                                      \
    : std::integral_constant<dnnl::memory::data_type,           \
                             dnnl::memory::data_type::EVAL>     \
  {};                                                           \
  template <>                                                   \
  struct IsSupportedTypeT<CPPTYPE> : std::true_type {}

// Basic types
ADD_ONEDNN_TYPE_MAP(  float, f32);
ADD_ONEDNN_TYPE_MAP(int32_t, s32);
ADD_ONEDNN_TYPE_MAP( int8_t,  s8);
ADD_ONEDNN_TYPE_MAP(uint8_t,  u8);

// 16-bit floating point
// TODO: bfloat16 types (not yet supported in Hydrogen)
#if defined LBANN_HAS_HALF
ADD_ONEDNN_TYPE_MAP(cpu_fp16, f16);
#endif
#if defined LBANN_HAS_GPU_FP16
ADD_ONEDNN_TYPE_MAP(fp16, f16);
#endif

}// namespace details

template <typename T>
using TypeMap = typename details::TypeMapT<T>;

template <typename T>
inline constexpr auto DataTypeValue = TypeMap<T>::value;

template <typename T>
inline constexpr bool IsSupportedType = details::IsSupportedTypeT<T>::value;

template <typename T, ::h2::meta::EnableWhen<IsSupportedType<T>, int> = 0>
inline constexpr dnnl::memory::data_type get_data_type()
{
  return DataTypeValue<T>;
}

template <typename T, ::h2::meta::EnableUnless<IsSupportedType<T>, int> = 0>
inline dnnl::memory::data_type get_data_type()
{
  LBANN_ERROR("Type \"", El::TypeName<T>(), "\" is not supported "
              "by the oneDNN runtime.");
}

} // namespace onednn
} // namespace lbann
#endif // LBANN_HAS_ONEDNN
#endif // LBANN_UTILS_DNN_LIB_ONEDNN_HPP
