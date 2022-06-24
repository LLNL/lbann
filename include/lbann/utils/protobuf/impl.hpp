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
#ifndef LBANN_UTILS_PROTOBUF_IMPL_HPP_INCLUDED
#define LBANN_UTILS_PROTOBUF_IMPL_HPP_INCLUDED

#include "decl.hpp"

#include "lbann/utils/exception.hpp"

#include <google/protobuf/any.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/reflection.h>
#include <h2/meta/core/ValueAsType.hpp>

#include <type_traits>

// Template definitions
namespace lbann {
namespace protobuf {

template <typename T, typename ContainerT>
void assign_to_repeated(google::protobuf::RepeatedField<T>& field,
                        ContainerT const& values)
{
  /** @todo Change to Assign if older versions of protobuf are no
   *  longer supported.
   */
  field.Clear();
  field.Add(begin(values), end(values));
}

template <typename ContainerT>
std::string to_space_sep_string(ContainerT values)
{
  std::string combined;
  for (auto const& value : values)
    combined += (value + " ");

  return combined;
}

namespace details {

template <typename T>
struct PBCppTypePOD;

template <typename T>
struct PBCppTypeS
{
private:
  using FD = google::protobuf::FieldDescriptor;
  using type = std::conditional<
    std::is_base_of_v<google::protobuf::Message, T>,
    h2::meta::ValueAsType<FD::CppType, FD::CPPTYPE_MESSAGE>,
    std::conditional_t<std::is_enum_v<T>,
                       h2::meta::ValueAsType<FD::CppType, FD::CPPTYPE_ENUM>,
                       PBCppTypePOD<T>>>;

public:
  static auto constexpr value = type::type::value;
};

template <typename T>
inline constexpr auto PBCppType = PBCppTypeS<T>::value;

#define ADD_PB_CPPTYPE(TYPE, ENUMTYPE)                                         \
  template <>                                                                  \
  struct PBCppTypePOD<TYPE>                                                    \
  {                                                                            \
    static constexpr auto value =                                              \
      google::protobuf::FieldDescriptor::CPPTYPE_##ENUMTYPE;                   \
  }
ADD_PB_CPPTYPE(int32, INT32);
ADD_PB_CPPTYPE(int64, INT64);
ADD_PB_CPPTYPE(uint32, UINT32);
ADD_PB_CPPTYPE(uint64, UINT64);
ADD_PB_CPPTYPE(double, DOUBLE);
ADD_PB_CPPTYPE(float, FLOAT);
ADD_PB_CPPTYPE(std::string, STRING);
ADD_PB_CPPTYPE(bool, BOOL);
#undef ADD_PB_CPPTYPE

/** @brief Common code to extract a repeated field reference from a
 *         message.
 */
template <typename T>
auto get_repeated_field_ref(google::protobuf::Message const& msg,
                            std::string const& field_name)
{
  auto field_handle = msg.GetDescriptor()->FindFieldByName(field_name);
  if (!field_handle)
    LBANN_ERROR(
      "Field \"",
      field_name,
      "\" in message does not exist or has not been set. Message:\n{\n",
      msg.DebugString(),
      "}\n");
  if (!field_handle->is_repeated())
    LBANN_ERROR(
      "Field \"",
      field_name,
      "\" in message does not refer to a repeated field. Message:\n{\n",
      msg.DebugString(),
      "}\n");
  // The type is valid if it's an exact match. Enums are special,
  // because they can match int32.
  if (!((field_handle->cpp_type() == PBCppType<T>) ||
        ((field_handle->cpp_type() ==
          google::protobuf::FieldDescriptor::CPPTYPE_ENUM) &&
         (PBCppType<T> == google::protobuf::FieldDescriptor::CPPTYPE_INT32))))
    LBANN_ERROR("Field has incompatible type: \"",
                field_handle->cpp_type_name(),
                "\".");
  return msg.GetReflection()->GetRepeatedFieldRef<T>(msg, field_handle);
}

} // namespace details
} // namespace protobuf
} // namespace lbann

template <typename T>
auto lbann::protobuf::as_vector(google::protobuf::Message const& msg,
                                std::string const& field_name) -> std::vector<T>
{
  auto&& field_ref = details::get_repeated_field_ref<T>(msg, field_name);
  return std::vector<T>{field_ref.begin(), field_ref.end()};
}

template <typename T>
auto lbann::protobuf::as_vector(google::protobuf::RepeatedField<T> const& rf)
  -> std::vector<T>
{
  return std::vector<T>{rf.cbegin(), rf.cend()};
}

template <typename T>
auto lbann::protobuf::as_vector(google::protobuf::RepeatedPtrField<T> const& rf)
  -> std::vector<T>
{
  return std::vector<T>{rf.cbegin(), rf.cend()};
}

template <typename OutT, typename InT>
auto lbann::protobuf::to_vector(google::protobuf::RepeatedField<InT> const& rf)
  -> std::vector<OutT>
{
  return std::vector<OutT>{rf.cbegin(), rf.cend()};
}

template <typename T>
auto lbann::protobuf::as_set(google::protobuf::Message const& msg,
                             std::string const& field_name) -> std::set<T>
{
  static_assert(
    !std::is_base_of_v<google::protobuf::Message, T>,
    "Messages are not weakly ordered; they cannot be added to a set.");
  auto&& field_ref = details::get_repeated_field_ref<T>(msg, field_name);
  return std::set<T>{field_ref.begin(), field_ref.end()};
}

template <typename T>
auto lbann::protobuf::as_set(google::protobuf::RepeatedField<T> const& rf)
  -> std::set<T>
{
  return std::set<T>{rf.cbegin(), rf.cend()};
}

inline auto lbann::protobuf::as_set(
  google::protobuf::RepeatedPtrField<std::string> const& rf)
  -> std::set<std::string>
{
  return std::set<std::string>{rf.cbegin(), rf.cend()};
}

template <typename OutT, typename InT>
auto lbann::protobuf::to_set(google::protobuf::RepeatedField<InT> const& rf)
  -> std::set<OutT>
{
  return std::set<OutT>{rf.cbegin(), rf.cend()};
}

template <typename T>
auto lbann::protobuf::as_unordered_set(google::protobuf::Message const& msg,
                                       std::string const& field_name)
  -> std::unordered_set<T>
{
  static_assert(
    !std::is_base_of_v<google::protobuf::Message, T>,
    "Messages are not hashable; they cannot be added to an unordered_set.");
  auto&& field_ref = details::get_repeated_field_ref<T>(msg, field_name);
  return std::unordered_set<T>{field_ref.begin(), field_ref.end()};
}

template <typename T>
auto lbann::protobuf::as_unordered_set(
  google::protobuf::RepeatedField<T> const& rf) -> std::unordered_set<T>
{
  return std::unordered_set<T>{rf.cbegin(), rf.cend()};
}

inline auto lbann::protobuf::as_unordered_set(
  google::protobuf::RepeatedPtrField<std::string> const& rf)
  -> std::unordered_set<std::string>
{
  return std::unordered_set<std::string>{rf.cbegin(), rf.cend()};
}

template <typename OutT, typename InT>
auto lbann::protobuf::to_unordered_set(
  google::protobuf::RepeatedField<InT> const& rf) -> std::unordered_set<OutT>
{
  return std::unordered_set<OutT>{rf.cbegin(), rf.cend()};
}

#endif // LBANN_UTILS_PROTOBUF_IMPL_HPP_INCLUDED
