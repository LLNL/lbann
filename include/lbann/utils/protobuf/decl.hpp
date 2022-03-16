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
#ifndef LBANN_UTILS_PROTOBUF_DECL_HPP_INCLUDED
#define LBANN_UTILS_PROTOBUF_DECL_HPP_INCLUDED

/** @file A small library of utilities for interfacing with Google
 *        protobuf messages.
 */

#include "lbann/utils/exception.hpp"

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>
#include <google/protobuf/reflection.h>
#include <google/protobuf/repeated_field.h>

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace lbann {
namespace protobuf {

/** @name Typedefs for working with Protobuf fixed-width types. */
///@{

using int8 = google::protobuf::int8;
using int16 = google::protobuf::int16;
using int32 = google::protobuf::int32;
using int64 = google::protobuf::int64;

using uint8 = google::protobuf::uint8;
using uint16 = google::protobuf::uint16;
using uint32 = google::protobuf::uint32;
using uint64 = google::protobuf::uint64;

///@}
/** @name Message queries */
///@{

/** @brief Test whether the message has a oneof field with the given
 *         name.
 *  @returns @c true iff the message has a field with the given name
 *           and the field is a oneof.
 */
bool has_oneof(google::protobuf::Message const& msg,
               std::string const& oneof_name);

/** @brief Get a message in a oneof from the given message.
 *
 *  A oneof field with the given name must exist in the message. The
 *  value set in the oneof must be a message.
 */
google::protobuf::Message const&
get_oneof_message(google::protobuf::Message const& msg,
                  std::string const& oneof_name);

/** @brief Get the name of the set field in the named oneof.
 *
 *  A oneof with the given name must exist in the message.
 */
std::string which_oneof(google::protobuf::Message const& msg,
                        std::string const& oneof_name);

/** @brief Get the name of the message type as a string. */
std::string message_type(google::protobuf::Message const& msg);

/** @brief Get the name of the underlying message type as a string. */
std::string message_type(google::protobuf::Any const& any);

///@}
/** @name Complex extraction methods */
///@{

/** @brief Extract the values from the named field as a vector. */
template <typename T>
auto as_vector(google::protobuf::Message const& msg,
               std::string const& field_name) -> std::vector<T>;

/** @brief Convert the repeated field to an STL vector.
 *
 *  This overload covers basic POD types, to include enum types.
 */
template <typename T>
auto as_vector(google::protobuf::RepeatedField<T> const&) -> std::vector<T>;

/** @brief Convert the repeated field to an STL vector.
 *
 *  This overload covers "complex" types, namely strings and messages.
 */
template <typename T>
auto as_vector(google::protobuf::RepeatedPtrField<T> const&) -> std::vector<T>;

/** @brief Extract the values from the named field as a set. */
template <typename T>
auto as_set(google::protobuf::Message const& msg, std::string const& field_name)
  -> std::set<T>;

/** @brief Convert the repeated field to an STL set.
 *
 *  This overload covers basic POD types, to include enum types.
 */
template <typename T>
auto as_set(google::protobuf::RepeatedField<T> const&) -> std::set<T>;

/** @brief Convert the repeated field to an STL set.
 *
 *  This overload is for strings.
 */
auto as_set(google::protobuf::RepeatedPtrField<std::string> const&)
  -> std::set<std::string>;

/** @brief Extract the values from the named field as an unordered_set. */
template <typename T>
auto as_unordered_set(google::protobuf::Message const& msg,
                      std::string const& field_name) -> std::unordered_set<T>;

/** @brief Convert the repeated field to an STL unordered_set.
 *
 *  This overload covers basic POD types, to include enum types.
 */
template <typename T>
auto as_unordered_set(google::protobuf::RepeatedField<T> const&)
  -> std::unordered_set<T>;

/** @brief Convert the repeated field to an STL unordered_set.
 *
 *  This overload is for strings.
 */
auto as_unordered_set(google::protobuf::RepeatedPtrField<std::string> const&)
  -> std::unordered_set<std::string>;

///@}
/** @name Message I/O */
///@{

/** @brief Fill the protobuf message from a binary stream. */
void fill(std::istream& is, google::protobuf::Message& msg);

/** @brief Fill the protobuf message from a string of bytes. */
void fill(std::string const& pbuf_str, google::protobuf::Message& msg);

/** @brief Fill the protobuf message from a binary file. */
void load(std::string const& pbuf_filename, google::protobuf::Message& msg);

/** @brief Serialize the protobuf message to a stream. */
void serialize(std::ostream& os, google::protobuf::Message const& msg);

/** @brief Serialize the protobuf message to a string. */
std::string serialize(google::protobuf::Message const& msg);

/** @brief Serialize the protobuf message to a file. */
void serialize(std::string const& pbuf_filename,
               google::protobuf::Message const& msg);

///@}

namespace text {
/** @name Prototext I/O */
///@{

/** @brief Fill the protobuf message from prototext in a stream. */
void fill(std::istream& is, google::protobuf::Message& msg);

/** @brief Fill the protobuf message from prototext in a string. */
void fill(std::string const& str, google::protobuf::Message& msg);

/** @brief Fill the protobuf message from prototext in a file. */
void load(std::string const& ptext_filename, google::protobuf::Message& msg);

/** @brief Write the protobuf message in prototext in a stream. */
void write(std::ostream& os, google::protobuf::Message const& msg);

/** @brief Write the protobuf message in prototext into a string. */
std::string write(google::protobuf::Message const& msg);

/** @brief Write the protobuf message in prototext to a file. */
void write(std::string const& ptext_filename,
           google::protobuf::Message const& msg);
///@}
} // namespace text
} // namespace protobuf
} // namespace lbann

#endif // LBANN_UTILS_PROTOBUF_DECL_HPP_INCLUDED
