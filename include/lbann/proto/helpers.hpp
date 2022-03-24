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

#ifndef LBANN_PROTO_HELPERS_HPP_INCLUDED
#define LBANN_PROTO_HELPERS_HPP_INCLUDED

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>

#include <functional>
#include <memory>
#include <string>

namespace lbann
{
namespace proto
{

namespace helpers
{

/** @brief Test whether the message has the oneof field. */
bool has_oneof(
  google::protobuf::Message const& msg, std::string const& oneof_name);

/** @brief Get a "derived type" message from the given message. */
google::protobuf::Message const&
get_oneof_message(
  google::protobuf::Message const& msg_in, std::string const& oneof_name);

/** @brief Get the name of the message type as a string. */
std::string message_type(google::protobuf::Message const& m);
std::string message_type(google::protobuf::Any const& m);

}// namespace helpers
}// namespace proto
}// namespace lbann
#endif /* LBANN_PROTO_HELPERS_HPP_INCLUDED */
