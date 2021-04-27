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

#include "lbann/proto/helpers.hpp"
#include "lbann/utils/exception.hpp"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <string>

namespace lbann {
namespace proto {
namespace helpers {
namespace {
google::protobuf::FieldDescriptor const* get_oneof_field_descriptor(
  google::protobuf::Message const& msg_in, std::string const& oneof_name) {
  auto desc = msg_in.GetDescriptor();
  auto reflex = msg_in.GetReflection();
  auto oneof_handle = desc->FindOneofByName(oneof_name);
  if (!oneof_handle)
  {
    return nullptr;
  }

  return reflex->GetOneofFieldDescriptor(msg_in, oneof_handle);
}
}// namespace

bool has_oneof(
  google::protobuf::Message const& msg, std::string const& oneof_name)
{
  return (bool) get_oneof_field_descriptor(msg, oneof_name);
}

google::protobuf::Message const&
get_oneof_message(
  google::protobuf::Message const& msg_in, std::string const& oneof_name) {
  auto oneof_field = get_oneof_field_descriptor(msg_in, oneof_name);
  if (!oneof_field) {
    LBANN_ERROR("Oneof field \"", oneof_name,
                "\" in message has not been set. Message:\n{",
                msg_in.DebugString(),"\n}\n");
  }

  if (oneof_field->type() != google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
    LBANN_ERROR("Oneof field is not of message type.");
  }

  return msg_in.GetReflection()->GetMessage(msg_in, oneof_field);
}

}// namespace helpers
}// namespace proto
}// namespace lbann

namespace {
std::string remove_scope_from_type(std::string const& type)
{
  auto pos = type.rfind('.');
  if (pos == std::string::npos)
    return type;// Assume the whole thing is just the type
  else
    return type.substr(pos+1);
}
} // namespace

std::string lbann::proto::helpers::message_type(
  google::protobuf::Message const& m)
{
  return m.GetDescriptor()->name();
}

std::string lbann::proto::helpers::message_type(
  google::protobuf::Any const& m)
{
  std::string full_type;
  google::protobuf::Any::ParseAnyTypeUrl(m.type_url(), &full_type);
  return remove_scope_from_type(full_type);
}
