////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/utils/protobuf.hpp"

#include "lbann/utils/exception.hpp"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>
#include <memory>
#include <string>

// Note that the implementations in this file emphasize clarity and
// simplicity over performance, primarily with respect to their use of
// streams. If any of the code ever becomes performance-critical, it
// should be refactored to use the specialized "zero-copy" streams
// that ship with protobuf.

namespace pb = google::protobuf;
static pb::FieldDescriptor const*
get_oneof_field_descriptor(pb::Message const& msg,
                           std::string const& oneof_name)
{
  auto oneof_handle = msg.GetDescriptor()->FindOneofByName(oneof_name);
  if (!oneof_handle)
    return nullptr;
  return msg.GetReflection()->GetOneofFieldDescriptor(msg, oneof_handle);
}

bool lbann::protobuf::has_oneof(pb::Message const& msg,
                                std::string const& oneof_name)
{
  return (bool)get_oneof_field_descriptor(msg, oneof_name);
}

pb::Message const&
lbann::protobuf::get_oneof_message(pb::Message const& msg,
                                   std::string const& oneof_name)
{
  auto oneof_field = get_oneof_field_descriptor(msg, oneof_name);
  if (!oneof_field) {
    LBANN_ERROR("Oneof field \"",
                oneof_name,
                "\" in message has not been set. Message:\n{\n",
                msg.DebugString(),
                "}\n");
  }

  if (oneof_field->type() != pb::FieldDescriptor::TYPE_MESSAGE) {
    LBANN_ERROR("Oneof \"",
                oneof_name,
                "\" has field \"",
                oneof_field->name(),
                "\" set but it is not of message type. Detected type: ",
                oneof_field->type_name());
  }

  return msg.GetReflection()->GetMessage(msg, oneof_field);
}

std::string lbann::protobuf::which_oneof(google::protobuf::Message const& msg,
                                         std::string const& oneof_name)
{
  auto oneof_field = get_oneof_field_descriptor(msg, oneof_name);
  if (!oneof_field)
    LBANN_ERROR("Oneof field \"",
                oneof_name,
                "\" in message has not been set. Message:\n{\n",
                msg.DebugString(),
                "}\n");
  return oneof_field->name();
}

static std::string remove_scope_from_type(std::string const& type)
{
  auto pos = type.rfind('.');
  if (pos == std::string::npos)
    return type; // Assume the whole thing is just the type
  else
    return type.substr(pos + 1);
}

std::string lbann::protobuf::message_type(pb::Message const& msg)
{
  return msg.GetDescriptor()->name();
}

std::string lbann::protobuf::message_type(pb::Any const& msg)
{
  std::string full_type;
  pb::Any::ParseAnyTypeUrl(msg.type_url(), &full_type);
  return remove_scope_from_type(full_type);
}

// I/O operations

void lbann::protobuf::fill(std::istream& is, google::protobuf::Message& msg)
{
  if (!msg.ParseFromIstream(&is))
    LBANN_ERROR("Failed to fill message from protocol buffer stream.");
}

void lbann::protobuf::fill(std::string const& pbuf_str,
                           google::protobuf::Message& msg)
{
  if (!msg.ParseFromString(pbuf_str))
    LBANN_ERROR("Failed to fill message from protocol buffer string.");
}

void lbann::protobuf::load(std::string const& pbuf_file,
                           google::protobuf::Message& msg)
{
  std::ifstream ifs(pbuf_file, std::ios::binary | std::ios::in);
  LBANN_ASSERT((bool)ifs);
  fill(ifs, msg);
}

void lbann::protobuf::serialize(std::ostream& os,
                                google::protobuf::Message const& msg)
{
  if (!msg.SerializeToOstream(&os))
    LBANN_ERROR("Failed to serialize protobuf to stream.");
}

std::string lbann::protobuf::serialize(google::protobuf::Message const& msg)
{
  return msg.SerializeAsString();
}

void lbann::protobuf::serialize(std::string const& pbuf_file,
                                google::protobuf::Message const& msg)
{
  std::ofstream ofs(pbuf_file, std::ios_base::binary);
  LBANN_ASSERT((bool)ofs);
  serialize(ofs, msg);
}

void lbann::protobuf::text::fill(std::istream& is,
                                 google::protobuf::Message& msg)
{
  google::protobuf::io::IstreamInputStream input(&is);
  if (!google::protobuf::TextFormat::Parse(&input, &msg))
    LBANN_ERROR("Unable to parse prototext from stream.");
}

void lbann::protobuf::text::fill(std::string const& str,
                                 google::protobuf::Message& msg)
{
  if (!pb::TextFormat::ParseFromString(str, &msg))
    LBANN_ERROR("Unable to parse prototext from string.");
}

void lbann::protobuf::text::load(std::string const& ptext_file,
                                 google::protobuf::Message& msg)
{
  std::ifstream ifs(ptext_file);
  LBANN_ASSERT((bool)ifs);
  fill(ifs, msg);
}

void lbann::protobuf::text::write(std::ostream& os,
                                  google::protobuf::Message const& msg)
{
  google::protobuf::io::OstreamOutputStream output(&os);
  if (!pb::TextFormat::Print(msg, &output))
    LBANN_ERROR("Failed to print prototext to stream.");
}

std::string lbann::protobuf::text::write(google::protobuf::Message const& msg)
{
  std::string str;
  write(str, msg);
  return str;
}

void lbann::protobuf::text::write(std::string const& ptext_file,
                                  google::protobuf::Message const& msg)
{
  std::ofstream ofs(ptext_file);
  LBANN_ASSERT((bool)ofs);
  write(ofs, msg);
}
