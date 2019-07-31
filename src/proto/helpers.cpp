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

google::protobuf::Message const&
get_oneof_message(
  google::protobuf::Message const& msg_in, std::string const& oneof_name)
{
  auto&& desc = msg_in.GetDescriptor();
  auto&& reflex = msg_in.GetReflection();
  auto&& oneof_handle = desc->FindOneofByName(oneof_name);
  if (!oneof_handle)
  {
      std::string msg_string;
      google::protobuf::TextFormat::PrintToString(msg_in, &msg_string);
    LBANN_ERROR(std::string("Message has no oneof field named \"")
                + oneof_name + "\"\n\nMessage("
                + desc->DebugString() +"):\n\n"
                + msg_string);
  }

  auto&& oneof_field = reflex->GetOneofFieldDescriptor(msg_in, oneof_handle);

  if (!oneof_field)
    LBANN_ERROR("Oneof field in message has not been set.");

  if (oneof_field->type() != google::protobuf::FieldDescriptor::TYPE_MESSAGE)
    LBANN_ERROR("Oneof field is not of message type.");

  return reflex->GetMessage(msg_in, oneof_field);
}

}// namespace helpers
}// namespace proto
}// namespace lbann
