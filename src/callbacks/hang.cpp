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

#include "lbann/callbacks/hang.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

namespace lbann {
namespace callback {

template <class Archive>
void hang::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_rank_to_hang));
}

void hang::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_hang();
  msg->set_rank(m_rank_to_hang);
}

void hang::setup(model* m)
{
  if (m->get_comm()->am_world_master()) {
    if (m_rank_to_hang == -1) {
      std::cout << "*** HANGING EVERY RANK IN HANG CALLBACK ***"
                << std::endl;
    } else {
      std::cout << "*** HANGING RANK " << m_rank_to_hang
                << " IN HANG CALLBACK ***" << std::endl;
    }
  }
}

std::unique_ptr<callback_base>
build_hang_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, std::shared_ptr<lbann_summary> const&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackHang&>(proto_msg);
  return std::make_unique<hang>(params.rank());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::hang
#define LBANN_CLASS_LIBNAME callback_hang
#include <lbann/macros/register_class_with_cereal.hpp>
