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

#include "lbann/callbacks/setup_communitygan_data_reader.hpp"
#include "lbann/data_readers/data_reader_communitygan.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/serialize.hpp"
#include <callbacks.pb.h>
#include "lbann/callbacks/callback.hpp"
#ifdef LBANN_HAS_COMMUNITYGAN_WALKER
#include "CommunityGANWalker.hpp"
#endif // LBANN_HAS_COMMUNITYGAN_WALKER

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_COMMUNITYGAN_WALKER

// Initialize static private variable
::lbann::communitygan_reader* setup_communitygan_data_reader::m_reader
  = nullptr;

setup_communitygan_data_reader* setup_communitygan_data_reader::copy() const {
  return new setup_communitygan_data_reader(*this);
}

std::string setup_communitygan_data_reader::name() const {
  return "setup_communitygan_data_reader";
}

template <class Archive>
void setup_communitygan_data_reader::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)));
}

void setup_communitygan_data_reader::on_setup_end(model *m) {

  // Get CommunityGAN data reader
  if (m_reader == nullptr) {
    LBANN_ERROR(
      "\"setup_communitygan_data_reader\" callback attempted to access ",
      "CommunityGAN data reader before it has been set");
  }
  auto& reader = *m_reader;

  // Get embeddings
  const auto& embedding_weights_name = reader.m_embedding_weights_name;
  weights* embedding_weights = nullptr;
  for (auto w : m->get_weights()) {
    if (w->get_name() == embedding_weights_name) {
      embedding_weights = w;
      break;
    }
  }
  if (embedding_weights == nullptr) {
    LBANN_ERROR(
      "\"setup_communitygan_data_reader\" callback could not find ",
      "weights \"",embedding_weights_name,"\" in ",
      "model \"",m->get_name(),"\"");
  }
  auto& embeddings
    = dynamic_cast<data_type_weights<float>*>(embedding_weights)->get_values();

  // Construct CommunityGAN walker
  reader.m_walker.reset(
    new ::CommunityGANWalker(
      reader.get_comm()->get_trainer_comm().GetMPIComm(),
      reader.m_graph_file,
      embeddings.Buffer(),
      static_cast<int>(embeddings.Width()),
      static_cast<int>(embeddings.Height()),
      static_cast<int>(reader.m_walk_length-1),
      static_cast<int>(reader.m_cache_size)));

}

void setup_communitygan_data_reader::register_communitygan_data_reader(
  ::lbann::communitygan_reader* reader) {
  m_reader = reader;
}

#endif // LBANN_HAS_COMMUNITYGAN_WALKER

std::unique_ptr<callback_base>
build_setup_communitygan_data_reader_callback_from_pbuf(
  const google::protobuf::Message&, const std::shared_ptr<lbann_summary>&) {
#ifndef LBANN_HAS_COMMUNITYGAN_WALKER
  LBANN_ERROR(
    "\"setup_communitygan_data_reader\" callback requires ",
    "LBANN to be built with CommunityGAN walker");
  return nullptr;
#else
  return make_unique<setup_communitygan_data_reader>();
#endif // LBANN_HAS_COMMUNITYGAN_WALKER
}

} // namespace callback
} // namespace lbann

#ifdef LBANN_HAS_COMMUNITYGAN_WALKER
#define LBANN_CLASS_NAME callback::setup_communitygan_data_reader
#include <lbann/macros/register_class_with_cereal.hpp>
#endif // LBANN_HAS_COMMUNITYGAN_WALKER
