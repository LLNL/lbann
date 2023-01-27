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

#include "Catch2BasicSupport.hpp"

#include "TestHelpers.hpp"
#include "MPITestHelpers.hpp"

#include <lbann/base.hpp>
#include <lbann/models/model.hpp>
#include <lbann/layers/io/input_layer.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/serialize.hpp>
#include <lbann/utils/lbann_library.hpp>
#include <lbann/proto/factories.hpp>

#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>

namespace pb = ::google::protobuf;

namespace {
// model_prototext string is defined here as a "const std::string".
#include "lenet.prototext.inc"

auto mock_datareader_metadata()
{
  lbann::DataReaderMetaData md;
  auto& md_dims = md.data_dims;
  // This is all that should be needed for this test.
  md_dims[lbann::data_reader_target_mode::CLASSIFICATION] = {10};
  md_dims[lbann::data_reader_target_mode::INPUT] = {1,28,28};
  return md;
}

template <typename T>
auto make_model(lbann::lbann_comm& comm)
{
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(model_prototext, &my_proto))
    throw "Parsing protobuf failed.";
  // Construct a trainer so that the model can register the input layer
  lbann::construct_trainer(&comm, my_proto.mutable_trainer(), my_proto);
  auto metadata = mock_datareader_metadata();
  auto my_model = lbann::proto::construct_model(&comm,
                                                -1,
                                                my_proto.optimizer(),
                                                my_proto.trainer(),
                                                my_proto.model());
  my_model->setup(1UL, metadata, {&comm.get_trainer_grid()});
  return my_model;
}

}// namespace <anon>

using unit_test::utilities::IsValidPtr;
TEST_CASE("Serializing models", "[mpi][model][serialize]")
{
  using DataType = float;

  auto& comm = unit_test::utilities::current_world_comm();

  auto& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;
  std::unique_ptr<lbann::model>
    model_src_ptr = make_model<DataType>(comm),
    model_tgt_ptr;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }
    if (IsValidPtr(model_tgt_ptr))
    {
      auto metadata = mock_datareader_metadata();
      REQUIRE_NOTHROW(model_tgt_ptr->setup(1UL, metadata, {&g}));
      // if (comm.get_rank_in_world() == 1)
      //   std::cout << model_tgt_ptr->get_description()
      //             << std::endl;
    }
  }

  SECTION("Rooted binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }

    if (IsValidPtr(model_tgt_ptr))
    {
      auto metadata = mock_datareader_metadata();
      REQUIRE_NOTHROW(model_tgt_ptr->setup(1UL, metadata, {&g}));
      // if (comm.get_rank_in_world() == 1)
      //   std::cout << model_tgt_ptr->get_description()
      //             << std::endl;
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }
  }

  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    //std::cout << ss.str() << std::endl;
    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
