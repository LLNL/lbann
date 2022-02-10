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

#include <catch2/catch.hpp>

#include "TestHelpers.hpp"
#include "MPITestHelpers.hpp"

#include <lbann/base.hpp>
#include <lbann/callbacks/print_statistics.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/serialize.hpp>

using unit_test::utilities::IsValidPtr;
TEST_CASE("Serializing \"print statistics\" callback",
          "[mpi][callback][serialize]")
{
  using CallbackType = lbann::callback::print_statistics;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  CallbackType src(3, true), tgt(4, false);
  std::unique_ptr<lbann::callback_base>
    src_ptr = lbann::make_unique<CallbackType>(3, true),
    tgt_ptr;

  // Verify that the callbacks differ in the first place.
  CHECK_FALSE(src.get_batch_interval() == tgt.get_batch_interval());
  CHECK(IsValidPtr(src_ptr));
  CHECK_FALSE(IsValidPtr(tgt_ptr));

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(src));
      REQUIRE_NOTHROW(oarchive(src_ptr));
    }

    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(tgt));
      REQUIRE_NOTHROW(iarchive(tgt_ptr));
      CHECK(src.get_batch_interval() == tgt.get_batch_interval());
      CHECK(IsValidPtr(tgt_ptr));
      CHECK(src_ptr->get_batch_interval() == tgt_ptr->get_batch_interval());
    }
  }

  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src));
      REQUIRE_NOTHROW(oarchive(src_ptr));
    }

    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(tgt));
      REQUIRE_NOTHROW(iarchive(tgt_ptr));
      CHECK(src.get_batch_interval() == tgt.get_batch_interval());
      CHECK(IsValidPtr(tgt_ptr));
      CHECK(src_ptr->get_batch_interval() == tgt_ptr->get_batch_interval());
    }
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(src));
      REQUIRE_NOTHROW(oarchive(src_ptr));
    }

    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(tgt));
      REQUIRE_NOTHROW(iarchive(tgt_ptr));
      CHECK(src.get_batch_interval() == tgt.get_batch_interval());
      CHECK(IsValidPtr(tgt_ptr));
      CHECK(src_ptr->get_batch_interval() == tgt_ptr->get_batch_interval());
    }
  }

  SECTION("Rooted binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src));
      REQUIRE_NOTHROW(oarchive(src_ptr));
    }

    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(tgt));
      REQUIRE_NOTHROW(iarchive(tgt_ptr));
      CHECK(src.get_batch_interval() == tgt.get_batch_interval());
      CHECK(IsValidPtr(tgt_ptr));
      CHECK(src_ptr->get_batch_interval() == tgt_ptr->get_batch_interval());
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
}
