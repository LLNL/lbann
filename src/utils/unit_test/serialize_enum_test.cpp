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
#include <lbann/utils/serialize.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

namespace
{
enum class myenum
{
  DEFAULT,
  NOT_THE_DEFAULT,
};
}

TEST_CASE("Serializing enums", "[serialize][utils][enum]")
{
  auto& comm = ::unit_test::utilities::current_world_comm();
  auto const& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;
  myenum a = myenum::NOT_THE_DEFAULT;
  myenum b = myenum::DEFAULT;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary Archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
  SECTION("Rooted Binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML Archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(a));
    }
    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(b));
    }

    REQUIRE(a == b);
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
}
