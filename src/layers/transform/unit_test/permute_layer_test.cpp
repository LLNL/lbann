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

// MUST include this
#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

#include "lbann/base.hpp"
#include "lbann/layers/transform/permute.hpp"
#include "lbann/utils/description.hpp"
#include <lbann/utils/serialize.hpp>
#include <lbann/utils/typename.hpp>
#include <lbann_config.hpp>

#include <sstream>
#include <vector>

using TheTestTypes = h2::meta::TL<
#ifdef LBANN_HAS_GPU_FP16
  ::lbann::fp16,
#endif
  float,
  double>;

static std::string to_str(::lbann::description const& d)
{
  std::ostringstream oss;
  oss << d;
  return oss.str();
}

static std::string desc(lbann::Layer const& l)
{
  return to_str(l.get_description());
}

using unit_test::utilities::IsValidPtr;
TEMPLATE_LIST_TEST_CASE("Serializing PermuteLayers",
                        "[mpi][layer][serialize]",
                        TheTestTypes)
{
  using DataT = TestType;
  using LayerType = ::lbann::PermuteLayer<DataT>;

  INFO("Type = " << ::lbann::TypeName<DataT>());

  auto& world_comm = unit_test::utilities::current_world_comm();

  auto const& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  LayerType const src_layer(std::vector<int>{3, 2, 0, 1});
  LayerType tgt_layer(std::vector<int>{0, 1, 2, 3});

  auto src_layer_ptr =
    std::make_unique<LayerType>(std::vector<int>{3, 2, 1, 0});
  auto tgt_layer_ptr =
    std::make_unique<LayerType>(std::vector<int>{0, 1, 2, 3});

  // Archive stream
  std::stringstream ss;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(src_layer));
      REQUIRE_NOTHROW(oarchive(src_layer_ptr));
    }

    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(tgt_layer));
      REQUIRE_NOTHROW(iarchive(tgt_layer_ptr));
      CHECK(IsValidPtr(tgt_layer_ptr));
    }

    CHECK(desc(src_layer) == desc(tgt_layer));
    CHECK(desc(*src_layer_ptr) == desc(*tgt_layer_ptr));
  }

  SECTION("Rooted binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src_layer));
      REQUIRE_NOTHROW(oarchive(src_layer_ptr));
    }

    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(tgt_layer));
      REQUIRE_NOTHROW(iarchive(tgt_layer_ptr));
      CHECK(IsValidPtr(tgt_layer_ptr));
    }

    CHECK(desc(src_layer) == desc(tgt_layer));
    CHECK(desc(*src_layer_ptr) == desc(*tgt_layer_ptr));
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(src_layer));
      REQUIRE_NOTHROW(oarchive(src_layer_ptr));
    }

    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(tgt_layer));
      REQUIRE_NOTHROW(iarchive(tgt_layer_ptr));
      CHECK(IsValidPtr(tgt_layer_ptr));
    }

    CHECK(desc(src_layer) == desc(tgt_layer));
    CHECK(desc(*src_layer_ptr) == desc(*tgt_layer_ptr));
  }

  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src_layer));
      REQUIRE_NOTHROW(oarchive(src_layer_ptr));
    }

    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(tgt_layer));
      REQUIRE_NOTHROW(iarchive(tgt_layer_ptr));
      CHECK(IsValidPtr(tgt_layer_ptr));
    }

    CHECK(desc(src_layer) == desc(tgt_layer));
    CHECK(desc(*src_layer_ptr) == desc(*tgt_layer_ptr));
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
