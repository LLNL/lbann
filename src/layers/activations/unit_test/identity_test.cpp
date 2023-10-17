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

#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

#include <lbann/base.hpp>
#include <lbann/layers/activations/identity.hpp>

#include <h2/patterns/multimethods/SwitchDispatcher.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/serialize.hpp>

// Some convenience typedefs

template <typename T, lbann::data_layout L, El::Device D>
using LayerType = lbann::identity_layer<T, L, D>;

template <typename T, El::Device D>
using LayerTypesAllLayouts =
  h2::meta::TL<LayerType<T, lbann::data_layout::DATA_PARALLEL, D>,
               LayerType<T, lbann::data_layout::MODEL_PARALLEL, D>>;

template <typename T>
using LayerTypesAllDevices = h2::meta::tlist::Append<
#ifdef LBANN_HAS_GPU
  LayerTypesAllLayouts<T, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  LayerTypesAllLayouts<T, El::Device::CPU>>;

using AllLayerTypes = h2::meta::tlist::Append<
#ifdef LBANN_HAS_DOUBLE
  LayerTypesAllDevices<double>,
#endif // LBANN_HAS_DOUBLE
  LayerTypesAllDevices<float>>;

using unit_test::utilities::IsValidPtr;
TEMPLATE_LIST_TEST_CASE("Serializing Identity layer",
                        "[mpi][layer][serialize]",
                        AllLayerTypes)
{
  using LayerType = TestType;

  auto& world_comm = unit_test::utilities::current_world_comm();
  // int const size_of_world = world_comm.get_procs_in_world();

  auto const& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  LayerType src_layer, tgt_layer;
  std::unique_ptr<lbann::Layer> src_layer_ptr = std::make_unique<LayerType>(),
                                tgt_layer_ptr;

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
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
