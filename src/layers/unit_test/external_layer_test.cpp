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

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/misc/external.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

using unit_test::utilities::IsValidPtr;
TEST_CASE("External Layer test", "[layer][externallayer]")
{
  auto& world_comm = unit_test::utilities::current_world_comm();

  SECTION("Construct a layer from an external library")
  {
    lbann::external_layer_setup_t setupfunc = nullptr;

    REQUIRE_NOTHROW(setupfunc = lbann::load_external_library(
                      "src/layers/unit_test/libexample_layer.so",
                      "layer"));
    CHECK(setupfunc != nullptr);

    // Configuration that exists
    lbann::Layer* layer = nullptr;
    REQUIRE_NOTHROW(layer = setupfunc(lbann_data::FLOAT,
                                      lbann::data_layout::DATA_PARALLEL,
                                      El::Device::CPU,
                                      &world_comm));
    CHECK(IsValidPtr(layer));

    // Configuration that does not exist
    lbann::Layer* nonexistent = nullptr;
    REQUIRE_NOTHROW(nonexistent = setupfunc(lbann_data::COMPLEX_DOUBLE,
                                            lbann::data_layout::DATA_PARALLEL,
                                            El::Device::GPU,
                                            &world_comm));
    CHECK(!IsValidPtr(nonexistent));
  }
  SECTION("Construct a nonexistent layer")
  {
    lbann::external_layer_setup_t setupfunc = nullptr;

    REQUIRE_THROWS(setupfunc = lbann::load_external_library(
                     "src/layers/unit_test/libexample_layer.so",
                     "DOES_NOT_EXIST"));
    CHECK(setupfunc == nullptr);
  }
  SECTION("Load a nonexistent external library")
  {
    lbann::external_layer_setup_t setupfunc = nullptr;

    REQUIRE_THROWS(setupfunc =
                     lbann::load_external_library("NO_LIBRARY_HERE", "layer"));
    CHECK(setupfunc == nullptr);
  }
}
