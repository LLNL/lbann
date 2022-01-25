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
#include <lbann/optimizers/sgd.hpp>
#include <lbann/weights/data_type_weights.hpp>
#include <lbann/weights/weights.hpp>
#include <lbann/weights/weights_proxy.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/serialize.hpp>

// Some convenience typedefs

template <typename T>
using DataTypeWeights = lbann::data_type_weights<T>;

template <typename T>
using ConstantInitializer = lbann::constant_initializer<T>;

template <typename T>
using SGD = lbann::sgd<T>;

namespace {
template <typename T>
auto make_weights(lbann::lbann_comm& comm)
{
  return DataTypeWeights<T>(comm);
}

template <typename T>
auto make_weights(lbann::lbann_comm& comm, size_t height, size_t width)
{
  T const value = El::To<T>(1.3);
  DataTypeWeights<T> out(comm);
  out.set_dims({height}, {width});
  out.set_initializer(
    lbann::make_unique<ConstantInitializer<T>>(value));
  return out;
}
}// namespace <>

template <typename T>
auto make_weights_ptr(lbann::lbann_comm& comm, size_t height, size_t width)
{
  T const value = El::To<T>(1.3);
  auto out = std::make_unique<DataTypeWeights<T>>(comm);
  out->set_dims({height}, {width});
  out->set_initializer(
    lbann::make_unique<ConstantInitializer<T>>(value));
  return out;
}

using unit_test::utilities::IsValidPtr;
TEST_CASE("Serializing weights", "[mpi][weights][serialize]")
{
  using DataType = float;

  auto& world_comm = unit_test::utilities::current_world_comm();
  size_t const size_of_world = world_comm.get_procs_in_world();

  auto const& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  size_t const weights_height = 3 * size_of_world;
  size_t const weights_width = 2 * size_of_world;

  auto dtw_src = make_weights<DataType>(world_comm,
                                        weights_height,
                                        weights_width);
  auto dtw_tgt = make_weights<DataType>(world_comm);
  std::unique_ptr<DataTypeWeights<DataType>>
    dtw_src_ptr_init = make_weights_ptr<DataType>(world_comm,
                                                  weights_height,
                                                  weights_width);

  dtw_src.set_optimizer(lbann::make_unique<SGD<DataType>>(1.f, 2.f, true));
  dtw_src.setup();
  dtw_src_ptr_init->set_optimizer(
    lbann::make_unique<SGD<DataType>>(3.f, 4.f, false));
  dtw_src_ptr_init->setup();
  std::unique_ptr<lbann::weights>
    dtw_src_ptr = std::move(dtw_src_ptr_init),
    dtw_tgt_ptr;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(dtw_src));
      REQUIRE_NOTHROW(oarchive(dtw_src_ptr));
    }

    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(dtw_tgt));
      REQUIRE_NOTHROW(iarchive(dtw_tgt_ptr));
      CHECK(IsValidPtr(dtw_tgt_ptr));
    }
  }

  SECTION("Rooted binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(dtw_src));
      REQUIRE_NOTHROW(oarchive(dtw_src_ptr));
    }
    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(dtw_tgt));
      REQUIRE_NOTHROW(iarchive(dtw_tgt_ptr));
      CHECK(IsValidPtr(dtw_tgt_ptr));
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(dtw_src));
      REQUIRE_NOTHROW(oarchive(dtw_src_ptr));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(dtw_tgt));
      REQUIRE_NOTHROW(iarchive(dtw_tgt_ptr));
      CHECK(IsValidPtr(dtw_tgt_ptr));
    }
  }

  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(dtw_src));
      REQUIRE_NOTHROW(oarchive(dtw_src_ptr));
    }
    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(dtw_tgt));
      REQUIRE_NOTHROW(iarchive(dtw_tgt_ptr));
      CHECK(IsValidPtr(dtw_tgt_ptr));
    }
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
