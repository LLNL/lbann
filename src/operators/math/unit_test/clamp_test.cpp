////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

// Testing framework stuff
#include <catch2/catch.hpp>

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

// CUT
#include "lbann/operators/math/clamp.hpp"

// Other stuff
#include "lbann/utils/serialize.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

#include <functional>
#include <memory>
#include <numeric>

// Build the input/expected output matrices.
template <typename T, El::Dist U, El::Dist V>
auto get_input(El::DistMatrix<T, U, V, El::ELEMENT, El::Device::CPU>& mat,
               std::vector<size_t> const& size,
               std::size_t ldim_factor = 0UL)
{
  auto width = size[0];
  auto height =
    std::accumulate(cbegin(size) + 1, cend(size), 1, std::multiplies<size_t>());
  auto ldim = height + ldim_factor;
  mat.Resize(height, width, ldim);
}

// Define the list of operators to test. Basically this is
// {float,double}x{CPU,GPU}.
template <typename T>
using ClampOperatorAllDevices = h2::meta::TL<
#ifdef LBANN_HAS_GPU
  lbann::ClampOperator<T, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  lbann::ClampOperator<T, El::Device::CPU>>;

using AllClampOpTypes =
  h2::meta::tlist::Append<ClampOperatorAllDevices<float>,
                          ClampOperatorAllDevices<double>>;

template <typename T>
struct OperatorTraits;

template <typename T, El::Device D>
struct OperatorTraits<lbann::ClampOperator<T, D>>
{
  using value_type = T;
  using base_type = lbann::Operator<T, T, D>;
  using data_parallel_mat_type =
    El::DistMatrix<T, El::Dist::STAR, El::Dist::VC, El::DistWrap::ELEMENT, D>;
  using model_parallel_mat_type =
    El::DistMatrix<T, El::Dist::MC, El::Dist::MR, El::DistWrap::ELEMENT, D>;
  using tensor_type = lbann::utils::DistTensorView<T,D>;
  using const_tensor_type = lbann::utils::ConstDistTensorView<T,D>;
  static constexpr El::Device device = D;
};

template <typename OpT>
using ValueType = typename OperatorTraits<OpT>::value_type;
template <typename OpT>
using BaseOperatorType = typename OperatorTraits<OpT>::base_type;
template <typename OpT>
using DataParallelMatType =
  typename OperatorTraits<OpT>::data_parallel_mat_type;
template <typename OpT>
using ModelParallelMatType =
  typename OperatorTraits<OpT>::model_parallel_mat_type;
template <typename OpT>
constexpr auto DeviceAlloc = OperatorTraits<OpT>::device;
template <typename OpT>
using TensorType = typename OperatorTraits<OpT>::tensor_type;
template <typename OpT>
using ConstTensorType = typename OperatorTraits<OpT>::const_tensor_type;

TEMPLATE_LIST_TEST_CASE("Clamp operator forward action",
                        "[mpi][operator][math][action]",
                        AllClampOpTypes)
{
  using ThisOpType = TestType;
  using DataType = ValueType<ThisOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();

  SECTION("Data parallel - all values in range")
  {
    El::Int const height = 13;
    El::Int const width = 17;

    ThisOpType op(El::To<DataType>(-1.0), El::To<DataType>(1.0));

    DataParallelMatType<ThisOpType> input(height, width, g, 0),
      output(height, width, g, 0);

    // Setup inputs/outputs
    El::MakeUniform(input);
    El::Fill(output, El::To<DataType>(2.0)); // Fill out of range.

    El::break_on_me();
    CHECK_FALSE(input == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(input == output);
  }
}

// Save some typing.
using unit_test::utilities::IsValidPtr;
TEMPLATE_LIST_TEST_CASE("Serializing Clamp operator",
                        "[mpi][operator][math][serialize]",
                        AllClampOpTypes)
{
  using ThisOpType = TestType;
  using BaseOpType = BaseOperatorType<ThisOpType>;
  using BaseOpPtr = std::unique_ptr<BaseOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  // int const size_of_world = world_comm.get_procs_in_world();

  auto const& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  ThisOpType src_layer(1.f, 2.f), tgt_layer(0.f, 1.f);
  BaseOpPtr src_layer_ptr = std::make_unique<ThisOpType>(3.f, 4.f),
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
