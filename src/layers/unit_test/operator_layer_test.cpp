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
#include <catch2/catch.hpp>
#include <memory>

#include "lbann/base.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/operator_layer.hpp"
#include "lbann/operators/math/clamp.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

template <typename T, lbann::data_layout L, El::Device D>
using LayerType = lbann::OperatorLayer<T, T, L, D>;

template <typename T, lbann::data_layout L>
using LayerTypeAllDevices = h2::meta::TL<
#ifdef LBANN_HAS_GPU
  LayerType<T, L, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  LayerType<T, L, El::Device::CPU>>;

template <typename T>
using LayerTypeAllLayoutsAndDevice = h2::meta::tlist::Append<
  LayerTypeAllDevices<T, lbann::data_layout::DATA_PARALLEL>,
  LayerTypeAllDevices<T, lbann::data_layout::MODEL_PARALLEL>>;

using AllLayerTypes = h2::meta::tlist::Append<
#ifdef LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU
  LayerTypeAllLayoutsAndDevice<lbann::fp16>,
#endif // LBANN_HAS_GPU
  LayerTypeAllLayoutsAndDevice<lbann::cpu_fp16>,
#endif // LBANN_HAS_HALF
  LayerTypeAllLayoutsAndDevice<float>,
  LayerTypeAllLayoutsAndDevice<double>>;

template <typename LayerT>
struct LayerTraits;

template <typename T, lbann::data_layout L, El::Device D>
struct LayerTraits<LayerType<T, L, D>>
{
  using value_type = T;
  static constexpr lbann::data_layout layout = L;
  static constexpr El::Device device = D;
};

template <typename LayerT>
using ValueType = typename LayerTraits<LayerT>::value_type;

template <typename LayerT>
inline constexpr auto DataLayout = LayerTraits<LayerT>::layout;

template <typename LayerT>
inline constexpr auto Device = LayerTraits<LayerT>::device;

using unit_test::utilities::IsValidPtr;
TEMPLATE_LIST_TEST_CASE("OperatorLayer lifecycle",
                        "[layer][operatorlayer][mpi][lifecycle]",
                        AllLayerTypes)
{
  using LayerPtr = std::unique_ptr<lbann::Layer>;
  using OperatorLayer = TestType;
  using ValueType = ValueType<TestType>;
  constexpr auto DeviceAlloc = Device<TestType>;

  using OpType = lbann::Operator<ValueType, ValueType, DeviceAlloc>;
  using ClampOpType = lbann::ClampOperator<ValueType, DeviceAlloc>;

  auto& world_comm = unit_test::utilities::current_world_comm();

  SECTION("Construct with a single operator")
  {
    LayerPtr layer = nullptr;
    REQUIRE_NOTHROW(layer = std::make_unique<OperatorLayer>(
                      world_comm,
                      std::make_unique<ClampOpType>(-1.0, 1.0)));
    CHECK(IsValidPtr(layer));
  }
  SECTION("Construct with a vector holding a single operator")
  {
    LayerPtr layer = nullptr;
    std::vector<std::unique_ptr<OpType>> ops;
    ops.reserve(1);
    ops.push_back(std::make_unique<ClampOpType>(-1.0, 1.0));
    REQUIRE_NOTHROW(
      layer = std::make_unique<OperatorLayer>(world_comm, std::move(ops)));
    CHECK(IsValidPtr(layer));
  }
  SECTION("Constructing with multiple operators fails")
  {
    LayerPtr layer = nullptr;
    std::vector<std::unique_ptr<OpType>> ops;
    ops.reserve(2);
    ops.push_back(std::make_unique<ClampOpType>(-1.0, 1.0));
    ops.push_back(std::make_unique<ClampOpType>(-0.5, 0.5));
    REQUIRE_THROWS(
      layer = std::make_unique<OperatorLayer>(world_comm, std::move(ops)));
    CHECK_FALSE(IsValidPtr(layer));
  }
  SECTION("Copy construction")
  {
    LayerPtr layer = nullptr;
    OperatorLayer src_layer(world_comm,
                            std::make_unique<ClampOpType>(-1.0, 1.0));
    REQUIRE_NOTHROW(layer = std::make_unique<OperatorLayer>(src_layer));
    CHECK(IsValidPtr(layer));
  }
  SECTION("Move construction")
  {
    LayerPtr layer = nullptr;
    OperatorLayer src_layer(world_comm,
                            std::make_unique<ClampOpType>(-1.0, 1.0));
    REQUIRE_NOTHROW(layer =
                      std::make_unique<OperatorLayer>(std::move(src_layer)));
    CHECK(IsValidPtr(layer));
  }
  SECTION("Copy assignment")
  {
    OperatorLayer src(world_comm, std::make_unique<ClampOpType>(-1.0, 1.0));
    OperatorLayer tgt(world_comm, std::make_unique<ClampOpType>(-2.0, 2.0));
    REQUIRE_NOTHROW(tgt = src);
  }
  SECTION("Move assignment")
  {
    OperatorLayer src(world_comm, std::make_unique<ClampOpType>(-1.0, 1.0));
    OperatorLayer tgt(world_comm, std::make_unique<ClampOpType>(-2.0, 2.0));
    REQUIRE_NOTHROW(tgt = std::move(src));
  }
}
