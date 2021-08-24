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
#include "MatrixHelpers.hpp"
#include "TestHelpers.hpp"

// CUT
#include "lbann/operators/math/clamp.hpp"

// Other stuff
#include "lbann/proto/factories.hpp"
#include "lbann/utils/serialize.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

#include <functional>
#include <memory>
#include <numeric>
#include <operators.pb.h>

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
  using tensor_type = lbann::utils::DistTensorView<T, D>;
  using const_tensor_type = lbann::utils::ConstDistTensorView<T, D>;
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

// Save some typing.
using unit_test::utilities::IsValidPtr;

TEMPLATE_LIST_TEST_CASE("Clamp operator lifecycle",
                        "[mpi][operator][math][clamp][lifecycle]",
                        AllClampOpTypes)
{
  using ThisOpType = TestType;
  using DataType = ValueType<ThisOpType>;

  SECTION("Construction with valid arguments")
  {
    std::unique_ptr<ThisOpType> op_ptr = nullptr;
    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>(0., 1.));
    REQUIRE(IsValidPtr(op_ptr));
    CHECK(op_ptr->get_min() == El::To<DataType>(0.0));
    CHECK(op_ptr->get_max() == El::To<DataType>(1.0));

    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>(1., 1.));
    REQUIRE(IsValidPtr(op_ptr));
    CHECK(op_ptr->get_min() == El::To<DataType>(1.0));
    CHECK(op_ptr->get_max() == El::To<DataType>(1.0));
  }
  SECTION("Construction with invalid arguments")
  {
    std::unique_ptr<ThisOpType> op_ptr = nullptr;
    CHECK_THROWS(op_ptr = std::make_unique<ThisOpType>(1.0, 0.0));
    CHECK_FALSE(IsValidPtr(op_ptr));
  }
  SECTION("Copy interface")
  {
    std::unique_ptr<ThisOpType> clone_ptr = nullptr;
    REQUIRE_NOTHROW(clone_ptr = ThisOpType(1.0, 3.0).clone());
    CHECK(clone_ptr->get_min() == El::To<DataType>(1.0));
    CHECK(clone_ptr->get_max() == El::To<DataType>(3.0));

    ThisOpType op(0.0, 1.0);
    REQUIRE_NOTHROW(op = *clone_ptr);

    CHECK(op.get_min() == El::To<DataType>(1.0));
    CHECK(op.get_max() == El::To<DataType>(3.0));
  }
  SECTION("Construct from protobuf")
  {
    constexpr auto D = DeviceAlloc<ThisOpType>;
    lbann_data::Operator proto_op;
    ThisOpType(-2.0, 5.0).write_proto(proto_op);

    std::unique_ptr<BaseOperatorType<ThisOpType>> base_ptr = nullptr;
    REQUIRE_NOTHROW(
      base_ptr =
        lbann::proto::construct_operator<DataType, DataType, D>(proto_op));
    CHECK(base_ptr->get_type() == "clamp");

    auto* specific_ptr = dynamic_cast<ThisOpType*>(base_ptr.get());
    CHECK((bool)specific_ptr);
    CHECK(specific_ptr->get_min() == El::To<DataType>(-2.0));
    CHECK(specific_ptr->get_max() == El::To<DataType>(5.0));
  }
}

TEMPLATE_LIST_TEST_CASE("Clamp operator action",
                        "[mpi][operator][math][clamp][action]",
                        AllClampOpTypes)
{
  using ThisOpType = TestType;
  using DataType = ValueType<ThisOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();

  // Some common data
  ThisOpType op(El::To<DataType>(-1.0), El::To<DataType>(1.0));

  El::Int const height = 13;
  El::Int const width = 17;
  DataParallelMatType<ThisOpType> input(height, width, g, 0),
    output(height, width, g, 0), grad_wrt_output(height, width, g, 0),
    grad_wrt_input(height, width, g, 0), true_output(height, width, g, 0),
    true_grad_wrt_input(height, width, g, 0);

  SECTION("Data parallel - all values in range")
  {
    // Setup inputs/outputs
    El::MakeUniform(input);
    true_output = input; // Operator has no effect.

    El::MakeUniform(grad_wrt_output);
    true_grad_wrt_input = grad_wrt_output;

    El::Fill(output, El::To<DataType>(2.0));         // Fill out of range.
    El::Fill(grad_wrt_input, El::To<DataType>(4.0)); // Fill out of range.

    CHECK_FALSE(true_output == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(true_output == output);

    REQUIRE_NOTHROW(
      op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
    CHECK(true_grad_wrt_input == grad_wrt_input);
  }

  SECTION("Data parallel - all values out of range")
  {
    // Setup inputs/outputs
    El::MakeUniform(input, El::To<DataType>(4), El::To<DataType>(1));
    El::Fill(output, El::To<DataType>(-2.0));
    El::Fill(true_output, El::To<DataType>(1.0));

    El::MakeUniform(grad_wrt_output);
    El::Fill(grad_wrt_input, El::To<DataType>(-1.0));
    El::Fill(true_grad_wrt_input, El::To<DataType>(0.0));

    CHECK_FALSE(true_output == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(true_output == output);

    REQUIRE_NOTHROW(
      op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
    CHECK(true_grad_wrt_input == grad_wrt_input);
  }
}

TEMPLATE_LIST_TEST_CASE("Clamp operator serialization",
                        "[mpi][operator][math][clamp][serialize]",
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
  ThisOpType src_operator(1.f, 2.f), tgt_operator(0.f, 1.f);
  BaseOpPtr src_operator_ptr = std::make_unique<ThisOpType>(3.f, 4.f),
            tgt_operator_ptr;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(src_operator));
      REQUIRE_NOTHROW(oarchive(src_operator_ptr));
    }

    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(tgt_operator));
      REQUIRE_NOTHROW(iarchive(tgt_operator_ptr));
      CHECK(IsValidPtr(tgt_operator_ptr));
    }

    // Check the by-value serialization.
    CHECK(tgt_operator.get_min() == src_operator.get_min());
    CHECK(tgt_operator.get_max() == src_operator.get_max());

    // Check the by-base-ptr serialization.
    ThisOpType const& src_op =
      dynamic_cast<ThisOpType const&>(*src_operator_ptr);
    ThisOpType const& tgt_op =
      dynamic_cast<ThisOpType const&>(*tgt_operator_ptr);
    CHECK(tgt_op.get_min() == src_op.get_min());
    CHECK(tgt_op.get_max() == src_op.get_max());
  }

  SECTION("Rooted binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src_operator));
      REQUIRE_NOTHROW(oarchive(src_operator_ptr));
    }

    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(tgt_operator));
      REQUIRE_NOTHROW(iarchive(tgt_operator_ptr));
      CHECK(IsValidPtr(tgt_operator_ptr));
    }

    // Check the by-value serialization.
    CHECK(tgt_operator.get_min() == src_operator.get_min());
    CHECK(tgt_operator.get_max() == src_operator.get_max());

    // Check the by-base-ptr serialization.
    ThisOpType const& src_op =
      dynamic_cast<ThisOpType const&>(*src_operator_ptr);
    ThisOpType const& tgt_op =
      dynamic_cast<ThisOpType const&>(*tgt_operator_ptr);
    CHECK(tgt_op.get_min() == src_op.get_min());
    CHECK(tgt_op.get_max() == src_op.get_max());
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(src_operator));
      REQUIRE_NOTHROW(oarchive(src_operator_ptr));
    }

    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(tgt_operator));
      REQUIRE_NOTHROW(iarchive(tgt_operator_ptr));
      CHECK(IsValidPtr(tgt_operator_ptr));
    }

    // Check the by-value serialization.
    CHECK(tgt_operator.get_min() == src_operator.get_min());
    CHECK(tgt_operator.get_max() == src_operator.get_max());

    // Check the by-base-ptr serialization.
    ThisOpType const& src_op =
      dynamic_cast<ThisOpType const&>(*src_operator_ptr);
    ThisOpType const& tgt_op =
      dynamic_cast<ThisOpType const&>(*tgt_operator_ptr);
    CHECK(tgt_op.get_min() == src_op.get_min());
    CHECK(tgt_op.get_max() == src_op.get_max());
  }

  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src_operator));
      REQUIRE_NOTHROW(oarchive(src_operator_ptr));
    }

    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(tgt_operator));
      REQUIRE_NOTHROW(iarchive(tgt_operator_ptr));
      CHECK(IsValidPtr(tgt_operator_ptr));
    }

    // Check the by-value serialization.
    CHECK(tgt_operator.get_min() == src_operator.get_min());
    CHECK(tgt_operator.get_max() == src_operator.get_max());

    // Check the by-base-ptr serialization.
    ThisOpType const& src_op =
      dynamic_cast<ThisOpType const&>(*src_operator_ptr);
    ThisOpType const& tgt_op =
      dynamic_cast<ThisOpType const&>(*tgt_operator_ptr);
    CHECK(tgt_op.get_min() == src_op.get_min());
    CHECK(tgt_op.get_max() == src_op.get_max());
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
