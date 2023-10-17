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

// Testing framework stuff
#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "MatrixHelpers.hpp"
#include "TestHelpers.hpp"

#include "OperatorTraits.hpp"

// CUT
#include "lbann/operators/math/clamp.hpp"

// Other stuff
#include "lbann/proto/factories.hpp"
#include "lbann/utils/serialize.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

#include "lbann/proto/operators.pb.h"
#include <functional>
#include <memory>
#include <numeric>

using namespace lbann;

// Define the list of operators to test. Basically this is
// {float,double}x{CPU,GPU}.
template <typename T>
using ClampOperatorAllDevices = h2::meta::TL<
#ifdef LBANN_HAS_GPU
  ClampOperator<T, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  ClampOperator<T, El::Device::CPU>>;

using AllClampOpTypes = h2::meta::tlist::Append<
#ifdef LBANN_HAS_DOUBLE
  ClampOperatorAllDevices<double>,
#endif // LBANN_HAS_DOUBLE
  ClampOperatorAllDevices<float>>;

namespace lbann {
template <typename T, El::Device D>
struct OperatorTraits<ClampOperator<T, D>> : OperatorTraits<Operator<T, T, D>>
{
};
} // namespace lbann

// Save some typing.
using unit_test::utilities::IsValidPtr;

TEMPLATE_LIST_TEST_CASE("Clamp operator lifecycle",
                        "[mpi][operator][math][clamp][lifecycle]",
                        AllClampOpTypes)
{
  using ThisOpType = TestType;
  using InOutDataType = InputValueType<ThisOpType>;

  auto AsOkType = [](auto const& x) { return El::To<InOutDataType>(x); };
  SECTION("Construction with valid arguments")
  {
    std::unique_ptr<ThisOpType> op_ptr = nullptr;
    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>(0., 1.));
    REQUIRE(IsValidPtr(op_ptr));
    CHECK(op_ptr->get_min() == AsOkType(0.));
    CHECK(op_ptr->get_max() == AsOkType(1.));

    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>(1., 1.));
    REQUIRE(IsValidPtr(op_ptr));
    CHECK(op_ptr->get_min() == AsOkType(1.));
    CHECK(op_ptr->get_max() == AsOkType(1.));
  }
  SECTION("Construction with invalid arguments")
  {
    std::unique_ptr<ThisOpType> op_ptr = nullptr;
    CHECK_THROWS(op_ptr = std::make_unique<ThisOpType>(1., 0.));
    CHECK_FALSE(IsValidPtr(op_ptr));
  }
  SECTION("Copy interface")
  {
    std::unique_ptr<ThisOpType> clone_ptr = nullptr;
    REQUIRE_NOTHROW(clone_ptr = ThisOpType{1., 3.}.clone());
    CHECK(clone_ptr->get_min() == AsOkType(1.));
    CHECK(clone_ptr->get_max() == AsOkType(3.));

    ThisOpType op(0., 1.);
    REQUIRE_NOTHROW(op = *clone_ptr);

    CHECK(op.get_min() == AsOkType(1.));
    CHECK(op.get_max() == AsOkType(3.));
  }
  SECTION("Construct from protobuf")
  {
    constexpr auto D = Device<ThisOpType>;
    lbann_data::Operator proto_op;
    ThisOpType{-2., 5.}.write_proto(proto_op);

    std::unique_ptr<BaseOperatorType<ThisOpType>> base_ptr = nullptr;
    REQUIRE_NOTHROW(
      base_ptr =
        proto::construct_operator<InOutDataType, InOutDataType, D>(proto_op));
    CHECK(base_ptr->get_type() == "clamp");

    auto* specific_ptr = dynamic_cast<ThisOpType*>(base_ptr.get());
    CHECK(IsValidPtr(specific_ptr));
    CHECK(specific_ptr->get_min() == AsOkType(-2.));
    CHECK(specific_ptr->get_max() == AsOkType(5.));
  }
}

TEMPLATE_LIST_TEST_CASE("Clamp operator action",
                        "[mpi][operator][math][clamp][action]",
                        AllClampOpTypes)
{
  using ThisOpType = TestType;
  using InOutDataType = InputValueType<ThisOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();

  // Some common data
  ThisOpType op(-1., 1.);

  El::Int const height = 13;
  El::Int const width = 17;
  InputDataParallelMatType<ThisOpType> input(height, width, g, 0),
    grad_wrt_input(height, width, g, 0),
    true_grad_wrt_input(height, width, g, 0);
  OutputDataParallelMatType<ThisOpType> output(height, width, g, 0),
    grad_wrt_output(height, width, g, 0), true_output(height, width, g, 0);

  auto AsOkType = [](auto const& x) { return El::To<InOutDataType>(x); };
  SECTION("Data parallel - all values in range")
  {
    // Setup inputs/outputs
    El::MakeUniform(input);
    true_output = input; // Operator has no effect.

    El::MakeUniform(grad_wrt_output);
    true_grad_wrt_input = grad_wrt_output;

    El::Fill(output, AsOkType(2.));         // Fill out of range.
    El::Fill(grad_wrt_input, AsOkType(4.)); // Fill out of range.

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
    El::MakeUniform(input, AsOkType(4), AsOkType(1));
    El::Fill(output, AsOkType(-2.));
    El::Fill(true_output, AsOkType(1.));

    El::MakeUniform(grad_wrt_output);
    El::Fill(grad_wrt_input, AsOkType(-1.));
    El::Fill(true_grad_wrt_input, AsOkType(0.));

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
  utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  ThisOpType src_operator(1., 2.), tgt_operator(0., 1.);
  BaseOpPtr src_operator_ptr = std::make_unique<ThisOpType>(3., 4.),
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
      RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src_operator));
      REQUIRE_NOTHROW(oarchive(src_operator_ptr));
    }

    {
      RootedBinaryInputArchive iarchive(ss, g);
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
      RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(src_operator));
      REQUIRE_NOTHROW(oarchive(src_operator_ptr));
    }

    {
      RootedXMLInputArchive iarchive(ss, g);
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
