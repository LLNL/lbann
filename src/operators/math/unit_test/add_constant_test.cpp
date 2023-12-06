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
#include "lbann/operators/math/binary_with_constant.hpp"

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
using AddConstantOperatorAllDevices = h2::meta::TL<
#ifdef LBANN_HAS_GPU
  AddConstantOperator<T, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  AddConstantOperator<T, El::Device::CPU>>;

using AllAddConstantOpTypes = h2::meta::tlist::Append<
#ifdef LBANN_HAS_DOUBLE
  AddConstantOperatorAllDevices<double>,
#endif // LBANN_HAS_DOUBLE
  AddConstantOperatorAllDevices<float>>;

namespace lbann {
template <typename T, El::Device D>
struct OperatorTraits<AddConstantOperator<T, D>>
  : OperatorTraits<Operator<T, T, D>>
{
};
} // namespace lbann

// Save some typing.
using unit_test::utilities::IsValidPtr;

TEMPLATE_LIST_TEST_CASE("AddConstant operator lifecycle",
                        "[mpi][operator][math][addconstant][lifecycle]",
                        AllAddConstantOpTypes)
{
  using ThisOpType = TestType;
  using DataT = InputValueType<ThisOpType>;

  SECTION("Construction with valid arguments")
  {
    std::unique_ptr<ThisOpType> op_ptr = nullptr;
    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>());
    REQUIRE(IsValidPtr(op_ptr));
    REQUIRE(op_ptr->get_constant() == El::To<DataT>(0));

    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>());
    REQUIRE(IsValidPtr(op_ptr));
  }
  SECTION("Copy interface")
  {
    std::unique_ptr<ThisOpType> clone_ptr = nullptr;
    REQUIRE_NOTHROW(clone_ptr = ThisOpType{}.clone());
    REQUIRE(clone_ptr->get_constant() == El::To<DataT>(0));

    ThisOpType op;
    REQUIRE_NOTHROW(op = *clone_ptr);
    REQUIRE(op.get_constant() == El::To<DataT>(0));
  }
  SECTION("Construct from protobuf")
  {
    constexpr auto D = Device<ThisOpType>;
    using InT = InputValueType<ThisOpType>;
    using OutT = OutputValueType<ThisOpType>;

    lbann_data::Operator proto_op;
    ThisOpType{El::To<DataT>(13.f)}.write_proto(proto_op);

    std::unique_ptr<BaseOperatorType<ThisOpType>> base_ptr = nullptr;
    REQUIRE_NOTHROW(base_ptr =
                      proto::construct_operator<InT, OutT, D>(proto_op));
    CHECK(base_ptr->get_type() == "add constant");

    auto* specific_ptr = dynamic_cast<ThisOpType*>(base_ptr.get());
    CHECK(IsValidPtr(specific_ptr));
    CHECK(specific_ptr->get_constant() == El::To<DataT>(13));
  }
}

TEMPLATE_LIST_TEST_CASE("AddConstant operator action",
                        "[mpi][operator][math][addconstant][action]",
                        AllAddConstantOpTypes)
{
  using ThisOpType = TestType;
  using InOutDataType = InputValueType<ThisOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();

  // Some common data
  ThisOpType op(El::To<InOutDataType>(13));

  El::Int const height = 13;
  El::Int const width = 17;

  SECTION("Data parallel")
  {
    InputDataParallelMatType<ThisOpType> input(height, width, g, 0),
      grad_wrt_input(height, width, g, 0),
      true_grad_wrt_input(height, width, g, 0);
    OutputDataParallelMatType<ThisOpType> output(height, width, g, 0),
      grad_wrt_output(height, width, g, 0), true_output(height, width, g, 0);

    // Setup inputs/outputs
    El::Fill(input, El::To<InOutDataType>(1.));
    El::Fill(true_output, El::To<InOutDataType>(14.));

    El::MakeUniform(grad_wrt_output);
    true_grad_wrt_input = grad_wrt_output;

    El::Fill(output, El::To<InOutDataType>(-32.)); // Fill out of range.
    El::Fill(grad_wrt_input,
             El::To<InOutDataType>(-42.)); // Fill out of range.

    CHECK_FALSE(true_output == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(true_output == output);

    REQUIRE_NOTHROW(
      op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
    CHECK(true_grad_wrt_input == grad_wrt_input);
  }

  SECTION("Model parallel")
  {
    InputModelParallelMatType<ThisOpType> input(height, width, g, 0),
      grad_wrt_input(height, width, g, 0),
      true_grad_wrt_input(height, width, g, 0);
    OutputModelParallelMatType<ThisOpType> output(height, width, g, 0),
      grad_wrt_output(height, width, g, 0), true_output(height, width, g, 0);

    // Setup inputs/outputs
    El::Fill(input, El::To<InOutDataType>(3.));
    El::Fill(true_output, El::To<InOutDataType>(16.));

    El::MakeUniform(grad_wrt_output);
    true_grad_wrt_input = grad_wrt_output;

    El::Fill(output, El::To<InOutDataType>(-32.)); // Fill out of range.
    El::Fill(grad_wrt_input,
             El::To<InOutDataType>(-52.)); // Fill out of range.

    CHECK_FALSE(true_output == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(true_output == output);

    REQUIRE_NOTHROW(
      op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
    CHECK(true_grad_wrt_input == grad_wrt_input);
  }
}

TEMPLATE_LIST_TEST_CASE("AddConstant operator serialization",
                        "[mpi][operator][math][addconstant][serialize]",
                        AllAddConstantOpTypes)
{
  using ThisOpType = TestType;
  using BaseOpType = BaseOperatorType<ThisOpType>;
  using BaseOpPtr = std::unique_ptr<BaseOpType>;
  using InOutDataType = InputValueType<ThisOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();

  auto const& g = world_comm.get_trainer_grid();
  utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  ThisOpType src_operator{El::To<InOutDataType>(12.)}, tgt_operator;
  BaseOpPtr src_operator_ptr =
              std::make_unique<ThisOpType>(El::To<InOutDataType>(1.)),
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

    // Check the by-base-ptr serialization.
    auto const* concrete_ptr =
      dynamic_cast<ThisOpType const*>(tgt_operator_ptr.get());
    REQUIRE(IsValidPtr(tgt_operator_ptr));
    CHECK(IsValidPtr(concrete_ptr));
    CHECK(concrete_ptr->get_constant() == El::To<InOutDataType>(1.));
    CHECK(tgt_operator.get_constant() == El::To<InOutDataType>(12.));
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

    // Check the by-base-ptr serialization.
    auto const* concrete_ptr =
      dynamic_cast<ThisOpType const*>(tgt_operator_ptr.get());
    REQUIRE(IsValidPtr(tgt_operator_ptr));
    CHECK(IsValidPtr(concrete_ptr));
    CHECK(concrete_ptr->get_constant() == El::To<InOutDataType>(1.));
    CHECK(tgt_operator.get_constant() == El::To<InOutDataType>(12.));
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

    // Check the by-base-ptr serialization.
    auto const* concrete_ptr =
      dynamic_cast<ThisOpType const*>(tgt_operator_ptr.get());
    REQUIRE(IsValidPtr(tgt_operator_ptr));
    CHECK(IsValidPtr(concrete_ptr));
    CHECK(concrete_ptr->get_constant() == El::To<InOutDataType>(1.));
    CHECK(tgt_operator.get_constant() == El::To<InOutDataType>(12.));
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

    // Check the by-base-ptr serialization.
    auto const* concrete_ptr =
      dynamic_cast<ThisOpType const*>(tgt_operator_ptr.get());
    REQUIRE(IsValidPtr(tgt_operator_ptr));
    CHECK(IsValidPtr(concrete_ptr));
    CHECK(concrete_ptr->get_constant() == El::To<InOutDataType>(1.));
    CHECK(tgt_operator.get_constant() == El::To<InOutDataType>(12.));
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
