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
#include "lbann/operators/math/abs.hpp"

// Other stuff
#include "lbann/proto/factories.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/operator_factory_impl.hpp"

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
using AbsOperatorAllDevices = h2::meta::TL<
#ifdef LBANN_HAS_GPU
  AbsOperator<T, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  AbsOperator<T, El::Device::CPU>>;

using AllAbsOpTypes = h2::meta::tlist::Append<
#if !defined LBANN_HAS_ROCM
  AbsOperatorAllDevices<El::Complex<float>>,
#ifdef LBANN_HAS_DOUBLE
  AbsOperatorAllDevices<El::Complex<double>>,
#endif // LBANN_HAS_DOUBLE
#endif // LBANN_HAS_ROCM
#ifdef LBANN_HAS_DOUBLE
  AbsOperatorAllDevices<double>,
#endif // LBANN_HAS_DOUBLE
  AbsOperatorAllDevices<float>>;

namespace lbann {
template <typename T, El::Device D>
struct OperatorTraits<AbsOperator<T, D>>
  : OperatorTraits<Operator<T, El::Base<T>, D>>
{
};
} // namespace lbann

// Save some typing.
using unit_test::utilities::IsValidPtr;

TEMPLATE_LIST_TEST_CASE("Abs operator lifecycle",
                        "[mpi][operator][math][abs][lifecycle]",
                        AllAbsOpTypes)
{
  using ThisOpType = TestType;

  SECTION("Construction with valid arguments")
  {
    std::unique_ptr<ThisOpType> op_ptr = nullptr;
    REQUIRE_NOTHROW(op_ptr = std::make_unique<ThisOpType>());
    REQUIRE(IsValidPtr(op_ptr));
  }
  SECTION("Copy interface")
  {
    std::unique_ptr<ThisOpType> clone_ptr = nullptr;
    REQUIRE_NOTHROW(clone_ptr = ThisOpType{}.clone());
    CHECK(clone_ptr->get_type() == "abs");

    ThisOpType op;
    REQUIRE_NOTHROW(op = *clone_ptr);
  }
  SECTION("Construct from protobuf")
  {
    using InputDataType = InputValueType<ThisOpType>;
    using OutputDataType = OutputValueType<ThisOpType>;
    constexpr auto D = Device<ThisOpType>;

    lbann_data::Operator proto_op;
    ThisOpType{}.write_proto(proto_op);

    std::unique_ptr<BaseOperatorType<ThisOpType>> base_ptr = nullptr;
    REQUIRE_NOTHROW(
      base_ptr =
        proto::construct_operator<InputDataType, OutputDataType, D>(proto_op));
    CHECK(base_ptr->get_type() == "abs");

    auto* specific_ptr = dynamic_cast<ThisOpType*>(base_ptr.get());
    CHECK(IsValidPtr(specific_ptr));
  }
}

TEMPLATE_LIST_TEST_CASE("Abs operator action",
                        "[mpi][operator][math][abs][action]",
                        AllAbsOpTypes)
{
  using ThisOpType = TestType;
  using InputDataType = InputValueType<ThisOpType>;
  using OutputDataType = OutputValueType<ThisOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();

  // Some common data
  ThisOpType op;

  El::Int const height = 23;
  El::Int const width = 17;
  InputDataParallelMatType<ThisOpType> input(height, width, g, 0),
    grad_wrt_input(height, width, g, 0),
    true_grad_wrt_input(height, width, g, 0);
  OutputDataParallelMatType<ThisOpType> output(height, width, g, 0),
    grad_wrt_output(height, width, g, 0), true_output(height, width, g, 0);

  SECTION("Data parallel - all values positive real")
  {
    // Setup inputs/outputs
    El::Fill(input, InputDataType{2.f});
    El::Fill(true_output, OutputDataType{2.f});

    El::MakeUniform(grad_wrt_output);
    El::Copy(grad_wrt_output, true_grad_wrt_input);

    El::Fill(output, OutputDataType{-32.f});        // Fill out of range.
    El::Fill(grad_wrt_input, InputDataType{-24.f}); // Fill out of range.

    CHECK_FALSE(true_output == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(true_output == output);

    REQUIRE_NOTHROW(
      op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
    CHECK(true_grad_wrt_input == grad_wrt_input);
  }

  SECTION("Data parallel - all values negative real")
  {
    // Setup inputs/outputs
    El::Fill(input, InputDataType{-2.f});
    El::Fill(true_output, OutputDataType{2.f});

    El::MakeUniform(grad_wrt_output);
    El::Copy(grad_wrt_output, true_grad_wrt_input);
    El::Scale(El::To<InputDataType>(-1.), true_grad_wrt_input);

    El::Fill(output, OutputDataType{-32.f});        // Fill out of range.
    El::Fill(grad_wrt_input, InputDataType{-24.f}); // Fill out of range.

    CHECK_FALSE(true_output == output);
    REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
    CHECK(true_output == output);

    REQUIRE_NOTHROW(
      op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
    CHECK(true_grad_wrt_input == grad_wrt_input);
  }

  // SECTION("Data parallel - all values out of range")
  // {
  //   // Setup inputs/outputs
  //   El::MakeUniform(input, El::To<DataType>(4), El::To<DataType>(1));
  //   El::Fill(output, El::To<DataType>(-2.0));
  //   El::Fill(true_output, El::To<DataType>(1.0));

  //   El::MakeUniform(grad_wrt_output);
  //   El::Fill(grad_wrt_input, El::To<DataType>(-1.0));
  //   El::Fill(true_grad_wrt_input, El::To<DataType>(0.0));

  //   CHECK_FALSE(true_output == output);
  //   REQUIRE_NOTHROW(op.fp_compute({input}, {output}));
  //   CHECK(true_output == output);

  //   REQUIRE_NOTHROW(
  //     op.bp_compute({input}, {grad_wrt_output}, {grad_wrt_input}));
  //   CHECK(true_grad_wrt_input == grad_wrt_input);
  // }
}

TEMPLATE_LIST_TEST_CASE("Abs operator serialization",
                        "[mpi][operator][math][abs][serialize]",
                        AllAbsOpTypes)
{
  using ThisOpType = TestType;
  using BaseOpType = BaseOperatorType<ThisOpType>;
  using BaseOpPtr = std::unique_ptr<BaseOpType>;

  auto& world_comm = unit_test::utilities::current_world_comm();

  auto const& g = world_comm.get_trainer_grid();
  utils::grid_manager mgr(g);

  std::stringstream ss;

  // Create the objects
  ThisOpType src_operator, tgt_operator;
  BaseOpPtr src_operator_ptr = std::make_unique<ThisOpType>(), tgt_operator_ptr;

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
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*src_operator_ptr));
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*tgt_operator_ptr));
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
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*src_operator_ptr));
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*tgt_operator_ptr));
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
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*src_operator_ptr));
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*tgt_operator_ptr));
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
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*src_operator_ptr));
    CHECK_NOTHROW(dynamic_cast<ThisOpType const&>(*tgt_operator_ptr));
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
