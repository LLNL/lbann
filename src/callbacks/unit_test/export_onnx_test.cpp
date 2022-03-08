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
///////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>
#include "TestHelpers.hpp"
#include "MPITestHelpers.hpp"

// The code being tested
#include <lbann/callbacks/export_onnx.hpp>

#include "lbann/callbacks/callback.hpp"
#include <google/protobuf/message.h>
#include <lbann/base.hpp>

#include <onnx/onnx_pb.h>

#include <iostream>
#include <memory>


using unit_test::utilities::IsValidPtr;
TEST_CASE("Serializing \"export onnx\" callback",
          "[mpi][callback][serialize][onnx]")
{
  using CallbackType = lbann::callback::export_onnx;

  auto& world_comm = unit_test::utilities::current_world_comm();
  auto const& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  CallbackType callback();

  // FIXME: Testing if onnx is defined? How to do this?
}
