////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "MPITestHelpers.hpp"

namespace unit_test {
namespace utilities {
namespace {
lbann::lbann_comm* global_comm_;
}

lbann::lbann_comm& current_world_comm()
{
  LBANN_ASSERT_POINTER(global_comm_);
  return *global_comm_;
}

namespace expert {
void register_world_comm(lbann::lbann_comm& comm) noexcept
{
  global_comm_ = &comm;
}

void reset_world_comm() noexcept
{
  global_comm_ = nullptr;
}
} // namespace expert
} // namespace utilities
} // namespace unit_test
