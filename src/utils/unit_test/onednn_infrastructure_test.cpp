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

// MUST include this
#include "Catch2BasicSupport.hpp"

#include <lbann/utils/dnn_lib/onednn.hpp>

using namespace lbann;
using namespace lbann::onednn;

using namespace dnnl;
TEST_CASE("oneDNN data type mapping", "[onednn][utilities]")
{
  REQUIRE(get_data_type<float>() == memory::data_type::f32);
  REQUIRE(get_data_type<int32_t>() == memory::data_type::s32);
  REQUIRE(get_data_type<int8_t>() == memory::data_type::s8);
  REQUIRE(get_data_type<uint8_t>() == memory::data_type::u8);

#ifdef LBANN_HAS_HALF
  REQUIRE(get_data_type<cpu_fp16>() == memory::data_type::f16);
#endif // LBANN_HAS_HALF
#if defined LBANN_HAS_GPU_FP16
  REQUIRE(get_data_type<fp16>() == memory::data_type::f16);
#endif // LBANN_HAS_GPU_FP16
}
