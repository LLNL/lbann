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
#ifndef LBANN_UNIT_TEST_UTILITIES_CATCH2BASICSUPPORT_HPP_INCLUDED
#define LBANN_UNIT_TEST_UTILITIES_CATCH2BASICSUPPORT_HPP_INCLUDED

/** @file
 *
 *  This header is used to allow easy compile-time switching between
 *  Catch2 v2.* and v3.*.
 *
 *  The v3 support includes the basic test macros and all of the
 *  matchers and generators, as well as the Approx class.. Additional
 *  components may be added piecemeal by the tests that require them
 *  checking if the LBANN_USE_CATCH2_V3 preprocessing macro is
 *  defined.
 */

#ifdef LBANN_USE_CATCH2_V3
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
using Catch::Approx;
#else
#include <catch2/catch.hpp>
using Catch::Contains;
#endif // LBANN_USE_CATCH2_V3
#endif // LBANN_UNIT_TEST_UTILITIES_CATCH2BASICSUPPORT_HPP_INCLUDED
