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
//
// profiling .hpp .cpp - Various routines for interfacing with profilers
///////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_UTILS_PROFILING_HPP
#define LBANN_UTILS_PROFILING_HPP

#include <lbann_config.hpp>

#if defined(LBANN_HAS_CALIPER)
#include <caliper/cali.h>
#include <caliper/cali_macros.h>

#define LBANN_CALIPER_MARK_SCOPE(x) \
  CALI_CXX_MARK_SCOPE(x)

#define LBANN_CALIPER_MARK_FUNCTION \
  CALI_CXX_MARK_FUNCTION

#define LBANN_CALIPER_MARK_BEGIN(x) \
  CALI_MARK_BEGIN(x)

#define LBANN_CALIPER_MARK_END(x) \
  CALI_MARK_END(x)

#define LBANN_CALIPER_LOOP_BEGIN(label, desc) CALI_CXX_MARK_LOOP_BEGIN(label, desc)
#define LBANN_CALIPER_LOOP_END(label) CALI_CXX_MARK_LOOP_END(label)
#define LBANN_CALIPER_LOOP_ITER(label, id) CALI_CXX_MARK_LOOP_ITERATION(label, id)

#else
#define LBANN_CALIPER_MARK_SCOPE(x) ((void)0)
#define LBANN_CALIPER_MARK_FUNCTION ((void)0)
#define LBANN_CALIPER_MARK_BEGIN(x) ((void)0)
#define LBANN_CALIPER_MARK_END(x) ((void)0)
#define LBANN_CALIPER_LOOP_BEGIN(...) ((void)0)
#define LBANN_CALIPER_LOOP_END(...) ((void)0)
#define LBANN_CALIPER_LOOP_ITER(...) ((void)0)
#endif

namespace lbann {

#if defined(LBANN_HAS_CALIPER)
void initialize_caliper();
void finalize_caliper();
bool is_caliper_initialized() noexcept;
#endif

// Colors to use for profiling.
constexpr int num_prof_colors = 20;
// http://there4.io/2012/05/02/google-chart-color-list/
constexpr int prof_colors[num_prof_colors] = {
  0x3366CC, 0xDC3912, 0xFF9900, 0x109618, 0x990099, 0x3B3EAC, 0x0099C6,
  0xDD4477, 0x66AA00, 0xB82E2E, 0x316395, 0x994499, 0x22AA99, 0xAAAA11,
  0x6633CC, 0xE67300, 0x8B0707, 0x329262, 0x5574A6, 0x3B3EAC};

void prof_start();
void prof_stop();
void prof_region_begin(const char* s, int c, bool sync);
void prof_region_end(const char* s, bool sync);

/** @brief RAII class for a prof region. */
class ProfRegion
{
public:
  /** @brief Create a prof region using an automatic color. */
  ProfRegion(char const* name, bool sync = false);
  /** @brief Create a prof region with an explicit color. */
  ProfRegion(char const* name, int color, bool sync = false);

  ~ProfRegion();

private:
  char const* m_name;
  bool m_sync;
}; // class ProfRegion

// Using a macro so it's easy to remove if needed.
#define BASIC_PROF_REGION(NAME) ProfRegion _(NAME)

} // namespace lbann
#endif // LBANN_UTILS_PROFILING_HPP
