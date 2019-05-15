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
//
// profiling .hpp .cpp - Various routines for interfacing with profilers
///////////////////////////////////////////////////////////////////////////////

namespace lbann {

// Colors to use for profiling.
constexpr int num_prof_colors = 20;
// http://there4.io/2012/05/02/google-chart-color-list/
constexpr int prof_colors[num_prof_colors] = {
  0x3366CC, 0xDC3912, 0xFF9900, 0x109618, 0x990099, 0x3B3EAC,
  0x0099C6, 0xDD4477, 0x66AA00, 0xB82E2E, 0x316395, 0x994499,
  0x22AA99, 0xAAAA11, 0x6633CC, 0xE67300, 0x8B0707, 0x329262,
  0x5574A6, 0x3B3EAC};

void prof_start();
void prof_stop();
void prof_region_begin(const char *s, int c, bool sync);
void prof_region_end(const char *s, bool sync);

}  // namespace lbann
