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
// patchworks_common.hpp - LBANN PATCHWORKS header for common definitions
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS common header
 *  - includes commonly used macros, definitions and declarations
 */

#include "lbann_config.hpp"

#ifdef LBANN_HAS_OPENCV
#ifndef _PATCHWORKS_COMMON_H_
#define _PATCHWORKS_COMMON_H_

#include <utility> // std::pair
#include <limits>
#include <cstdint>
#include <string>
#include "lbann/data_readers/opencv_extensions.hpp"

namespace lbann {
namespace patchworks {

/// Patch displacement type
using displacement_type = std::pair<int, int>;

#if 0
// using 32-bit floating point for intermediate image data processing
using pw_fp_t = float;
using pw_cv_vec3 = cv::Vec3f;
#define _PATCHWORKS_STAT_FLOAT_ 32
#define _PW_CV_FP_ CV_32FC3
#else
// using 64-bit floating point for intermediate image data processing
using pw_fp_t = double;
using pw_cv_vec3 = cv::Vec3d;
#define _PATCHWORKS_STAT_FLOAT_ 64
#define _PW_CV_FP_ CV_64FC3
#endif

} // end of namespace patchworks
} // end of namespace lbann

#endif // _PATCHWORKS_COMMON_H_
#endif // LBANN_HAS_OPENCV
