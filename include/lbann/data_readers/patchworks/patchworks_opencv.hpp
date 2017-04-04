////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// patchworks_opencv.hpp - LBANN PATCHWORKS header for opencv
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS header for opencv
 *  - includes opencv headers according to the version
 */

#ifndef _PATCHWORKS_OPENCV_H_INCLUDED_
#define _PATCHWORKS_OPENCV_H_INCLUDED_

#include <opencv2/core/version.hpp>
  #if (!defined(CV_VERSION_EPOCH) && (CV_VERSION_MAJOR >= 3))
  #include <opencv2/core.hpp>
  //#include <opencv2/highgui.hpp>
  #include <opencv2/imgproc.hpp>
  #define DEFAULT_CV_WINDOW_KEEPRATIO cv::WINDOW_KEEPRATIO
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/core/core_c.h>
  //#include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
  #define DEFAULT_CV_WINDOW_KEEPRATIO CV_WINDOW_KEEPRATIO
#endif

#endif // _PATCHWORKS_OPENCV_H_INCLUDED_
