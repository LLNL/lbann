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
// opencv.hpp - LBANN header for opencv
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN header for opencv
 *  - includes OpenCV headers according to the version
 *  - use newer built-in variables in place of the deprecated ones for newer OpenCV
 */

#include "lbann_config.hpp"

#ifdef LBANN_HAS_OPENCV
#ifndef _LBANN_OPENCV_H_INCLUDED_
#define _LBANN_OPENCV_H_INCLUDED_

#include <opencv2/core/version.hpp>
#if (!defined(CV_VERSION_EPOCH) && (CV_VERSION_MAJOR >= 3))
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#define _LBANN_CV_UNCHANGED_ cv::IMREAD_UNCHANGED
#define _LBANN_CV_GRAYSCALE_ cv::IMREAD_GRAYSCALE
#define _LBANN_CV_COLOR_     cv::IMREAD_COLOR
#define _LBANN_CV_ANYDEPTH_  cv::IMREAD_ANYDEPTH
#define _LBANN_CV_ANYCOLOR_  cv::IMREAD_ANYCOLOR
#else
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define _LBANN_CV_UNCHANGED_ CV_LOAD_IMAGE_UNCHANGED
#define _LBANN_CV_GRAYSCALE_ CV_LOAD_IMAGE_GRAYSCALE
#define _LBANN_CV_COLOR_     CV_LOAD_IMAGE_COLOR
#define _LBANN_CV_ANYDEPTH_  CV_LOAD_IMAGE_ANYDEPTH
#define _LBANN_CV_ANYCOLOR_  CV_LOAD_IMAGE_ANYCOLOR
#endif

#define _LBANN_CV_BLUE_  0
#define _LBANN_CV_GREEN_ 1
#define _LBANN_CV_RED_   2

#endif // _LBANN_OPENCV_H_INCLUDED_
#endif // LBANN_HAS_OPENCV
