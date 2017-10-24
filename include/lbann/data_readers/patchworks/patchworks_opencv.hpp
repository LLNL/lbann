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
 *  - cv::Mat pixel type handling mechanisms
 */

#ifdef __LIB_OPENCV
#ifndef _PATCHWORKS_OPENCV_H_INCLUDED_
#define _PATCHWORKS_OPENCV_H_INCLUDED_
#include "lbann/data_readers/opencv.hpp"

namespace lbann {
namespace patchworks {

/// A template structure to convert an OpenCV identifier of channel depth to a standard C++ type
template<int T> struct cv_depth_type {};

/// Convert CV_8U to uint8_t
template<> struct cv_depth_type<CV_8U>  {
  typedef  uint8_t standard_type;
};
/// Convert CV_8S to int8_t
template<> struct cv_depth_type<CV_8S>  {
  typedef   int8_t standard_type;
};
/// Convert CV_16U to uint16_t
template<> struct cv_depth_type<CV_16U> {
  typedef uint16_t standard_type;
};
/// Convert CV_16S to int16_t
template<> struct cv_depth_type<CV_16S> {
  typedef  int16_t standard_type;
};
/// Convert CV_32S to int32_t
template<> struct cv_depth_type<CV_32S> {
  typedef  int32_t standard_type;
};
/// Convert CV_32F to float
template<> struct cv_depth_type<CV_32F> {
  typedef    float standard_type;
};
/// Convert CV_64F to double
template<> struct cv_depth_type<CV_64F> {
  typedef   double standard_type;
};

/// Convert an OpenCV identifier of image depth to a standard C++ type
#define _depth_type(_cv_depth_) lbann::patchworks::cv_depth_type<_cv_depth_>::standard_type


/** A template structure to map the type of channel and the number of
 * channels into the corresponding OpenCV type identifier of such an image.
 */
template<typename T, int NCh> struct cv_image_type {};

/** A template structure to map a standard c++ type of channel
 *  into the corresponding OpenCV type identifier.
 */
template<typename T> struct cv_channel_type {};

/** Define a template struct cv_image_type<_T_,_C_> that contains a static member
 * function T() which returns an OpenCV image type based on the macro arguments:
 * - _B_: The number of bits of a color depth (i.e., the number of bits for _T_)
 * - _S_: S/U/F for signed/unsigned/floating point respectively
 * - _T_: The intensity value type
 * - _C_: The number of channels
 */
#define _def_cv_image_type(_B_, _S_, _T_, _C_) \
template<> struct cv_image_type< _T_ , _C_ > \
{ static int T() { return CV_ ## _B_ ## _S_ ## C ## _C_; } }

/** Define a template struct cv_channel_type<_T_> that contains a static member
 * function T() which returns an OpenCV channel type based on the macro arguments:
 * - _B_: The number of bits of a color depth (i.e., the number of bits for _T_)
 * - _S_: S/U/F for signed/unsigned/floating point respectively
 * - _T_: The intensity value type
 */
#define _def_cv_channel_type(_B_, _S_, _T_) \
template<> struct cv_channel_type< _T_ > \
{ static int T() { return CV_ ## _B_ ## _S_; } }

/// Define cv_image_type<_T_,*> for various number of channels and cv_channel_type<_T_>
#define _def_cv_image_type_B_U(_B_,_S_,_T_) \
        _def_cv_channel_type(_B_, _S_, _T_); \
        _def_cv_image_type(_B_, _S_, _T_, 3); \
        _def_cv_image_type(_B_, _S_, _T_, 1); \
        _def_cv_image_type(_B_, _S_, _T_, 2); \
        _def_cv_image_type(_B_, _S_, _T_, 4);

/// Define cv_image_type<uint8_t, x>, of which T() returns CV_8UCx
_def_cv_image_type_B_U(8,  U, uint8_t)
/// Define cv_image_type<int8_t, x>, of which T() returns CV_8SCx
_def_cv_image_type_B_U(8,  S, int8_t)
/// Define cv_image_type<uint16_t, x>, of which T() returns CV_16UCx
_def_cv_image_type_B_U(16, U, uint16_t)
/// Define cv_image_type<int16_t, x>, of which T() returns CV_16SCx
_def_cv_image_type_B_U(16, S, int16_t)
/// Define cv_image_type<int32_t, x>, of which T() returns CV_32SCx
_def_cv_image_type_B_U(32, S, int32_t)
/// Define cv_image_type<float, x>, of which T() returns CV_32FCx
_def_cv_image_type_B_U(32, F, float)
/// Define cv_image_type<double, x>, of which T() returns CV_64FCx
_def_cv_image_type_B_U(64, F, double)


template<typename T>
struct depth_normalzing {
  static double factor() {
    if ((std::is_same<T, float>::value) || (std::is_same<T, double>::value)) {
      return 1.0;
    } else {
      return 1.0/std::numeric_limits<T>::max();
    }
  }
};

} // end of namespace patchworks
} // end of namespace lbann

#define _SWITCH_CV_FUNC_KNOWN_TYPE_1PARAM(_SW_CH_,_T_,_FUNC_,_P1_) \
  switch (_SW_CH_) { \
    case 1: return _FUNC_<_T_,1>(_P1_); \
    case 2: return _FUNC_<_T_,2>(_P1_); \
    case 3: return _FUNC_<_T_,3>(_P1_); \
    case 4: return _FUNC_<_T_,4>(_P1_); \
  }

#define _SWITCH_CV_FUNC_KNOWN_TYPE_2PARAMS(_SW_CH_,_T_,_FUNC_,_P1_,_P2_) \
  switch (_SW_CH_) { \
    case 1: return _FUNC_<_T_,1>(_P1_,_P2_); \
    case 2: return _FUNC_<_T_,2>(_P1_,_P2_); \
    case 3: return _FUNC_<_T_,3>(_P1_,_P2_); \
    case 4: return _FUNC_<_T_,4>(_P1_,_P2_); \
  }

#define _SWITCH_CV_FUNC_KNOWN_TYPE_3PARAMS(_SW_CH_,_T_,_FUNC_,_P1_,_P2_,_P3_) \
  switch (_SW_CH_) { \
    case 1: return _FUNC_<_T_,1>(_P1_,_P2_,_P3_); \
    case 2: return _FUNC_<_T_,2>(_P1_,_P2_,_P3_); \
    case 3: return _FUNC_<_T_,3>(_P1_,_P2_,_P3_); \
    case 4: return _FUNC_<_T_,4>(_P1_,_P2_,_P3_); \
  }

#define _SWITCH_CV_FUNC_KNOWN_TYPE_4PARAMS(_SW_CH_,_T_,_FUNC_,_P1_,_P2_,_P3_,_P4_) \
  switch (_SW_CH_) { \
    case 1: return _FUNC_<_T_,1>(_P1_,_P2_,_P3_,_P4_); \
    case 2: return _FUNC_<_T_,2>(_P1_,_P2_,_P3_,_P4_); \
    case 3: return _FUNC_<_T_,3>(_P1_,_P2_,_P3_,_P4_); \
    case 4: return _FUNC_<_T_,4>(_P1_,_P2_,_P3_,_P4_); \
  }

#define _SWITCH_CV_FUNC_1PARAM(_SW_D_,_FUNC_,_P1_) \
  switch (_SW_D_) { \
    case CV_8U : return _FUNC_<_depth_type(CV_8U) >(_P1_); \
    case CV_8S : return _FUNC_<_depth_type(CV_8S) >(_P1_); \
    case CV_16U: return _FUNC_<_depth_type(CV_16U)>(_P1_); \
    case CV_16S: return _FUNC_<_depth_type(CV_16S)>(_P1_); \
    case CV_32S: return _FUNC_<_depth_type(CV_32S)>(_P1_); \
    case CV_32F: return _FUNC_<_depth_type(CV_32F)>(_P1_); \
    case CV_64F: return _FUNC_<_depth_type(CV_64F)>(_P1_); \
  }

#define _SWITCH_CV_FUNC_2PARAMS(_SW_D_,_FUNC_,_P1_,_P2_) \
  switch (_SW_D_) { \
    case CV_8U : return _FUNC_<_depth_type(CV_8U) >(_P1_,_P2_); \
    case CV_8S : return _FUNC_<_depth_type(CV_8S) >(_P1_,_P2_); \
    case CV_16U: return _FUNC_<_depth_type(CV_16U)>(_P1_,_P2_); \
    case CV_16S: return _FUNC_<_depth_type(CV_16S)>(_P1_,_P2_); \
    case CV_32S: return _FUNC_<_depth_type(CV_32S)>(_P1_,_P2_); \
    case CV_32F: return _FUNC_<_depth_type(CV_32F)>(_P1_,_P2_); \
    case CV_64F: return _FUNC_<_depth_type(CV_64F)>(_P1_,_P2_); \
  }

#define _SWITCH_CV_FUNC_3PARAMS(_SW_D_,_FUNC_,_P1_,_P2_,_P3_) \
  switch (_SW_D_) { \
    case CV_8U : return _FUNC_<_depth_type(CV_8U) >(_P1_,_P2_,_P3_); \
    case CV_8S : return _FUNC_<_depth_type(CV_8S) >(_P1_,_P2_,_P3_); \
    case CV_16U: return _FUNC_<_depth_type(CV_16U)>(_P1_,_P2_,_P3_); \
    case CV_16S: return _FUNC_<_depth_type(CV_16S)>(_P1_,_P2_,_P3_); \
    case CV_32S: return _FUNC_<_depth_type(CV_32S)>(_P1_,_P2_,_P3_); \
    case CV_32F: return _FUNC_<_depth_type(CV_32F)>(_P1_,_P2_,_P3_); \
    case CV_64F: return _FUNC_<_depth_type(CV_64F)>(_P1_,_P2_,_P3_); \
  }

#define _SWITCH_CV_FUNC_4PARAMS(_SW_D_,_FUNC_,_P1_,_P2_,_P3_,_P4_) \
  switch (_SW_D_) { \
    case CV_8U : return _FUNC_<_depth_type(CV_8U) >(_P1_,_P2_,_P3_,_P4_); \
    case CV_8S : return _FUNC_<_depth_type(CV_8S) >(_P1_,_P2_,_P3_,_P4_); \
    case CV_16U: return _FUNC_<_depth_type(CV_16U)>(_P1_,_P2_,_P3_,_P4_); \
    case CV_16S: return _FUNC_<_depth_type(CV_16S)>(_P1_,_P2_,_P3_,_P4_); \
    case CV_32S: return _FUNC_<_depth_type(CV_32S)>(_P1_,_P2_,_P3_,_P4_); \
    case CV_32F: return _FUNC_<_depth_type(CV_32F)>(_P1_,_P2_,_P3_,_P4_); \
    case CV_64F: return _FUNC_<_depth_type(CV_64F)>(_P1_,_P2_,_P3_,_P4_); \
  }

#endif // _PATCHWORKS_OPENCV_H_INCLUDED_
#endif // __LIB_OPENCV
