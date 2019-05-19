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
// opencv_extensions.hpp - LBANN's cv::Mat pixel type handling mechanisms
////////////////////////////////////////////////////////////////////////////////

#ifdef LBANN_HAS_OPENCV
#ifndef _LBANN_OPENCV_EXTENSIONS_H_INCLUDED_
#define _LBANN_OPENCV_EXTENSIONS_H_INCLUDED_
#include "lbann/data_readers/opencv.hpp"

namespace lbann {

/// A template structure to convert an OpenCV identifier of channel depth to a standard C++ type
template<int T> class cv_depth_type {};

/// define a specialized mapper from a CV channel type to its c++ native type
#define _def_cv_depth_translation(_CV_TYPE_, _NATIVE_TYPE_) \
template<> struct cv_depth_type<_CV_TYPE_>  { \
 public: \
  using standard_type =  _NATIVE_TYPE_; \
}

/// cv_depth_type<CV_8U> maps to uint8_t
_def_cv_depth_translation(CV_8U, uint8_t);
/// cv_depth_type<CV_8S> maps to int8_t
_def_cv_depth_translation(CV_8S, int8_t);
/// cv_depth_type<CV_16U> maps to uint16_t
_def_cv_depth_translation(CV_16U, uint16_t);
/// cv_depth_type<CV_16S> maps to int16_t
_def_cv_depth_translation(CV_16S, int16_t);
/// cv_depth_type<CV_32S> maps to int32_t
_def_cv_depth_translation(CV_32S, int32_t);
/// cv_depth_type<CV_32F> maps to float
_def_cv_depth_translation(CV_32F, float);
/// cv_depth_type<CV_64F> maps to double
_def_cv_depth_translation(CV_64F, double);


/// Convert an OpenCV identifier of image depth to a standard C++ type
#define _depth_type(_cv_depth_) lbann::cv_depth_type<_cv_depth_>::standard_type


/** A template structure to map the type of channel into the
 * corresponding OpenCV type identifier of image.
   * - _T_: The channel value type as a native C++ type
 */
template<typename _T_>
struct cv_image_type {
  /** A static member function which returns the OpenCV image type based on
   *  the channel type and number of channels:
   *  - _C_: The number of channels It ranges from 1 to CV_CN_MAX which is 512
   */
  static int T(const int _C_) {
    return CV_MAKETYPE(cv::DataType<_T_>::depth, _C_);
  }
  /** A static member function which maps a native c++ type to the corresponding
   *  OpenCV channel type.
   *  The depth value returned ranges from 0 to (CV_DEPTH_MAX-1) which is 7
   */
  static int T() {
    return cv::DataType<_T_>::depth;
  }
};


template<typename T>
struct depth_normalization {
  static double factor() {
    if (!std::is_integral<T>::value) {
      return 1.0;
    } else {
      return 1.0/std::numeric_limits<T>::max();
    }
  }
  static double inverse_factor() {
    if (!std::is_integral<T>::value) {
      return 1.0;
    } else {
      return std::numeric_limits<T>::max();
    }
  }
};

template<>
struct depth_normalization<void> {
  static double factor() {
    return 1.0;
  }
  static double inverse_factor() {
    return 1.0;
  }
};

/// Checks if an OpenCV depth code corresponds to an integral type
inline bool is_float(const int cv_depth) {
  return ((cv_depth == CV_64F) || (cv_depth == CV_32F));
}

inline bool check_if_cv_Mat_is_float_type(const cv::Mat& image) {
  return is_float(image.depth());
}

inline bool check_if_cv_Mat_has_same_shape(const cv::Mat& image1, const cv::Mat& image2) {
  return ((image1.cols == image2.cols) &&
          (image1.rows == image2.rows) &&
          (image1.channels() == image2.channels()));
}

template<typename T>
static double depth_norm_factor() {
  return depth_normalization<T>::factor();
}

template<typename T>
static double depth_norm_inverse_factor() {
  return depth_normalization<T>::inverse_factor();
}

/// Return the factor for unit scaling with the type indicated by the OpenCV depth
double get_depth_normalizing_factor(const int cv_depth);
/// Return the factor to inverse the unit scaling
double get_depth_denormalizing_factor(const int cv_depth);

/// returns the number of bytes that would be used for the image without compresstion and any header
inline size_t image_data_amount(const cv::Mat& img) {
  return static_cast<size_t>(CV_ELEM_SIZE(img.depth())*
                             CV_MAT_CN(img.type())*
                             img.cols*img.rows);
}

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

#endif // _LBANN_OPENCV_EXTENSIONS_H_INCLUDED_
#endif // LBANN_HAS_OPENCV
