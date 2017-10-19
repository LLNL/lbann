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
// lbann_cv_utils .cpp .hpp - operations related to opencv images
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_utils.hpp"
//#include <iostream>

#ifdef __LIB_OPENCV
namespace lbann {

/** The mathematical constant (this is the way to get it in C++). */
const float cv_transform::pi = static_cast<float>(std::acos(-1));


/// returns the number of bytes that would be used for the image without compresstion and any header
size_t cv_utils::image_data_amount(const cv::Mat& img) {
  return static_cast<size_t>(CV_ELEM_SIZE(img.depth())*
                             CV_MAT_CN(img.type())*
                             img.cols*img.rows);
}


bool cv_utils::copy_cvMat_to_buf(const cv::Mat& image, std::vector<uint8_t>& buf, const cv_process& pp) {
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  _SWITCH_CV_FUNC_3PARAMS(image.depth(), \
                          copy_cvMat_to_buf_with_known_type, \
                          image, buf, pp)
  return false;
}


cv::Mat cv_utils::copy_buf_to_cvMat(const std::vector<uint8_t>& buf,
                                    const int Width, const int Height, const int Type, const cv_process& pp) {
  _LBANN_MILD_EXCEPTION(buf.size() != \
                        static_cast<size_t>(Width * Height * CV_MAT_CN(Type) * CV_ELEM_SIZE(CV_MAT_DEPTH(Type))), \
                        "Size mismatch: Buffer has " << buf.size() << " items when " \
                        << static_cast<size_t>(Width * Height * CV_MAT_CN(Type) * CV_ELEM_SIZE(CV_MAT_DEPTH(Type))) \
                        << " are expected.", \
                        cv::Mat())

  _SWITCH_CV_FUNC_4PARAMS(CV_MAT_DEPTH(Type), \
                          copy_buf_to_cvMat_with_known_type, \
                          buf, Width, Height, pp)

  _LBANN_DEBUG_MSG("Unknown image depth: " << CV_MAT_DEPTH(Type));
  return cv::Mat();
}


bool cv_utils::copy_cvMat_to_buf(const cv::Mat& image, ::Mat& buf, const cv_process& pp) {
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  _SWITCH_CV_FUNC_3PARAMS(image.depth(), \
                          copy_cvMat_to_buf_with_known_type, \
                          image, buf, pp)
  return false;
}


cv::Mat cv_utils::copy_buf_to_cvMat(const ::Mat& buf,
                                    const int Width, const int Height, const int Type, const cv_process& pp) {
  _SWITCH_CV_FUNC_4PARAMS(CV_MAT_DEPTH(Type), \
                          copy_buf_to_cvMat_with_known_type, \
                          buf, Width, Height, pp)

  _LBANN_DEBUG_MSG("Unknown image depth: " << CV_MAT_DEPTH(Type));
  return cv::Mat();
}

std::ostream& operator<<(std::ostream& os, const cv_transform& tr) {
  tr.print(os);
  return os;
}

} // end of namespace lbann
#endif // __LIB_OPENCV
