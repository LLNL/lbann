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
// image_utils .cpp .hpp - Image I/O utility functions
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_IMAGE_UTILS_HPP
#define LBANN_IMAGE_UTILS_HPP

#include "lbann/base.hpp"
#include <type_traits>
#include <typeinfo>   // operator typeid

#ifdef LBANN_HAS_OPENCV
#include "lbann/data_readers/cv_utils.hpp"
#include "lbann/data_readers/cv_process_patches.hpp"
#endif


namespace lbann {
class image_utils {
 public:
  static bool loadIMG(std::vector<unsigned char>& image_buf, int& Width, int& Height, bool Flip, unsigned char *&Pixels);
  static bool loadIMG(const std::string& Imagefile, int& Width, int& Height, bool Flip, unsigned char *&Pixels, std::vector<char>& buf);
  static bool saveIMG(const std::string& Imagefile, int Width, int Height, bool Flip, unsigned char *Pixels);

#ifdef LBANN_HAS_OPENCV
  // The other load/import methods rely on these core methods
  /// process an image and put it into an LBANN Mat data block
  static bool process_image(cv::Mat& image, int& Width, int& Height, int& Type, cv_process& pp, CPUMat& out);
  /// process an image and put it into a serialized buffer
  static bool process_image(cv::Mat& image, int& Width, int& Height, int& Type, cv_process& pp, std::vector<uint8_t>& out);
  /// process an image and put it into an LBANN Mat data blocks
  static bool process_image(cv::Mat& image, int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& out);
#endif // LBANN_HAS_OPENCV

  // new function, to support sharded data reader and data store functionality
  static bool load_image(std::vector<unsigned char>& image_buf, int& Width, int& Height, int& Type, cv_process& pp, CPUMat& data, cv::Mat* cv_buf = nullptr);

  // new function, to support sharded data reader and data store functionality
  static bool load_image(std::vector<unsigned char>& image_buf,
                         int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& data, cv::Mat* cv_buf = nullptr);

  // load/save an image into/from an LBANN data block of El::Matrix<DataType> type
  // Use a thread save temporary buffer for decoding the image
  /// Load an image from a file and put it into an LBANN Mat data block
  static bool load_image(const std::string& filename, int& Width, int& Height, int& Type, cv_process& pp, CPUMat& data, std::vector<char>& buf, cv::Mat* cv_buf = nullptr);
  /// Load an image from a file, extract patches from it and put them into LBANN Mat data blocks
  static bool load_image(const std::string& filename, int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& data, std::vector<char>& buf, cv::Mat* cv_buf = nullptr);
  /// Save an image using data from an LBANN Mat data block
  static bool save_image(const std::string& filename, const int Width, const int Height, const int Type, cv_process& pp, const CPUMat& data);

  // import/export via a buffer of std::vector<uchar> containg the raw bytes of an image file
  /// Import an image from a file buffer (inbuf) and put it into an LBANN Mat data block
  static bool import_image(cv::InputArray inbuf, int& Width, int& Height, int& Type, cv_process& pp, CPUMat& data, cv::Mat* cv_buf = nullptr);
  /// Import an image from a file buffer (inbuf), extract patches from it and put them into LBANN Mat data blocks
  static bool import_image(cv::InputArray inbuf, int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& data, cv::Mat* cv_buf = nullptr);
  /// Export an image using data from an LBANN Mat block into a file buffer (outbuf)
  static bool export_image(const std::string& fileExt, std::vector<uchar>& outbuf, const int Width, const int Height, const int Type, cv_process& pp, const CPUMat& data);
};

} // end of namespace lbann

#endif // LBANN_IMAGE_UTILS_HPP
