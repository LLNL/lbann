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
// lbann_image_utils .cpp .hpp - Image I/O utility functions
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_IMAGE_UTILS_HPP
#define LBANN_IMAGE_UTILS_HPP

#ifdef __LIB_OPENCV
#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#endif

namespace lbann
{
  class image_utils
  {
  public:
    static bool loadBMP(const char* Imagefile, int& Width, int& Height, int& BPP, bool Flip, unsigned char*& Pixels);
    static bool saveBMP(const char* Imagefile, int Width, int Height, int BPP, bool Flip, unsigned char* Pixels);
    
    static bool loadPGM(const char* Imagefile, int& Width, int& Height, int& BPP, bool Flip, unsigned char*& Pixels);
    static bool savePGM(const char* Imagefile, int Width, int Height, int BPP, bool Flip, unsigned char* Pixels);
    static bool loadPNG(const char* Imagefile, int& Width, int& Height, bool Flip, unsigned char*& Pixels);
    static bool savePNG(const char* Imagefile, int Width, int Height, bool Flip, unsigned char* Pixels);

    static bool loadJPG(const char* Imagefile, int& Width, int& Height, bool Flip, unsigned char*& Pixels);
    static bool saveJPG(const char* Imagefile, int Width, int Height, bool Flip, unsigned char* Pixels);

  #ifdef __LIB_OPENCV
    template<typename T = uint8_t, int NCh = 3>
    static bool copy_cvMat_to_buf_with_full_info(const cv::Mat& image, std::vector<uint8_t>& buf, const bool vFlip);

    template<typename T = uint8_t>
    static bool copy_cvMat_to_buf_with_known_type(const cv::Mat& image, std::vector<uint8_t>& buf, const bool vFlip);

    /** Copy a cv::Mat image into a serialized buffer.
     *  The argument vFlip specifies whether to vertically flip the image while
     *  coping. Returns true if successful.
     */ 
    static bool copy_cvMat_to_buf(const cv::Mat& image, std::vector<uint8_t>& buf, const bool vFlip);

    template<typename T = uint8_t, int NCh = 3>
    static cv::Mat copy_buf_to_cvMat_with_full_info(const std::vector<uint8_t>& buf, const int Width, const int Height, const bool vFlip);

    template<typename T = uint8_t>
    static cv::Mat copy_buf_to_cvMat_with_known_type(const std::vector<uint8_t>& buf, const int Width, const int Height, const bool vFlip);

    /** Reconstruct a cv::Mat image from a serialized buffer.
     *  The image size is specified by Width and Height. The last argument vFlip
     *  specifies whether to vertically flip the image while coping. Returns an
     *  emptyimage if unsuccessful.
     */ 
    static cv::Mat copy_buf_to_cvMat(const std::vector<uint8_t>& buf, const int Width, const int Height, const int Depth, const bool vFlip);
  #endif // __LIB_OPENCV
  };


#ifdef __LIB_OPENCV
/**
 * Copy a cv::Mat image into a serialized buffer. This requires the type of
 * pixel intensity values and the number of channels in the image to be known at
 * compile time. The default for these are the type uint8_t and 3 channels. 
 * The last argument vFlip specifies whether to vertically flip the image while
 * coping. Returns true if successful.
 */
template<typename T, int NCh>
inline bool image_utils::copy_cvMat_to_buf_with_full_info(const cv::Mat& image, std::vector<uint8_t>& buf, const bool vFlip)
{
  if (image.empty()) return false;

  const int Width = image.cols;
  const int Height = image.rows;
  const int sz = Height*Width;

  typedef cv::Vec<T, NCh> Vec_T;

  buf.resize(Height*Width*NCh*sizeof(T));
  T* Pixels = reinterpret_cast<T*>(&(buf[0]));

  for (int y = 0; y < Height; ++y) {
    for (int x = 0; x < Width; ++x) {
      const Vec_T pixel = image.at<Vec_T>(y, x);
      const int offset = (vFlip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);

      T* px_ptr = Pixels + offset;
      for(int ch = 0; ch < NCh; ++ch, px_ptr += sz) {
        *px_ptr = pixel[ch];
      }
    }
  }

  return true;
}

/**
 * Copy a cv::Mat image into a serialized buffer. This requires the type of
 * pixel intensity values to be known at compile time. The default type is uint8_t.
 * The last argument vFlip specifies whether to vertically flip the image while
 * coping. Returns true if successful.
 */
template<typename T>
inline bool image_utils::copy_cvMat_to_buf_with_known_type(const cv::Mat& image, std::vector<uint8_t>& buf, const bool vFlip)
{
  switch(image.channels()) {
    case 1: return copy_cvMat_to_buf_with_full_info<T,1>(image, buf, vFlip);
    case 3: return copy_cvMat_to_buf_with_full_info<T,3>(image, buf, vFlip);
  }
  return false;
}

/**
 * Reconstruct a cv::Mat image from a serialized buffer. This requires the type of
 * pixel intensity values and the number of channels in the image to be known at
 * compile time. The default for these are the type uint8_t and 3 channels. 
 * The image size is specified by Width and Height. The last argument vFlip
 * specifies whether to vertically flip the image while coping. Returns an empty
 * image if unsuccessful.
 */
template<typename T, int NCh>
inline cv::Mat image_utils::copy_buf_to_cvMat_with_full_info(const std::vector<uint8_t>& buf, const int Width, const int Height, const bool vFlip)
{
  using namespace lbann::patchworks;
  typedef cv_image_type<T, NCh> Img_T;
  typedef cv::Vec<T, NCh> Vec_T;

  const int sz = Height*Width;

  if (sz*NCh*sizeof(T) != buf.size()) return cv::Mat();

  const T* Pixels = reinterpret_cast<const T*>(&(buf[0]));

  cv::Mat image = cv::Mat(Height, Width, Img_T::T());

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      Vec_T pixel;
      const int offset = (vFlip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      const T* px_ptr = Pixels + offset;
      for(int ch = 0; ch < NCh; ++ch, px_ptr += sz) {
        pixel[ch] = *px_ptr;
      }
      image.at<Vec_T>(y, x) = pixel;
    }
  }

  return image;
}

/**
 * Reconstrut a cv::Mat image from a serialized buffer. This requires the type of
 * pixel intensity values to be known at compile time. The default type is uint8_t.
 * The image size is specified by Width and Height. The last argument vFlip
 * specifies whether to vertically flip the image while coping. Returns an empty
 * image if unsuccessful.
 */
template<typename T>
inline cv::Mat image_utils::copy_buf_to_cvMat_with_known_type(const std::vector<uint8_t>& buf, const int Width, const int Height, const bool vFlip)
{
  const size_t S = static_cast<size_t>(Width*Height*sizeof(T));
  const size_t NCh = buf.size()/S;

  if (S*NCh != buf.size()) return cv::Mat();

  switch(NCh) {
    case 1u: return copy_buf_to_cvMat_with_full_info<T,1>(buf, Width, Height, vFlip);
    case 3u: return copy_buf_to_cvMat_with_full_info<T,3>(buf, Width, Height, vFlip);
  }
  return cv::Mat();
}
#endif // __LIB_OPENCV

}

#endif // LBANN_IMAGE_UTILS_HPP
