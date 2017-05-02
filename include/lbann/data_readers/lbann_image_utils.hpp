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
#include "lbann/lbann_base.hpp"
#include <type_traits>

namespace lbann
{
  /** A structure packs the parameters for image preprocessing that takes
   *  advantage of the OpenCV framework.
   */
  struct cvMat_proc_params {
    /// OpenCV flip codes: c<0 for top_left <-> bottom_right, c=0 for top<->down, and c>0 for left<->right
    enum flpping {_both_axes_=-1, _vertical_=0, _horizontal_=1, _none_=2};

    /// Whether to flip an image
    flpping m_flip;

    /// Whether to split channels
    bool m_split;

    //used to compute newvalue = cv::saturate_cast<T>(alpha*value + beta)
    double alpha; ///< for scaling channel values
    double beta; ///< for shifting channel values


    /// Check whether to flip
    bool to_flip(void) const { return (m_flip != _none_); }
    /// Tell how to flip
    int how_to_flip(void) const { return static_cast<int>(m_flip); }
    /// Set the flpping behavior
    void set_to_flip(flpping f) { m_flip = f; }
    /// Set to split channels
    bool to_split(void) const { return m_split; }

    cvMat_proc_params(void) : m_flip(_none_), m_split(true), alpha(1.0), beta(0.0) {}

    cvMat_proc_params(const flpping flip_code, const bool tosplit)
    : m_flip(flip_code), m_split(tosplit), alpha(1.0), beta(0.0) {}
  };


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
    // preprocess (with cv::Mat type image data)
    template<typename T = uint8_t, int NCh = 3>
    static bool preprocess_cvMat_with_full_info(cv::Mat& image, const cvMat_proc_params& pp);

    template<typename T = uint8_t>
    static bool preprocess_cvMat_with_known_type(cv::Mat& image, const cvMat_proc_params& pp);

    static bool preprocess_cvMat(cv::Mat& image, const cvMat_proc_params& pp);


    // postprocess (with cv::Mat type image data)
    template<typename T = uint8_t, int NCh = 3>
    static bool postprocess_cvMat_with_full_info(cv::Mat& image, const cvMat_proc_params& pp);

    template<typename T = uint8_t>
    static bool postprocess_cvMat_with_known_type(cv::Mat& image, const cvMat_proc_params& pp);

    static bool postprocess_cvMat(cv::Mat& image, const cvMat_proc_params& pp);


    // copy_cvMat_to_buf (with a tempoary buffer)
    template<typename T = uint8_t, int NCh = 3>
    static bool copy_cvMat_to_buf_with_full_info(const cv::Mat& image, std::vector<uint8_t>& buf, const cvMat_proc_params& pp);

    template<typename T = uint8_t>
    static bool copy_cvMat_to_buf_with_known_type(const cv::Mat& image, std::vector<uint8_t>& buf, const cvMat_proc_params& pp);

    /** Copy a cv::Mat image into a serialized buffer.
     *  The argument pp specifies the parameters for image preprocessing that
     *  takes advantage of the OpenCV framework. Returns true if successful.
     */
    static bool copy_cvMat_to_buf(const cv::Mat& image, std::vector<uint8_t>& buf, const cvMat_proc_params& pp);


    // copy_buf_to_cvMat (with a tempoary buffer)
    template<typename T = uint8_t, int NCh = 3>
    static cv::Mat copy_buf_to_cvMat_with_full_info(const std::vector<uint8_t>& buf, const int Width, const int Height, const cvMat_proc_params& pp);

    template<typename T = uint8_t>
    static cv::Mat copy_buf_to_cvMat_with_known_type(const std::vector<uint8_t>& buf, const int Width, const int Height, const cvMat_proc_params& pp);

    /** Reconstruct a cv::Mat image from a serialized buffer.
     *  The image size is specified by Width and Height. Type indetifies the
     *  OpenCV image type. The last argument pp specifies the parameters for
     *  image postprocessing that takes advantage of the OpenCV framework.
     *  Returns a reconstructed cv::Mat image if successful and an empty one
     *  otherwise.
     */
    static cv::Mat copy_buf_to_cvMat(const std::vector<uint8_t>& buf, const int Width, const int Height, const int Type, const cvMat_proc_params& pp);


    // copy_buf_to_cvMat (with an El::Matrix<DataType> block)
    template<typename T = DataType, int NCh = 3>
    static bool copy_cvMat_to_buf_with_full_info(const cv::Mat& image, ::Mat& buf, const cvMat_proc_params& pp);

    template<typename T = DataType>
    static bool copy_cvMat_to_buf_with_known_type(const cv::Mat& image, ::Mat& buf, const cvMat_proc_params& pp);

    /** Copy a cv::Mat image into an El::Matrix<DataType> block.
     *  The argument pp specifies the parameters for image preprocessing that
     *  takes advantage of the OpenCV framework. Returns true if successful.
     */
    static bool copy_cvMat_to_buf(const cv::Mat& image, ::Mat& buf, const cvMat_proc_params& pp);


    // copy_buf_to_cvMat (with an El::Matrix<DataType> block)
    template<typename T = DataType, int NCh = 3>
    static cv::Mat copy_buf_to_cvMat_with_full_info(const ::Mat& buf, const int Width, const int Height, const cvMat_proc_params& pp);

    template<typename T = DataType>
    static cv::Mat copy_buf_to_cvMat_with_known_type(const ::Mat& buf, const int Width, const int Height, const cvMat_proc_params& pp);

    /** Reconstruct a cv::Mat image from an El::Matrix<DataType> block.
     *  The image size is specified by Width and Height. Type indetifies the
     *  OpenCV image type. The last argument pp specifies the parameters for
     *  image postprocessing that takes advantage of the OpenCV framework.
     *  Returns a reconstructed cv::Mat image if successful and an empty one
     *  otherwise.
     */
    static cv::Mat copy_buf_to_cvMat(const ::Mat& buf, const int Width, const int Height, const int Type, const cvMat_proc_params& pp);
  #endif // __LIB_OPENCV

    // load/save an image into/from a temporary buffer
    /// Load an image from a file and put it into a serialized buffer
    static bool load_image(const std::string& filename, int& Width, int& Height, int& Type, const cvMat_proc_params& pp, std::vector<uint8_t>& buf);
    /// Save an image from a serialized buffer into a file
    static bool save_image(const std::string& filename, const int Width, const int Height, const int Type, const cvMat_proc_params& pp, const std::vector<uint8_t>& buf);

    // load/save an image into/from an LBANN data block of El::Matrix<DataType> type
    /// Load an image from a file and put it into an LBANN Mat data block
    static bool load_image(const std::string& filename, int& Width, int& Height, int& Type, const cvMat_proc_params& pp, ::Mat& data);
    /// Save an image using data from an LBANN Mat data block
    static bool save_image(const std::string& filename, const int Width, const int Height, const int Type, const cvMat_proc_params& pp, const ::Mat& data);

    // import/export via a buffer of std::vector<uchar> containg the raw bytes of an image file
    /// Import an image from a file buffer (inbuf) and put it into an LBANN Mat data block
    static bool import_image(const std::vector<uchar>& inbuf, int& Width, int& Height, int& Type, const cvMat_proc_params& pp, ::Mat& data);
    /// Export an image using data from an LBANN Mat block into a file buffer (outbuf)
    static bool export_image(const std::string& fileExt, std::vector<uchar>& outbuf, const int Width, const int Height, const int Type, const cvMat_proc_params& pp, const ::Mat& data);
  };


#ifdef LBANN_DEBUG
  #define _LBANN_DEBUG_MSG(_MSG_)
/*
  #define _LBANN_DEBUG_MSG(_MSG_) { \
    std::stringstream err; \
    err << __FILE__ << " " << __LINE__ << " : " << _MSG_ << std::endl; \
    throw lbann_exception(err.str()); \
  }
*/
#else
  #if 1
    #define _LBANN_DEBUG_MSG(_MSG_) \
      std::cerr << __FILE__ << " " << __LINE__ << " : " << _MSG_ << std::endl;
  #else
    #define _LBANN_DEBUG_MSG(_MSG_)
  #endif
#endif


#ifdef __LIB_OPENCV

/**
 * Transform linearly while copying data from one sequential container to another
 * The transformation is alpha*input + beta -> output
 * @param first  The beginning of the input interator
 * @param last   The last of the input iterator
 * @param result The beginning of the output iterator
 * @param alpha  linear transform parameter for scaling
 * @param beta  linear transform parameter for shifting
 * @return the last of output iterator
 */
template<class InputIterator, class OutputIterator>
  OutputIterator scale (InputIterator first, InputIterator last, OutputIterator result, const double alpha, const double beta)
{
  if ((alpha == 1.0) && (beta == 0.0)) {
    if ((std::is_same< typename std::remove_cv<InputIterator>::type, typename std::remove_cv<OutputIterator>::type >::value) &&
        (reinterpret_cast<const void*>(first) == reinterpret_cast<const void*>(result)))
    {
      std::advance(result, std::distance(first,last));
      return result;
    } else {
      return std::copy(first, last, result);
    }
  }

  typedef typename std::iterator_traits<OutputIterator>::value_type T;
  while (first != last) {
    *result = cv::saturate_cast<T>(alpha * (*first) + beta);
    ++result; ++first;
  }
  return result;
}

//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                           preprocess (cv::Mat)
/**
 * The preprocessing place holder function specific to a particular pixel type
 * (in terms of the channel type and the number of channels). Most of the OpenCV
 * routines handel it internally. However, custom routines that access each
 * individual pixel can find such information here. The image given as the first
 * parameter will be modified according to the processing parameters described
 * in the second parameter. If successful, it returns true.
 */
template<typename T, int NCh>
inline bool image_utils::preprocess_cvMat_with_full_info(cv::Mat& image, const cvMat_proc_params& pp)
{
  if (image.empty()) return false;
  //cv::Mat _img = image(cv::Rect(cv::Point2i(4250, 700), cv::Point2i(4500, 900)));
  //image = _img;
  if (pp.to_flip())
    cv::flip(image, image, pp.how_to_flip());
  return true;
}

template<typename T>
inline bool image_utils::preprocess_cvMat_with_known_type(cv::Mat& image, const cvMat_proc_params& pp)
{
  switch(image.channels()) {
    case 1: return preprocess_cvMat_with_full_info<T,1>(image, pp);
    case 2: return preprocess_cvMat_with_full_info<T,2>(image, pp);
    case 3: return preprocess_cvMat_with_full_info<T,3>(image, pp);
    case 4: return preprocess_cvMat_with_full_info<T,4>(image, pp);
  }
  return false;
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                           postprocess (cv::Mat)
/**
 * The postprocessing place holder function specific to a particular pixel type
 * (in terms of the channel type and the number of channels). Most of the OpenCV
 * routines handel it internally. However, custom routines that access each
 * individual pixel can find such information here. The image given as the first
 * parameter will be modified according to the processing parameters described
 * in the second parameter. If successful, it returns true.
 */
template<typename T, int NCh>
inline bool image_utils::postprocess_cvMat_with_full_info(cv::Mat& image, const cvMat_proc_params& pp)
{
  if (image.empty()) return false;
  if (pp.to_flip())
    cv::flip(image, image, pp.how_to_flip());
  return true;
}

template<typename T>
inline bool image_utils::postprocess_cvMat_with_known_type(cv::Mat& image, const cvMat_proc_params& pp)
{
  switch(image.channels()) {
    case 1: return postprocess_cvMat_with_full_info<T,1>(image, pp);
    case 2: return postprocess_cvMat_with_full_info<T,2>(image, pp);
    case 3: return postprocess_cvMat_with_full_info<T,3>(image, pp);
    case 4: return postprocess_cvMat_with_full_info<T,4>(image, pp);
  }
  return false;
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                       copy_cvMat_to_buf (vector<uchar>)
/**
 * Copy a cv::Mat image into a serialized buffer. This requires the type of
 * channel values and the number of channels in the image to be known at
 * compile time. The default for these are the type uint8_t and 3 channels.
 * The argument pp specifies the parameters for image preprocessing that
 * takes advantage of the OpenCV framework. Returns true if successful.
 */
template<typename T, int NCh>
inline bool image_utils::copy_cvMat_to_buf_with_full_info(const cv::Mat& image, std::vector<uint8_t>& buf, const cvMat_proc_params& pp)
{
  if (image.empty()) return false;

  const int Width = image.cols;
  const int Height = image.rows;
  const int sz = Height*Width;

  buf.resize(sz*NCh*sizeof(T));
  T* Pixels = reinterpret_cast<T*>(&(buf[0]));

#if 0
  // If this path is enabled, disable flip() inside of preprocess_cvMat()
  typedef cv::Vec<T, NCh> Vec_T;

  for (int y = 0; y < Height; ++y) {
    for (int x = 0; x < Width; ++x) {
      const Vec_T pixel = image.at<Vec_T>(y, x);
      const int offset = (pp.m_flip == cvMat_proc_params::_vertical_) ? ((Height - 1 - y) * Width + x) : (y * Width + x);

      T* px_ptr = Pixels + offset;
      for(int ch = 0; ch < NCh; ++ch, px_ptr += sz) {
        *px_ptr = pixel[ch];
      }
    }
  }
#else
  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);
    for(size_t ch=0; ch < NCh; ++ch, Pixels += sz)
      channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1), Pixels);
    cv::split(image, channels);
    Pixels = reinterpret_cast<T*>(&(buf[0]));
    scale(Pixels, Pixels + sz*NCh, Pixels, pp.alpha, pp.beta);
  } else {
    if (image.isContinuous()) {
      scale(reinterpret_cast<const T*>(image.datastart), reinterpret_cast<const T*>(image.dataend), Pixels, pp.alpha, pp.beta);
    } else {
      const int stride = Width*NCh;
      for (int i = 0; i < Height; ++i, Pixels += stride) {
        const T* ptr = reinterpret_cast<const T*>(image.ptr<const T>(i));
        scale(ptr, ptr+stride, Pixels, pp.alpha, pp.beta);
      }
    }
  }
#endif

  return true;
}

/**
 * Copy a cv::Mat image into a serialized buffer. This requires the type of
 * channel values to be known at compile time. The default type is uint8_t.
 * The argument pp specifies the parameters for image preprocessing that
 * takes advantage of the OpenCV framework. Returns true if successful.
 */
template<typename T>
inline bool image_utils::copy_cvMat_to_buf_with_known_type(const cv::Mat& image, std::vector<uint8_t>& buf, const cvMat_proc_params& pp)
{
  switch(image.channels()) {
    case 1: return copy_cvMat_to_buf_with_full_info<T,1>(image, buf, pp);
    case 2: return copy_cvMat_to_buf_with_full_info<T,2>(image, buf, pp);
    case 3: return copy_cvMat_to_buf_with_full_info<T,3>(image, buf, pp);
    case 4: return copy_cvMat_to_buf_with_full_info<T,4>(image, buf, pp);
  }
  return false;
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                       copy_buf_to_cvMat (vector<uchar>)
/**
 * Reconstruct a cv::Mat image from a serialized buffer. This requires the type
 * of channel values and the number of channels in the image to be known at
 * compile time. The default for these are the type uint8_t and 3 channels.
 * The image size is specified by Width and Height. The argument pp specifies
 * the parameters for image postprocessing that takes advantage of the OpenCV
 * framework. Returns an empty image if unsuccessful.
 */
template<typename T, int NCh>
inline cv::Mat image_utils::copy_buf_to_cvMat_with_full_info(const std::vector<uint8_t>& buf, const int Width, const int Height, const cvMat_proc_params& pp)
{
  using namespace lbann::patchworks;
  //typedef cv_image_type<T, NCh> Img_T;

  const int sz = Height*Width;

  if (sz*NCh*sizeof(T) != buf.size()) {
    _LBANN_DEBUG_MSG("Size mismatch. Buffer has " << buf.size() \
                     << " items when " << sz*NCh*sizeof(T) << " are expected.");
    return cv::Mat();
  }

  const T* Pixels = reinterpret_cast<const T*>(&(buf[0]));

  //cv::Mat image = cv::Mat(Height, Width, Img_T::T());
  cv::Mat image = cv::Mat(Height, Width, CV_MAKETYPE(cv::DataType<T>::depth, NCh));

#if 0
  typedef cv::Vec<T, NCh> Vec_T;

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      Vec_T pixel;
      const int offset = (pp.m_flip == cvMat_proc_params::_vertical_) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      const T* px_ptr = Pixels + offset;
      for(int ch = 0; ch < NCh; ++ch, px_ptr += sz) {
        pixel[ch] = *px_ptr;
      }
      image.at<Vec_T>(y, x) = pixel;
    }
  }
  // if you decide use this path, make cvMat_proc_params::m_flip mutable, and then
  // set pp.m_flip = _none_ here as to avoid redundant post processing
#else
  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);

    for(size_t ch=0; ch < NCh; ++ch, Pixels += sz)
      channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1), const_cast<T*>(Pixels));

    cv::merge(channels, image);
    scale(reinterpret_cast<const T*>(image.datastart), reinterpret_cast<const T*>(image.dataend), reinterpret_cast<T*>(image.data), pp.alpha, pp.beta);
  } else {
    scale(Pixels, Pixels + sz*NCh, reinterpret_cast<T*>(image.data), pp.alpha, pp.beta);
  }
#endif

  return image;
}

/**
 * Reconstruct a cv::Mat image from a serialized buffer. This requires the type
 * of channel values to be known at compile time. The default type is uint8_t.
 * The image size is specified by Width and Height. The last argument pp
 * specifies the parameters for image postprocessing that takes advantage of the
 * OpenCV framework. Returns a reconstructed cv::Mat image if successful and an
 * empty one otherwise.
 */
template<typename T>
inline cv::Mat image_utils::copy_buf_to_cvMat_with_known_type(const std::vector<uint8_t>& buf, const int Width, const int Height, const cvMat_proc_params& pp)
{
  if (buf.size() == 0u || Width == 0 || Height == 0) {
    _LBANN_DEBUG_MSG("An empty image (" << Height << " x " << Width \
                     << ") or a buffer (" << buf.size() << ")");
    return cv::Mat();
  }

  const size_t sz = static_cast<size_t>(Width*Height*sizeof(T));
  const size_t NCh = buf.size()/sz;

  if (sz*NCh != buf.size()) {
    _LBANN_DEBUG_MSG("Size mismatch. Buffer has " << buf.size() \
                     << " items when " << sz*NCh << " are expected.");
    return cv::Mat();
  }

  switch(NCh) {
    case 1u: return copy_buf_to_cvMat_with_full_info<T,1>(buf, Width, Height, pp);
    case 2u: return copy_buf_to_cvMat_with_full_info<T,2>(buf, Width, Height, pp);
    case 3u: return copy_buf_to_cvMat_with_full_info<T,3>(buf, Width, Height, pp);
    case 4u: return copy_buf_to_cvMat_with_full_info<T,4>(buf, Width, Height, pp);
  }

  _LBANN_DEBUG_MSG(NCh << "-channel image is not supported.");
  return cv::Mat();
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv



//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                       copy_cvMat_to_buf (Elemental)
/**
 * Copy a cv::Mat image into a data block of El::Matrix<DataType> type. This
 * requires the type of channel values and the number of channels in the image
 * to be known at compile time. The default for these are the DataType of LBANN
 * and 3 channels.
 * The argument pp specifies the parameters for image preprocessing that
 * takes advantage of the OpenCV framework. Returns true if successful.
 */
template<typename T, int NCh>
inline bool image_utils::copy_cvMat_to_buf_with_full_info(const cv::Mat& image, ::Mat& buf, const cvMat_proc_params& pp)
{
  // NCh need not be a template parameter here. It can be a function argument.
  // However, keeping it as a static parameter enables custom accesses on pixels
  // For example,
  //   typedef cv::Vec<T, NCh> Vec_T;
  //   image.at<Vec_T>(y, x) = newPixel;
  if (image.empty()) return false;

  const int Width = image.cols;
  const int Height = image.rows;
  const int sz = Height*Width;

  if (buf.Height() != sz*NCh) {
  #if 0
    return false;
  #else
    //_LBANN_DEBUG_MSG("Resizing buffer height to " << sz*NCh);
    buf.Resize(sz*NCh, ((buf.Width()<1)? 1 : buf.Width()));
  #endif
  }

  DataType* Pixels = buf.Buffer();

  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);

    if (std::is_same<DataType, T>::value) {
      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz) {
        // create a separate image per channel aliasing the memory of buf
        channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1), Pixels);
      }
      cv::split(image, channels);

      Pixels = buf.Buffer();
      scale(Pixels, Pixels + sz*NCh, Pixels, pp.alpha, pp.beta);
    } else {
      cv::split(image, channels);

      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz) {
        scale(reinterpret_cast<const T*>(channels[ch].datastart), reinterpret_cast<const T*>(channels[ch].dataend), Pixels, pp.alpha, pp.beta);
      }
    }
  } else {
    if (image.isContinuous()) {
      scale(reinterpret_cast<const T*>(image.datastart), reinterpret_cast<const T*>(image.dataend), Pixels, pp.alpha, pp.beta);
    } else {
      const int stride = Width*NCh;
      for (int i = 0; i < Height; ++i, Pixels += stride) {
        const T* ptr = reinterpret_cast<const T*>(image.ptr<const T>(i));
        scale(ptr, ptr+stride, Pixels, pp.alpha, pp.beta);
      }
    }
  }

  return true;
}

/**
 * Copy a cv::Mat image into a data block of El::Matrix<DataType> type. This
 * requires the type of channel values in the image to be known at compile time.
 * The default for these are the DataType of LBANN.
 * The argument pp specifies the parameters for image preprocessing that
 * takes advantage of the OpenCV framework. Returns true if successful.
 */
template<typename T>
inline bool image_utils::copy_cvMat_to_buf_with_known_type(const cv::Mat& image, ::Mat& buf, const cvMat_proc_params& pp)
{
  switch(image.channels()) {
    case 1: return copy_cvMat_to_buf_with_full_info<T,1>(image, buf, pp);
    case 2: return copy_cvMat_to_buf_with_full_info<T,2>(image, buf, pp);
    case 3: return copy_cvMat_to_buf_with_full_info<T,3>(image, buf, pp);
    case 4: return copy_cvMat_to_buf_with_full_info<T,4>(image, buf, pp);
  }
  return false;
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                       copy_buf_to_cvMat (Elemental)
/**
 * Reconstruct a cv::Mat image from a data block of El::Matrix<DataType> type.
 * This requires the type of channel values and the number of channels in the
 * image to be known at compile time. The default for these are DataType of
 * LBANN and 3 channels.
 * The image size is specified by Width and Height. The argument pp specifies
 * the parameters for image postprocessing that takes advantage of the OpenCV
 * framework. Returns an empty image if unsuccessful.
 */
template<typename T, int NCh>
inline cv::Mat image_utils::copy_buf_to_cvMat_with_full_info(const ::Mat& buf, const int Width, const int Height, const cvMat_proc_params& pp)
{
  using namespace lbann::patchworks;
  //typedef cv_image_type<T, NCh> Img_T;

  const int sz = Height*Width;
  if (sz*NCh != buf.Height()) {
    _LBANN_DEBUG_MSG("Size mismatch. Buffer has " << buf.Height() \
                     << " items in a column when " << sz*NCh << " are expected.");
    return cv::Mat();
  }

  const DataType* Pixels = buf.LockedBuffer();

  //cv::Mat image = cv::Mat(Height, Width, Img_T::T());
  cv::Mat image = cv::Mat(Height, Width, CV_MAKETYPE(cv::DataType<T>::depth, NCh));

  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);

    if (std::is_same<DataType, T>::value) {
      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz)
        channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1), const_cast<DataType*>(Pixels));
      cv::merge(channels, image);
      scale(reinterpret_cast<const T*>(image.datastart), reinterpret_cast<const T*>(image.dataend), reinterpret_cast<T*>(image.data), pp.alpha, pp.beta);
    } else {
      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz) {
        channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1));
        scale(Pixels, Pixels+sz, reinterpret_cast<T*>(channels[ch].data), pp.alpha, pp.beta);
      }
      cv::merge(channels, image);
    }
  } else {
    scale(Pixels, Pixels + sz*NCh, reinterpret_cast<T*>(image.data), pp.alpha, pp.beta);
  }

  return image;
}

/**
 * Reconstruct a cv::Mat image from a data block of El::Matrix<DataType> type.
 * This requires the type of channel values to be known at compile time. The
 * default type is DataType. In this case, the new image may require conversion
 * to an integer type during postprocessing such that it can be stored in an
 * typical image file format. An image can sometimes be constructed even when
 * T is different from DataType if the type casting of a DataType value into T
 * is valid.
 * The image size is specified by Width and Height. The last argument pp
 * specifies the parameters for image postprocessing that takes advantage of the
 * OpenCV framework. This returns a reconstructed cv::Mat image if successful
 * and an empty one otherwise.
 */
template<typename T>
inline cv::Mat image_utils::copy_buf_to_cvMat_with_known_type(const ::Mat& buf, const int Width, const int Height, const cvMat_proc_params& pp)
{
  if (buf.Height() == 0u || buf.Width() == 0u || Width == 0 || Height == 0) {
    _LBANN_DEBUG_MSG("An empty image (" << Height << " x " << Width \
                     << ") or a buffer (" << buf.Height() << " x " << buf.Width() << ").");
    return cv::Mat();
  }

  const int sz = Height*Width;
  const int NCh = buf.Height()/sz;

  if (sz*NCh != buf.Height()) {
    _LBANN_DEBUG_MSG("Size mismatch. Buffer has " << buf.Height() \
                     << " items in a column when " << sz*NCh << " are expected.");
    return cv::Mat();
  }

  switch(NCh) {
    case 1: return copy_buf_to_cvMat_with_full_info<T,1>(buf, Width, Height, pp);
    case 2: return copy_buf_to_cvMat_with_full_info<T,2>(buf, Width, Height, pp);
    case 3: return copy_buf_to_cvMat_with_full_info<T,3>(buf, Width, Height, pp);
    case 4: return copy_buf_to_cvMat_with_full_info<T,4>(buf, Width, Height, pp);
  }

  _LBANN_DEBUG_MSG(NCh << "-channel image is not supported.");
  return cv::Mat();
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

#endif // __LIB_OPENCV

} // end of namespace lbann

#endif // LBANN_IMAGE_UTILS_HPP
