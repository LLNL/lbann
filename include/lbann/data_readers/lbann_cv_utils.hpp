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

#ifndef LBANN_CV_UTILS_HPP
#define LBANN_CV_UTILS_HPP

#include <type_traits>
#include <typeinfo>   // operator typeid
#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#include "lbann/data_readers/lbann_cv_process.hpp"
#include "lbann/utils/lbann_mild_exception.hpp"


#ifdef __LIB_OPENCV
namespace lbann
{

class cv_utils
{
 public:
  static size_t image_data_amount(const cv::Mat& img);

  // preprocess (with cv::Mat type image data)
  template<typename T = uint8_t, int NCh = 3>
  static bool preprocess_cvMat_with_full_info(cv::Mat& image, cv_process& pp);

  template<typename T = uint8_t>
  static bool preprocess_cvMat_with_known_type(cv::Mat& image, cv_process& pp);

  static bool preprocess_cvMat(cv::Mat& image, cv_process& pp);

  // postprocess (with cv::Mat type image data)
  template<typename T = uint8_t, int NCh = 3>
  static bool postprocess_cvMat_with_full_info(cv::Mat& image, cv_process& pp);

  template<typename T = uint8_t>
  static bool postprocess_cvMat_with_known_type(cv::Mat& image, cv_process& pp);

  static bool postprocess_cvMat(cv::Mat& image, cv_process& pp);


  // copy_cvMat_to_buf (with a tempoary buffer)
  template<typename T = uint8_t, int NCh = 3>
  static bool copy_cvMat_to_buf_with_full_info(const cv::Mat& image, std::vector<uint8_t>& buf, const cv_process& pp);

  template<typename T = uint8_t>
  static bool copy_cvMat_to_buf_with_known_type(const cv::Mat& image, std::vector<uint8_t>& buf, const cv_process& pp);

  /** Copy a cv::Mat image into a serialized buffer.
   *  The argument pp specifies the parameters for image preprocessing that
   *  takes advantage of the OpenCV framework. Returns true if successful.
   */
  static bool copy_cvMat_to_buf(const cv::Mat& image, std::vector<uint8_t>& buf, const cv_process& pp);


  // copy_buf_to_cvMat (with a tempoary buffer)
  template<typename T = uint8_t, int NCh = 3>
  static cv::Mat copy_buf_to_cvMat_with_full_info(const std::vector<uint8_t>& buf, const int Width, const int Height, const cv_process& pp);

  template<typename T = uint8_t>
  static cv::Mat copy_buf_to_cvMat_with_known_type(const std::vector<uint8_t>& buf, const int Width, const int Height, const cv_process& pp);

  /** Reconstruct a cv::Mat image from a serialized buffer.
   *  The image size is specified by Width and Height. Type indetifies the
   *  OpenCV image type. The last argument pp specifies the parameters for
   *  image postprocessing that takes advantage of the OpenCV framework.
   *  Returns a reconstructed cv::Mat image if successful and an empty one
   *  otherwise.
   */
  static cv::Mat copy_buf_to_cvMat(const std::vector<uint8_t>& buf, const int Width, const int Height, const int Type, const cv_process& pp);


  // copy_buf_to_cvMat (with an El::Matrix<DataType> block)
  template<typename T = DataType, int NCh = 3>
  static bool copy_cvMat_to_buf_with_full_info(const cv::Mat& image, ::Mat& buf, const cv_process& pp);

  template<typename T = DataType>
  static bool copy_cvMat_to_buf_with_known_type(const cv::Mat& image, ::Mat& buf, const cv_process& pp);

  /** Copy a cv::Mat image into an El::Matrix<DataType> block.
   *  The argument pp specifies the parameters for image preprocessing that
   *  takes advantage of the OpenCV framework. Returns true if successful.
   */
  static bool copy_cvMat_to_buf(const cv::Mat& image, ::Mat& buf, const cv_process& pp);


  // copy_buf_to_cvMat (with an El::Matrix<DataType> block)
  template<typename T = DataType, int NCh = 3>
  static cv::Mat copy_buf_to_cvMat_with_full_info(const ::Mat& buf, const int Width, const int Height, const cv_process& pp);

  template<typename T = DataType>
  static cv::Mat copy_buf_to_cvMat_with_known_type(const ::Mat& buf, const int Width, const int Height, const cv_process& pp);

  /** Reconstruct a cv::Mat image from an El::Matrix<DataType> block.
   *  The image size is specified by Width and Height. Type indetifies the
   *  OpenCV image type. The last argument pp specifies the parameters for
   *  image postprocessing that takes advantage of the OpenCV framework.
   *  Returns a reconstructed cv::Mat image if successful and an empty one
   *  otherwise.
   */
  static cv::Mat copy_buf_to_cvMat(const ::Mat& buf, const int Width, const int Height, const int Type, const cv_process& pp);
};

//------------------------------------------------------------------------------


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
inline bool cv_utils::preprocess_cvMat_with_full_info(cv::Mat& image, cv_process& pp)
{
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  bool ok = true;

  if (ok && pp.custom_transform1().is_set()) {
    ok = pp.custom_transform1().apply(image);
  }

  // place for augmentation
  if (pp.to_flip())
    cv::flip(image, image, pp.how_to_flip());

  ok = pp.augment(image);

  if (ok && pp.custom_transform2().is_set()) {
    ok = pp.custom_transform2().apply(image);
  }

  // The place for early-normalization in case that there is something to be done
  // after normalization. Otherwise, normalization will be done during copying
  // via scaling. If early-normalization is done, scaling parameters should be
  // reset to avoid normalizing again.

  pp.compute_normalization_params(image);
 #if 0
  for (size_t i=0u; i < pp.alpha().size(); ++i) {
    std::cout << "scaling: " << pp.alpha()[i] << ' ' << pp.beta()[i] << std::endl;
  }
 #endif

  if (ok && pp.custom_transform3().is_set()) {
    std::cout << "custom_transform3 " << std::endl;
    // normalization
    ok = pp.normalize(image);
    pp.reset_normalization_params();
    ok = pp.custom_transform3().apply(image);
  }

  return ok;
}

template<typename T>
inline bool cv_utils::preprocess_cvMat_with_known_type(cv::Mat& image, cv_process& pp)
{
  _SWITCH_CV_FUNC_KNOWN_TYPE_2PARAMS(image.channels(), T, \
                                     preprocess_cvMat_with_full_info, \
                                     image, pp)
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
inline bool cv_utils::postprocess_cvMat_with_full_info(cv::Mat& image, cv_process& pp)
{
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  if (pp.to_flip())
    cv::flip(image, image, pp.how_to_flip());
 #if 0
  for (size_t i=0u; i < pp.alpha().size(); ++i) {
    std::cout << "scaling: " << pp.alpha()[i] << ' ' << pp.beta()[i] << std::endl;
  }
 #endif
  return true;
}

template<typename T>
inline bool cv_utils::postprocess_cvMat_with_known_type(cv::Mat& image, cv_process& pp)
{
  _SWITCH_CV_FUNC_KNOWN_TYPE_2PARAMS(image.channels(), T, \
                                     postprocess_cvMat_with_full_info, \
                                     image, pp)
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
inline bool cv_utils::copy_cvMat_to_buf_with_full_info(
  const cv::Mat& image, std::vector<uint8_t>& buf, const cv_process& pp)
{
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  const int Width = image.cols;
  const int Height = image.rows;
  const int sz = Height*Width;

  buf.resize(sz*NCh*sizeof(T));
  T* Pixels = reinterpret_cast<T*>(&(buf[0]));

  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);
    for(size_t ch=0; ch < NCh; ++ch, Pixels += sz)
      channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1), Pixels);
    cv::split(image, channels);
    Pixels = reinterpret_cast<T*>(&(buf[0]));
    cv_preprocessor::
    scale(Pixels, Pixels + sz*NCh, Pixels, pp.alpha(), pp.beta());
  } else {
    if (image.isContinuous()) {
      cv_preprocessor::
      scale(reinterpret_cast<const T*>(image.datastart),
            reinterpret_cast<const T* const>(image.dataend),
            Pixels, pp.alpha(), pp.beta());
    } else {
      const int stride = Width*NCh;
      for (int i = 0; i < Height; ++i, Pixels += stride) {
        const T* ptr = reinterpret_cast<const T*>(image.ptr<const T>(i));
        cv_preprocessor::
        scale(ptr, ptr+stride, Pixels, pp.alpha(), pp.beta());
      }
    }
  }

  return true;
}

/**
 * Copy a cv::Mat image into a serialized buffer. This requires the type of
 * channel values to be known at compile time. The default type is uint8_t.
 * The argument pp specifies the parameters for image preprocessing that
 * takes advantage of the OpenCV framework. Returns true if successful.
 */
template<typename T>
inline bool cv_utils::copy_cvMat_to_buf_with_known_type(
  const cv::Mat& image, std::vector<uint8_t>& buf, const cv_process& pp)
{
  _SWITCH_CV_FUNC_KNOWN_TYPE_3PARAMS(image.channels(), T, \
                                     copy_cvMat_to_buf_with_full_info, \
                                     image, buf, pp)
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
inline cv::Mat cv_utils::copy_buf_to_cvMat_with_full_info(
  const std::vector<uint8_t>& buf, const int Width, const int Height, const cv_process& pp)
{
  using namespace lbann::patchworks;
  //typedef cv_image_type<T, NCh> Img_T;

  const int sz = Height*Width;

  _LBANN_MILD_EXCEPTION(sz*NCh*sizeof(T) != buf.size(), \
    "Size mismatch. Buffer has " << buf.size() << " items when " \
    << sz*NCh*sizeof(T) << " are expected.", \
    cv::Mat())

  const T* Pixels = reinterpret_cast<const T*>(&(buf[0]));

  //cv::Mat image = cv::Mat(Height, Width, Img_T::T());
  cv::Mat image = cv::Mat(Height, Width, CV_MAKETYPE(cv::DataType<T>::depth, NCh));

  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);

    for(size_t ch=0; ch < NCh; ++ch, Pixels += sz)
      channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1), const_cast<T*>(Pixels));

    cv::merge(channels, image);
    cv_preprocessor::
    scale(reinterpret_cast<const T*>(image.datastart),
          reinterpret_cast<const T* const>(image.dataend),
          reinterpret_cast<T*>(image.data), pp.alpha(), pp.beta());
  } else {
    cv_preprocessor::
    scale(Pixels, Pixels + sz*NCh, reinterpret_cast<T*>(image.data), pp.alpha(), pp.beta());
  }

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
inline cv::Mat cv_utils::copy_buf_to_cvMat_with_known_type(
  const std::vector<uint8_t>& buf, const int Width, const int Height, const cv_process& pp)
{
  _LBANN_MILD_EXCEPTION(buf.size() == 0u || Width == 0 || Height == 0, \
    "An empty image (" << Height << " x " << Width << ") or a buffer (" << buf.size() << ")", \
    cv::Mat())

  const size_t sz = static_cast<size_t>(Width*Height*sizeof(T));
  const size_t NCh = buf.size()/sz;

  _LBANN_MILD_EXCEPTION(sz*NCh != buf.size(), \
    "Size mismatch. Buffer has " << buf.size() << " items when " << sz*NCh << " are expected.", \
    cv::Mat())

  _SWITCH_CV_FUNC_KNOWN_TYPE_4PARAMS(NCh, T, \
                                     copy_buf_to_cvMat_with_full_info, \
                                     buf, Width, Height, pp);

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
 * and 3 channels. In case of copying a single image into a collection of
 * images as an existing El::Matrix<DataType> matrix, a sub-matrix View can be passed.
 * The argument pp specifies the parameters for image preprocessing that
 * takes advantage of the OpenCV framework. Returns true if successful.
 */
template<typename T, int NCh>
inline bool cv_utils::copy_cvMat_to_buf_with_full_info(
  const cv::Mat& image, ::Mat& buf, const cv_process& pp)
{
  // NCh need not be a template parameter here. It can be a function argument.
  // However, keeping it as a static parameter enables custom accesses on pixels
  // For example,
  //   typedef cv::Vec<T, NCh> Vec_T;
  //   image.at<Vec_T>(y, x) = newPixel;
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

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
      cv_preprocessor::
      scale(Pixels, Pixels + sz*NCh,
            Pixels, pp.alpha(), pp.beta());
    } else {
      cv::split(image, channels);

      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz) {
        cv_preprocessor::
        scale(reinterpret_cast<const T*>(channels[ch].datastart),
              reinterpret_cast<const T* const>(channels[ch].dataend),
              Pixels, pp.alpha(), pp.beta());
      }
    }
  } else {
    if (image.isContinuous()) {
      cv_preprocessor::
      scale(reinterpret_cast<const T*>(image.datastart),
            reinterpret_cast<const T* const>(image.dataend),
            Pixels, pp.alpha(), pp.beta());
    } else {
      const int stride = Width*NCh;
      for (int i = 0; i < Height; ++i, Pixels += stride) {
        const T* ptr = reinterpret_cast<const T*>(image.ptr<const T>(i));
        cv_preprocessor::
        scale(ptr, ptr+stride, Pixels, pp.alpha(), pp.beta());
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
inline bool cv_utils::copy_cvMat_to_buf_with_known_type(
  const cv::Mat& image, ::Mat& buf, const cv_process& pp)
{
  _SWITCH_CV_FUNC_KNOWN_TYPE_3PARAMS(image.channels(), T, \
                                     copy_cvMat_to_buf_with_full_info, \
                                     image, buf, pp)
  return false;
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                       copy_buf_to_cvMat (Elemental)
/**
 * Reconstruct a cv::Mat image from a data block of El::Matrix<DataType> type.
 * This requires the type of channel values and the number of channels in the
 * image to be known at compile time. The default for these are DataType of
 * LBANN and 3 channels. In case of copying a single image data in a matrix
 * of multiple images, a sub-matrix View can be passed.
 * The image size is specified by Width and Height. The argument pp specifies
 * the parameters for image postprocessing that takes advantage of the OpenCV
 * framework. Returns an empty image if unsuccessful.
 */
template<typename T, int NCh>
inline cv::Mat cv_utils::copy_buf_to_cvMat_with_full_info(
  const ::Mat& buf, const int Width, const int Height, const cv_process& pp)
{
  using namespace lbann::patchworks;
  //typedef cv_image_type<T, NCh> Img_T;

  const int sz = Height*Width;
  _LBANN_MILD_EXCEPTION(sz*NCh != buf.Height(), \
    "Size mismatch. Buffer has " << buf.Height() << " items in a column when " \
    << sz*NCh << " are expected.", \
    cv::Mat())

  const DataType* Pixels = buf.LockedBuffer();

  //cv::Mat image = cv::Mat(Height, Width, Img_T::T());
  cv::Mat image = cv::Mat(Height, Width, CV_MAKETYPE(cv::DataType<T>::depth, NCh));

  if (pp.to_split()) {
    std::vector<cv::Mat> channels(NCh);

    if (std::is_same<DataType, T>::value) {
      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz)
        channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1),
                               const_cast<DataType*>(Pixels));

      cv::merge(channels, image);

      cv_preprocessor::
      scale(reinterpret_cast<const T*>(image.datastart),
            reinterpret_cast<const T* const>(image.dataend),
            reinterpret_cast<T*>(image.data),
            pp.alpha(), pp.beta());
    } else {
      for(size_t ch=0; ch < NCh; ++ch, Pixels += sz) {
        channels[ch] = cv::Mat(Height, Width, CV_MAKETYPE(image.depth(),1));
        cv_preprocessor::
        scale(Pixels, Pixels+sz,
              reinterpret_cast<T*>(channels[ch].data),
              pp.alpha(), pp.beta());
      }
      cv::merge(channels, image);
    }
  } else {
    cv_preprocessor::
    scale(Pixels, Pixels + sz*NCh,
          reinterpret_cast<T*>(image.data),
          pp.alpha(), pp.beta());
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
inline cv::Mat cv_utils::copy_buf_to_cvMat_with_known_type(
  const ::Mat& buf, const int Width, const int Height, const cv_process& pp)
{
  _LBANN_MILD_EXCEPTION(buf.Height() == 0u || buf.Width() == 0u || Width == 0 || Height == 0, \
    "An empty image (" << Height << " x " << Width << ") or a buffer (" \
    << buf.Height() << " x " << buf.Width() << ").", \
    cv::Mat())

  const int sz = Height*Width;
  const int NCh = buf.Height()/sz;

  _LBANN_MILD_EXCEPTION(sz*NCh != buf.Height(), \
    "Size mismatch. Buffer has " << buf.Height() << " items in a column when " \
    << sz*NCh << " are expected.", \
    cv::Mat())

  _SWITCH_CV_FUNC_KNOWN_TYPE_4PARAMS(NCh, T, \
                                     copy_buf_to_cvMat_with_full_info, \
                                     buf, Width, Height, pp)

  _LBANN_DEBUG_MSG(NCh << "-channel image is not supported.");
  return cv::Mat();
}
//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_UTILS_HPP
