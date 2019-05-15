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

#include "lbann/data_readers/image_utils.hpp"
#include "lbann/utils/exception.hpp"

#define _THROW_EXCEPTION_NO_OPENCV_() { \
  std::stringstream err; \
  err << __FILE__ << " " << __LINE__ \
      << " :: not compiled with LBANN_ENABLE_OPENCV!"; \
  throw lbann_exception(err.str()); \
}


namespace lbann {

bool image_utils::loadIMG(const std::string& Imagefile, int& Width, int& Height, bool Flip, unsigned char *&Pixels, std::vector<char>& buf) {
#ifdef LBANN_HAS_OPENCV
    cv::Mat image = cv_utils::lbann_imread(Imagefile, _LBANN_CV_COLOR_, buf);
  if (image.empty()) {
    return false;
  }

  Width = image.cols;
  Height = image.rows;

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      Pixels[offset]                  = pixel[_LBANN_CV_BLUE_];
      Pixels[offset + Height*Width]   = pixel[_LBANN_CV_GREEN_];
      Pixels[offset + 2*Height*Width] = pixel[_LBANN_CV_RED_];
    }
  }

  return true;
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif
}

bool image_utils::loadIMG(std::vector<unsigned char>& image_buf, int& Width, int& Height, bool Flip, unsigned char *&Pixels) {
#ifdef LBANN_HAS_OPENCV
  cv::Mat image = cv::imdecode(image_buf, _LBANN_CV_COLOR_);
  //cv::Mat image = cv::imdecode(image_buf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  if (image.empty()) {
    return false;
  }

  Width = image.cols;
  Height = image.rows;

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      Pixels[offset]                  = pixel[_LBANN_CV_BLUE_];
      Pixels[offset + Height*Width]   = pixel[_LBANN_CV_GREEN_];
      Pixels[offset + 2*Height*Width] = pixel[_LBANN_CV_RED_];
    }
  }

  return true;
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif
}

bool image_utils::saveIMG(const std::string& Imagefile, int Width, int Height, bool Flip, unsigned char *Pixels) {
#ifdef LBANN_HAS_OPENCV
  cv::Mat image = cv::Mat(Height, Width, CV_8UC3);

  for (int y = 0; y < Height; y++) {
    for (int x = 0; x < Width; x++) {
      cv::Vec3b pixel;
      int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
      pixel[_LBANN_CV_BLUE_]  = Pixels[offset];
      pixel[_LBANN_CV_GREEN_] = Pixels[offset + Height*Width];
      pixel[_LBANN_CV_RED_]   = Pixels[offset + 2*Height*Width];
      image.at<cv::Vec3b>(y, x) = pixel;
    }
  }
  cv::imwrite(Imagefile, image);

  return true;
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif
}


#ifdef LBANN_HAS_OPENCV
bool image_utils::process_image(cv::Mat& image, int& Width, int& Height, int& Type, cv_process& pp, CPUMat& out) {
  bool ok1 = !image.empty() && pp.preprocess(image);
  bool ok2 = ok1 && cv_utils::copy_cvMat_to_buf(image, out, pp);
  // Disabling normalizer is needed because normalizer is not necessarily
  // called during preprocessing but implicitly applied during data copying to
  // reduce overhead.
  pp.disable_lazy_normalizer();

  if (!ok2) {
    throw lbann_exception(std::string("image_utils::process_image(): image ") +
      (image.empty()? "is empty." :
                      (ok1? "copying failed." :
                            "preprocessing failed.")));
  }

  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();

  return ok2;
}

bool image_utils::process_image(cv::Mat& image, int& Width, int& Height, int& Type, cv_process& pp, std::vector<uint8_t>& out) {
  bool ok1 = !image.empty() && pp.preprocess(image);
  bool ok2 = ok1 && cv_utils::copy_cvMat_to_buf(image, out, pp);
  pp.disable_lazy_normalizer();

  if (!ok2) {
    throw lbann_exception(std::string("image_utils::process_image(): image ") +
      (image.empty()? "is empty." :
                      (ok1? "copying failed." :
                            "preprocessing failed.")));
  }

  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();

  return ok2;
}

bool image_utils::process_image(cv::Mat& image, int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& out) {
  std::vector<cv::Mat> patches;
  bool ok1 = !image.empty() && pp.preprocess(image, patches);
  bool ok2 = ok1 && (patches.size() != 0u) && (patches.size() == out.size());
  bool ok3 = ok2;

  for(size_t i=0u; ok3 && (i < patches.size()); ++i) {
    ok3 = cv_utils::copy_cvMat_to_buf(patches[i], out[i], pp);
  }
  pp.disable_lazy_normalizer();

  if (!ok3) {
    throw lbann_exception(std::string("image_utils::process_image(): image ") +
      (image.empty()? "is empty." :
                      (ok1? (ok2? "copying failed." :
                                  "extracted to invalid number of patches: " +
                                   std::to_string(patches.size()) + " != " +
                                   std::to_string(out.size())) :
                            "preprocessing failed.")));
  }

  Width  = patches[0].cols;
  Height = patches[0].rows;
  Type   = patches[0].type();

  return ok3;
}
#endif // LBANN_HAS_OPENCV

/**
 *  @param filename The name of the image file to read in
 *  @param Width    The width of the image read
 *  @param Height   The height of the image read
 *  @param Type     The type of the image read (OpenCV code used for cv::Mat)
 *  @param pp       The pre-processing parameters
 *  @param data     The pre-processed image data to be stored in El::Matrix<DataType> format
 *  @param buf      A thread safe buffer for local, temporary, image decoding
 */
bool image_utils::load_image(const std::string& filename,
                             int& Width, int& Height, int& Type, cv_process& pp, CPUMat& data, std::vector<char>& buf, cv::Mat* cv_buf) {
#ifdef LBANN_HAS_OPENCV
  cv::Mat image = cv_utils::lbann_imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, buf, cv_buf);

  return process_image(image, Width, Height, Type, pp, data);
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif // LBANN_HAS_OPENCV
}

/**
 *  @param filename The name of the image file to read in
 *  @param Width    The width of a patch from the image read
 *  @param Height   The height of a patch from the image read
 *  @param Type     The type of the image patches (OpenCV code used for cv::Mat)
 *  @param pp       The pre-processing parameters
 *  @param data     The pre-processed image data to be stored in El::Matrix<DataType> format
 *  @param buf      A thread safe buffer for local, temporary, image decoding
 */
bool image_utils::load_image(const std::string& filename,
                                    int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& data, std::vector<char>& buf, cv::Mat* cv_buf) {
#ifdef LBANN_HAS_OPENCV
  cv::Mat image = cv_utils::lbann_imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, buf, cv_buf);

  return process_image(image, Width, Height, Type, pp, data);
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif // LBANN_HAS_OPENCV
}

//XX
/**
 *  @param filename The name of the image file to read in
 *  @param Width    The width of a patch from the image read
 *  @param Height   The height of a patch from the image read
 *  @param Type     The type of the image patches (OpenCV code used for cv::Mat)
 *  @param pp       The pre-processing parameters
 *  @param data     The pre-processed image data to be stored in El::Matrix<DataType> format
 */
bool image_utils::load_image(std::vector<unsigned char>& image_buf,
                                    int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& data, cv::Mat* cv_buf) {

  return import_image(image_buf, Width, Height, Type, pp, data, cv_buf);
}

/**
 *  @param filename The name of the image file to write
 *  @param Width    The width of the image to be written
 *  @param Height   The height of the image to be written
 *  @param Type     The type of the image to be written (OpenCV code used for cv::Mat)
 *  @param pp       The post-processing parameters
 *  @param data     The image data in El::Matrix<DataType> format to post-process and write
 */
bool image_utils::save_image(const std::string& filename,
                                    const int Width, const int Height, const int Type, cv_process& pp, const CPUMat& data) {
#ifdef LBANN_HAS_OPENCV
  pp.determine_inverse_lazy_normalization();
  cv::Mat image = cv_utils::copy_buf_to_cvMat(data, Width, Height, Type, pp);
  bool ok = !image.empty() && pp.postprocess(image);

  _LBANN_MILD_EXCEPTION(!ok, "Image postprocessing has failed.", false)

  return (ok && cv::imwrite(filename, image));
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif // LBANN_HAS_OPENCV
}

/**
 *  @param inbuf   The buffer that contains the raw bytes read from an image file
 *                 This can be for example, const std:vector<uchar>& or const cv::Mat&.
 *                 http://docs.opencv.org/trunk/d4/d32/classcv_1_1__InputArray.html
 *  @param Width   The width of the image consturcted out of inbuf
 *  @param Height  The height of the image consructed
 *  @param Type    The type of the image constructed (OpenCV code used for cv::Mat)
 *  @param pp      The pre-processing parameters
 *  @param data    The pre-processed image data. A set of sub-matrix Views can be used to store the data.
 */
bool image_utils::import_image(cv::InputArray inbuf,
                                      int& Width, int& Height, int& Type, cv_process& pp, CPUMat& data, cv::Mat* cv_buf) {
#ifdef LBANN_HAS_OPENCV
  cv::Mat image;
  if(cv_buf != nullptr) {
    image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, cv_buf);
  }else {
    image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  }

  return process_image(image, Width, Height, Type, pp, data);
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif // LBANN_HAS_OPENCV
}

/**
 *  @param inbuf   The buffer that contains the raw bytes read from an image file
 *                 This can be for example, const std:vector<uchar>& or const cv::Mat&.
 *                 http://docs.opencv.org/trunk/d4/d32/classcv_1_1__InputArray.html
 *  @param Width   The width of a patch from the image consturcted out of inbuf
 *  @param Height  The height of a patch from the image consructed
 *  @param Type    The type of the image patches (OpenCV code used for cv::Mat)
 *  @param pp      The pre-processing parameters
 *  @param data    The pre-processed image data. A set of sub-matrix Views can be used to store the data.
 */
bool image_utils::import_image(cv::InputArray inbuf,
                                      int& Width, int& Height, int& Type, cv_process_patches& pp, std::vector<CPUMat>& data, cv::Mat* cv_buf) {
#ifdef LBANN_HAS_OPENCV
  cv::Mat image;
  if(cv_buf != nullptr) {
    image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH, cv_buf);
  }else {
    image = cv::imdecode(inbuf, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  }

  return process_image(image, Width, Height, Type, pp, data);
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif // LBANN_HAS_OPENCV
}

/**
 *  @param fileExt The format extension name of image file: e.g., ".jpeg", ".png"
 *  @param outbuf  The preallocated buffer to contain the bytes to be written into an image file
 *  @param Width   The width of the image to be consturcted based on the given data of ::Mat
 *  @param Height  The height of the image
 *  @param Type    The type of the image (OpenCV code used for cv::Mat)
 *  @param pp      The post-processing parameters
 *  @param data    The image data. A sub-matrix View can be passed instead of the entire matrix.
 */
bool image_utils::export_image(const std::string& fileExt, std::vector<uchar>& outbuf,
                                      const int Width, const int Height, const int Type, cv_process& pp, const CPUMat& data) {
#ifdef LBANN_HAS_OPENCV
  pp.determine_inverse_lazy_normalization();
  cv::Mat image = cv_utils::copy_buf_to_cvMat(data, Width, Height, Type, pp);
  bool ok = !image.empty() && pp.postprocess(image);

  _LBANN_MILD_EXCEPTION(!ok, "Either the image is empty or postprocessing has failed.", false)
  _LBANN_MILD_EXCEPTION(fileExt.empty(), "Empty file format extension!", false)

  const std::string ext = ((fileExt[0] != '.')? ("." + fileExt) : fileExt);

  static const size_t max_img_header_size = 1024;
  const size_t capacity = image_data_amount(image) + max_img_header_size;

  if (outbuf.size() < capacity) {
    //std::cout << "bytes reserved for the image: " << image_data_amount(image) << std::endl;
    outbuf.resize(capacity);
  }

  return (ok && cv::imencode(ext, image, outbuf));
#else
  _THROW_EXCEPTION_NO_OPENCV_();
  return false;
#endif // LBANN_HAS_OPENCV
}



bool image_utils::load_image(std::vector<unsigned char>& image_buf,
                                    int& Width, int& Height, int& Type, cv_process& pp, CPUMat& data, cv::Mat* cv_buf) {
  return import_image(image_buf, Width, Height, Type, pp, data, cv_buf);
}

} // namespace lbann
