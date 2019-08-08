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
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <arpa/inet.h>
#include <opencv2/imgcodecs.hpp>
#include "lbann/utils/image.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {

namespace {

// Read filename into buf.
void read_file_to_buf(const std::string& filename, El::Matrix<uint8_t>& buf,
                      size_t& size) {
  FILE* f = fopen(filename.c_str(), "r");
  if (f == nullptr) {
    LBANN_ERROR("Could not open file " + filename);
  }
  // Determine the length.
  if (fseeko(f, 0, SEEK_END) != 0) {
    LBANN_ERROR("Could not seek to end of file " + filename);
  }
  off_t size_ = ftello(f);
  if (size_ == -1) {
    LBANN_ERROR("Could not get offset in file " + filename);
  }
  size = static_cast<size_t>(size_);
  rewind(f);
  // Allocate sufficient space and read.
  buf.Resize(size, 1);
  if (fread(buf.Buffer(), 1, size, f) != size) {
    LBANN_ERROR("Could not real file " + filename);
  }
  fclose(f);
}

// There are other SOFs, but these are the common ones.
const bool is_jpg_sof[16] = {
  true, true, true, true, false, true, true, true,
  false, true, true, true, false, true, true, true};

// Attempt to guess the decoded size of an image.
// May not return the actual size (and may just return 0), so treat this as a
// hint.
void guess_image_size(const El::Matrix<uint8_t>& buf_, size_t size,
                      size_t& height, size_t& width, size_t& channels) {
  height = 0;
  width = 0;
  channels = 0;
  const uint8_t* buf = buf_.LockedBuffer();
  if (size >= 2 &&  // Size
      buf[0] == 0xFF && buf[1] == 0xD8) {  // Signature
    // JPEG image.
    // See: https://en.wikipedia.org/wiki/JPEG#Syntax_and_structure
    // and https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
    // and https://github.com/python-pillow/Pillow/blob/master/src/PIL/JpegImagePlugin.py
    // JPEG is complicated, this will probably not work for every image.
    // Try to find a start-of-frame marker, and then get the size.
    for (size_t cur_pos = 2; cur_pos < size;) {
      uint8_t b = buf[cur_pos];
      if (b == 0xFF) {
        if (cur_pos + 1 >= size) { return; }  // Shouldn't happen.
        uint8_t marker = buf[cur_pos + 1];
        if (marker >= 0xC0 && marker <= 0xCF && is_jpg_sof[marker - 0xC0]) {
          // Found the SOF.
          // 2 for the marker, 2 for the frame header length, 1 for the precision.
          cur_pos += 5;
          if (cur_pos + 4 >= size) { return; }  // Shouldn't happen.
          uint16_t h_w[2];
          memcpy(h_w, &buf[cur_pos], 4);
          height = ntohs(h_w[0]);
          width = ntohs(h_w[1]);
          channels = 3;  // Assume color.
          return;
        } else {
          cur_pos += 2;
          if (cur_pos + 2 >= size) { return; }  // Shouldn't happen.
          // Skip ahead by the length of this segment.
          uint16_t l;
          memcpy(&l, &buf[cur_pos], 2);
          cur_pos += ntohs(l);
        }
      } else {
        // Skip non-0xFFs.
        cur_pos += 1;
      }
    }
  } else if (size >= 24 &&  // Size
             // Check signature
             buf[0] == 0x89 && buf[1] == 0x50 &&
             buf[2] == 0x4E && buf[3] == 0x47 &&
             buf[4] == 0x0D && buf[5] == 0x0A &&
             buf[6] == 0x1A && buf[7] == 0x0A &&
             // Need IHDR chunk.
             buf[12] == 'I' && buf[13] == 'H' &&
             buf[14] == 'D' && buf[15] == 'R') {
    // PNG image
    // See: https://en.wikipedia.org/wiki/Portable_Network_Graphics#File_header
    uint32_t h_w[2];
    memcpy(h_w, buf + 16, 8);
    // Convert from network byte order and get size.
    width = ntohl(h_w[0]);
    height = ntohl(h_w[1]);
    channels = 3;  // Assume color.
  }
  // Give up.
}

// Decode an image from a buffer using OpenCV.
void opencv_decode(El::Matrix<uint8_t>& buf, El::Matrix<uint8_t>& dst,
                   std::vector<size_t>& dims, const std::string filename) {
  const size_t encoded_size = buf.Height() * buf.Width();
  std::vector<size_t> buf_dims = {1, encoded_size, 1};
  cv::Mat cv_encoded = utils::get_opencv_mat(buf, buf_dims);
  // Attempt to guess the decoded size.
  // Warning: These may be wrong.
  size_t height, width, channels;
  guess_image_size(buf, encoded_size, height, width, channels);
  if (height != 0) {
    // We have a guess.
    dst.Resize(height*width*channels, 1);
    std::vector<size_t> guessed_dims = {channels, height, width};
    // Decode the image.
    cv::Mat cv_dst = utils::get_opencv_mat(dst, guessed_dims);
    cv::Mat real_decoded = cv::imdecode(cv_encoded,
                                        cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH,
                                        &cv_dst);
    // For now we only support 8-bit 1- or 3-channel images.
    if (real_decoded.type() != CV_8UC1 && real_decoded.type() != CV_8UC3) {
      LBANN_ERROR("Only support 8-bit 1- or 3-channel images, cannot load " + filename);
    }
    dims = {real_decoded.type() == CV_8UC1 ? 1ull : 3ull,
            static_cast<size_t>(real_decoded.rows),
            static_cast<size_t>(real_decoded.cols)};
    // If we did not guess the size right, need to copy.
    if (real_decoded.ptr() != dst.Buffer()) {
      dst.Resize(utils::get_linearized_size(dims), 1);
      cv_dst = utils::get_opencv_mat(dst, dims);
      real_decoded.copyTo(cv_dst);
    }
  } else {
    cv::Mat decoded = cv::imdecode(cv_encoded,
                                   cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    if (decoded.type() != CV_8UC1 && decoded.type() != CV_8UC3) {
      LBANN_ERROR("Only support 8-bit 1- or 3-channel images, cannot load " + filename);
    }
    dims = {decoded.type() == CV_8UC1 ? 1ull : 3ull,
            static_cast<size_t>(decoded.rows),
            static_cast<size_t>(decoded.cols)};
    // Copy to dst.
    dst.Resize(utils::get_linearized_size(dims), 1);
    cv::Mat cv_dst = utils::get_opencv_mat(dst, dims);
    decoded.copyTo(cv_dst);
  }
}

}  // anonymous namespace

void load_image(const std::string& filename, El::Matrix<uint8_t>& dst,
                std::vector<size_t>& dims) {
  // Load the encoded image.
  El::Matrix<uint8_t> buf;
  size_t encoded_size;
  read_file_to_buf(filename, buf, encoded_size);
  opencv_decode(buf, dst, dims, filename);
}

void decode_image(El::Matrix<uint8_t>& src, El::Matrix<uint8_t>& dst,
                  std::vector<size_t>& dims) {
  opencv_decode(src, dst, dims, "encoded image");
}

void save_image(const std::string& filename, El::Matrix<uint8_t>& src,
                const std::vector<size_t>& dims) {
  cv::Mat cv_src = utils::get_opencv_mat(src, dims);
  if (!cv::imwrite(filename, cv_src)) {
    LBANN_ERROR("Could not save image to " + filename);
  }
}

void save_image(const std::string& filename, const CPUMat& src,
                const std::vector<size_t>& dims) {
  if (dims.size() != 3 || (dims[0] != 1 && dims[0] != 3)) {
    LBANN_ERROR("Unsupported dimensions for saving an image.");
  }

  El::Matrix<uint8_t> cv_mat = get_uint8_t_image(src, dims);

  save_image(filename, cv_mat, dims);
}

El::Matrix<uint8_t> get_uint8_t_image(const CPUMat& image,
                            const std::vector<size_t>& dims)
{
  // Need to convert to uint8_t matrix in OpenCV format.
  // We will normalize to [0, 1], then map to [0, 255].
  const size_t size = utils::get_linearized_size(dims);
  El::Matrix<uint8_t> cv_mat = El::Matrix<uint8_t>(size, 1);
  // Find the minimum and maximum to normalize with.
  const DataType* __restrict__ img_buf = image.LockedBuffer();
  DataType min = std::numeric_limits<DataType>::max();
  DataType max = std::numeric_limits<DataType>::lowest();
  for (size_t i = 0; i < size; ++i) {
    min = std::min(min, img_buf[i]);
    max = std::max(max, img_buf[i]);
  }
  const DataType norm_denom = max - min;
  // Construct the OpenCV buffer.
  uint8_t* __restrict__ cv_buf = cv_mat.Buffer();
  for (size_t channel = 0; channel < dims[0]; ++channel) {
    const size_t img_offset = channel*dims[1]*dims[2];
    for (size_t col = 0; col < dims[2]; ++col) {
      for (size_t row = 0; row < dims[1]; ++row) {
        const DataType norm_img_val =
          (img_buf[img_offset + row + col*dims[1]] - min) / norm_denom;
        cv_buf[dims[0]*(col + row*dims[2]) + channel] =
          static_cast<uint8_t>(std::min(std::floor(norm_img_val) * 256, DataType(255)));
      }
    }
  }
  return cv_mat;
}

std::string encode_image(const El::Matrix<uint8_t>& image,
                         const std::vector<size_t>& dims)
{
  cv::Mat Mat_img = utils::get_opencv_mat(
    const_cast<El::Matrix<uint8_t>&>(image), dims);
  std::vector<uint8_t> encoded_img;
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 20};

  cv::imencode(".jpg", Mat_img, encoded_img, params);

  return std::string{encoded_img.begin(), encoded_img.end()};
}

}  // namespace lbann
