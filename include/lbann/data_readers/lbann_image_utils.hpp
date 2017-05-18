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

#include "lbann/lbann_base.hpp"
#include <type_traits>
#include <typeinfo>   // operator typeid

#ifdef __LIB_OPENCV
#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#include "lbann/data_readers/patchworks/patchworks.hpp"
#include "lbann/data_readers/lbann_cv_utils.hpp"
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

    static bool loadJPG(std::vector<unsigned char> &image_buf, int& Width, int& Height, bool Flip, unsigned char*& Pixels);
    static bool loadJPG(const char* Imagefile, int& Width, int& Height, bool Flip, unsigned char*& Pixels);
    static bool saveJPG(const char* Imagefile, int Width, int Height, bool Flip, unsigned char* Pixels);

    // load/save an image into/from a temporary buffer
    /// Load an image from a file and put it into a serialized buffer
    static bool load_image(const std::string& filename, int& Width, int& Height, int& Type, cv_process& pp, std::vector<uint8_t>& buf);
    /// Save an image from a serialized buffer into a file
    static bool save_image(const std::string& filename, const int Width, const int Height, const int Type, cv_process& pp, const std::vector<uint8_t>& buf);

    // load/save an image into/from an LBANN data block of El::Matrix<DataType> type
    /// Load an image from a file and put it into an LBANN Mat data block
    static bool load_image(const std::string& filename, int& Width, int& Height, int& Type, cv_process& pp, ::Mat& data);
    /// Save an image using data from an LBANN Mat data block
    static bool save_image(const std::string& filename, const int Width, const int Height, const int Type, cv_process& pp, const ::Mat& data);

    // import/export via a buffer of std::vector<uchar> containg the raw bytes of an image file
    /// Import an image from a file buffer (inbuf) and put it into an LBANN Mat data block
    static bool import_image(const std::vector<uchar>& inbuf, int& Width, int& Height, int& Type, cv_process& pp, ::Mat& data);
    /// Export an image using data from an LBANN Mat block into a file buffer (outbuf)
    static bool export_image(const std::string& fileExt, std::vector<uchar>& outbuf, const int Width, const int Height, const int Type, cv_process& pp, const ::Mat& data);
  };

} // end of namespace lbann

#endif // LBANN_IMAGE_UTILS_HPP
