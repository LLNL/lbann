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

#include "lbann/data_readers/lbann_image_utils.hpp"
#include <stdlib.h>
#include <stdio.h>

#ifdef __LIB_OPENCV
#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
using namespace cv;
#endif


#define BMP_HEADER_MARKER   ((unsigned short) ('M' << 8) | 'B')

#pragma pack(push)
#pragma pack(2)
typedef struct __BMP_FILEHEADER
{
	unsigned short bfType;
	unsigned long  bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	unsigned long  bfOffBits;
    
} BMP_FILEHEADER;

typedef struct __BMP_INFOHEADER
{
	unsigned long  biSize;
	long           biWidth;
	long           biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned long  biCompression;
	unsigned long  biSizeImage;
	long           biXPelsPerMeter;
	long           biYPelsPerMeter;
	unsigned long  biClrUsed;
	unsigned long  biClrImportant;
    
} BMP_INFOHEADER;

typedef struct __BMP_RGBQUAD
{
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
    unsigned char rgbReserved;
    
} BMP_RGBQUAD;

typedef struct __BMP_INFO
{
    BMP_INFOHEADER bmiHeader;
    BMP_RGBQUAD    bmiColors[1];
    
} BMP_INFO;
#pragma pack(pop)


bool lbann::image_utils::loadBMP(const char* Imagefile, int& Width, int& Height, int& BPP, bool Flip, unsigned char*& Pixels)
{
    FILE *infile = fopen(Imagefile, "rb");
    if (infile == NULL) {
        fprintf(stderr, "can't open %s\n", Imagefile);
        return false;
	}
    
	// Read Bitmap File Header
    BMP_FILEHEADER header;
    fread(&header, sizeof(BMP_FILEHEADER), 1, infile);
	if (header.bfType != BMP_HEADER_MARKER) {
		fclose(infile);
        return false;
	}
    
	// Read Bitmap Info
    int bisize = header.bfOffBits - sizeof(BMP_FILEHEADER);
    BMP_INFO* info = (BMP_INFO*)malloc(bisize);
	fread(info, bisize, 1, infile);
    
	// Check Palette Count
	if (info->bmiHeader.biClrUsed != 0 || info->bmiHeader.biBitCount != 24) {
		free(info);
		fclose(infile);
        return false;
	}
    
	// Read DIB Bits
    int bitrowsize = ((info->bmiHeader.biWidth * info->bmiHeader.biBitCount + 31) / 32) * 4;
    int bitsize = bitrowsize * info->bmiHeader.biHeight;
    unsigned char* bits = (unsigned char*)malloc(bitsize);
	fread(bits, bitsize, 1, infile);
    
	// Set Pixels
    Width = info->bmiHeader.biWidth;
    Height = info->bmiHeader.biHeight;
    BPP = 3;
    Pixels = new unsigned char[Width * Height * BPP];
    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            Pixels[offset] = bits[y * bitrowsize + x * 3 + 2];
            Pixels[offset + Height*Width] = bits[y * bitrowsize + x * 3 + 1];
            Pixels[offset + 2*Height*Width] = bits[y * bitrowsize + x * 3 + 0];
        }
    }
    
	free(info);
	free(bits);
	fclose(infile);

    return true;
}

bool lbann::image_utils::saveBMP(const char* Imagefile, int Width, int Height, int BPP, bool Flip, unsigned char* Pixels)
{
    if (BPP != 3)
        return false;



    return false;
}

bool lbann::image_utils::loadPGM(const char* Imagefile, int& Width, int& Height, int& BPP, bool Flip, unsigned char*& Pixels)
{
    FILE *infile = fopen(Imagefile, "rb");
    if (infile == NULL) {
        fprintf(stderr, "can't open %s\n", Imagefile);
        return false;
    }

    char format[5];
    fscanf(infile, "%s", format);
    int width, height;
    fscanf(infile, "%d%d", &width, &height);
    int maxpixel;
    fscanf(infile, "%d", &maxpixel);

    Width = width;
    Height = height;
    BPP = 1;
    Pixels = new unsigned char[Width * Height * BPP];

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            int pixel = fgetc(infile);
            Pixels[offset] = (unsigned char)((double)pixel / maxpixel * 255.0);
        }
    }

    fclose(infile);
    return true;
}

bool lbann::image_utils::savePGM(const char* Imagefile, int Width, int Height, int BPP, bool Flip, unsigned char* Pixels)
{
    if (BPP != 1)
        return false;

    FILE* outfile = fopen(Imagefile, "wb");
    if (outfile == NULL) {
        fprintf(stderr, "can't create %s\n", Imagefile);
        return false;
    }

    fprintf(outfile, "P5\n");
    fprintf(outfile, "%d %d\n", Width, Height);
    fprintf(outfile, "255\n");

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            fputc(Pixels[offset], outfile);
        }
    }

    fclose(outfile);
    return true;
}

bool lbann::image_utils::loadPNG(const char* Imagefile, int& Width, int& Height, bool Flip, uchar*& Pixels)
{
#ifdef __LIB_OPENCV
    cv::Mat image = cv::imread(Imagefile, _LBANN_CV_COLOR_);
    if (image.empty())
        return false;

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
    return false;
#endif
}

bool lbann::image_utils::savePNG(const char* Imagefile, int Width, int Height, bool Flip, uchar* Pixels)
{
#ifdef __LIB_OPENCV
    cv::Mat image = cv::Mat(Height, Width, CV_8UC3);

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            cv::Vec3b pixel;
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            pixel[_LBANN_CV_BLUE_] = Pixels[offset];
            pixel[_LBANN_CV_GREEN_] = Pixels[offset + Height*Width];
            pixel[_LBANN_CV_RED_] = Pixels[offset + 2*Height*Width];
            image.at<cv::Vec3b>(y, x) = pixel;
        }
    }
    imwrite(Imagefile, image);

    return true;
#else
    return false;
#endif
}

bool lbann::image_utils::loadJPG(const char* Imagefile, int& Width, int& Height, bool Flip, unsigned char*& Pixels)
{
#ifdef __LIB_OPENCV
    cv::Mat image = cv::imread(Imagefile, _LBANN_CV_COLOR_);
    if (image.empty())
        return false;

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
    return false;
#endif
}

bool lbann::image_utils::saveJPG(const char* Imagefile, int Width, int Height, bool Flip, unsigned char* Pixels)
{
#ifdef __LIB_OPENCV
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
    imwrite(Imagefile, image);

    return true;
#else
    return false;
#endif
}

#ifdef __LIB_OPENCV
bool lbann::image_utils::preprocess_cvMat(cv::Mat& image, const lbann::cvMat_proc_params& pp)
{
  if (image.empty()) return false;
  switch(image.depth()) {
    case CV_8U:  return preprocess_cvMat_with_known_type<_depth_type(CV_8U)>(image, pp);
    case CV_8S:  return preprocess_cvMat_with_known_type<_depth_type(CV_8S)>(image, pp);
    case CV_16U: return preprocess_cvMat_with_known_type<_depth_type(CV_16U)>(image, pp);
    case CV_16S: return preprocess_cvMat_with_known_type<_depth_type(CV_16S)>(image, pp);
    case CV_32S: return preprocess_cvMat_with_known_type<_depth_type(CV_32S)>(image, pp);
    case CV_32F: return preprocess_cvMat_with_known_type<_depth_type(CV_32F)>(image, pp);
    case CV_64F: return preprocess_cvMat_with_known_type<_depth_type(CV_64F)>(image, pp);
  }
  return false;
}

bool lbann::image_utils::postprocess_cvMat(cv::Mat& image, const lbann::cvMat_proc_params& pp)
{
  if (image.empty()) return false;
  switch(image.depth()) {
    case CV_8U:  return postprocess_cvMat_with_known_type<_depth_type(CV_8U)>(image, pp);
    case CV_8S:  return postprocess_cvMat_with_known_type<_depth_type(CV_8S)>(image, pp);
    case CV_16U: return postprocess_cvMat_with_known_type<_depth_type(CV_16U)>(image, pp);
    case CV_16S: return postprocess_cvMat_with_known_type<_depth_type(CV_16S)>(image, pp);
    case CV_32S: return postprocess_cvMat_with_known_type<_depth_type(CV_32S)>(image, pp);
    case CV_32F: return postprocess_cvMat_with_known_type<_depth_type(CV_32F)>(image, pp);
    case CV_64F: return postprocess_cvMat_with_known_type<_depth_type(CV_64F)>(image, pp);
  }
  return false;
}

bool lbann::image_utils::copy_cvMat_to_buf(const cv::Mat& image, std::vector<uint8_t>& buf, const cvMat_proc_params& pp)
{
  if (image.empty()) return false;
  switch(image.depth()) {
    case CV_8U:  return copy_cvMat_to_buf_with_known_type<_depth_type(CV_8U)>(image, buf, pp);
    case CV_8S:  return copy_cvMat_to_buf_with_known_type<_depth_type(CV_8S)>(image, buf, pp);
    case CV_16U: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_16U)>(image, buf, pp);
    case CV_16S: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_16S)>(image, buf, pp);
    case CV_32S: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_32S)>(image, buf, pp);
    case CV_32F: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_32F)>(image, buf, pp);
    case CV_64F: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_64F)>(image, buf, pp);
  }
  return false;
}

cv::Mat lbann::image_utils::copy_buf_to_cvMat(const std::vector<uint8_t>& buf, const int Width, const int Height, const int Type, const cvMat_proc_params& pp)
{ 
  if (buf.size() != static_cast<size_t>(Width * Height * CV_MAT_CN(Type) * CV_ELEM_SIZE(CV_MAT_DEPTH(Type)))) {
    _LBANN_DEBUG_MSG("Size mismatch: Buffer has " << buf.size() << " items when " \
              << static_cast<size_t>(Width * Height * CV_MAT_CN(Type) * CV_ELEM_SIZE(CV_MAT_DEPTH(Type))) \
              << " are expected.");
    return cv::Mat();
  }

  switch(CV_MAT_DEPTH(Type)) {
    case CV_8U:  return copy_buf_to_cvMat_with_known_type<_depth_type(CV_8U)>(buf, Width, Height, pp);
    case CV_8S:  return copy_buf_to_cvMat_with_known_type<_depth_type(CV_8S)>(buf, Width, Height, pp);
    case CV_16U: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_16U)>(buf, Width, Height, pp);
    case CV_16S: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_16S)>(buf, Width, Height, pp);
    case CV_32S: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_32S)>(buf, Width, Height, pp);
    case CV_32F: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_32F)>(buf, Width, Height, pp);
    case CV_64F: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_64F)>(buf, Width, Height, pp);
  }
  _LBANN_DEBUG_MSG("Unknown image depth: " << CV_MAT_DEPTH(Type));
  return cv::Mat();
}

bool lbann::image_utils::copy_cvMat_to_buf(const cv::Mat& image, ::Mat& buf, const cvMat_proc_params& pp)
{
  if (image.empty()) return false;
  switch(image.depth()) {
    case CV_8U:  return copy_cvMat_to_buf_with_known_type<_depth_type(CV_8U)>(image, buf, pp);
    case CV_8S:  return copy_cvMat_to_buf_with_known_type<_depth_type(CV_8S)>(image, buf, pp);
    case CV_16U: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_16U)>(image, buf, pp);
    case CV_16S: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_16S)>(image, buf, pp);
    case CV_32S: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_32S)>(image, buf, pp);
    case CV_32F: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_32F)>(image, buf, pp);
    case CV_64F: return copy_cvMat_to_buf_with_known_type<_depth_type(CV_64F)>(image, buf, pp);
  }
  return false;
}

cv::Mat lbann::image_utils::copy_buf_to_cvMat(const ::Mat& buf, const int Width, const int Height, const int Type, const cvMat_proc_params& pp)
{
  switch(CV_MAT_DEPTH(Type)) {
    case CV_8U:  return copy_buf_to_cvMat_with_known_type<_depth_type(CV_8U)>(buf, Width, Height, pp);
    case CV_8S:  return copy_buf_to_cvMat_with_known_type<_depth_type(CV_8S)>(buf, Width, Height, pp);
    case CV_16U: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_16U)>(buf, Width, Height, pp);
    case CV_16S: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_16S)>(buf, Width, Height, pp);
    case CV_32S: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_32S)>(buf, Width, Height, pp);
    case CV_32F: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_32F)>(buf, Width, Height, pp);
    case CV_64F: return copy_buf_to_cvMat_with_known_type<_depth_type(CV_64F)>(buf, Width, Height, pp);
  }
  _LBANN_DEBUG_MSG("Unknown image depth: " << CV_MAT_DEPTH(Type));
  return cv::Mat();
}
#endif // __LIB_OPENCV

bool lbann::image_utils::load_image(const std::string& filename, int& Width, int& Height, int& Type, const cvMat_proc_params& pp, std::vector<uint8_t>& buf)
{
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  bool ok = preprocess_cvMat(image, pp);
  if (!ok || !copy_cvMat_to_buf(image, buf, pp)) {
    _LBANN_DEBUG_MSG("Image preprocessing or copying failed.");
    return false;
  }
  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();
  return true;
#else
  return false;
#endif // __LIB_OPENCV
}

bool lbann::image_utils::save_image(const std::string& filename, const int Width, const int Height, const int Type, const cvMat_proc_params& pp, const std::vector<uint8_t>& buf)
{
#ifdef __LIB_OPENCV
  cv::Mat image = copy_buf_to_cvMat(buf, Width, Height, Type, pp);
  bool ok = !image.empty() && postprocess_cvMat(image, pp);
  if (!ok) {
    _LBANN_DEBUG_MSG("Either the image is empty or postprocessing has failed.");
    return false;
  }
  return (ok && cv::imwrite(filename, image));
#else
  return false;
#endif // __LIB_OPENCV
}

bool lbann::image_utils::load_image(const std::string& filename, int& Width, int& Height, int& Type, const cvMat_proc_params& pp, ::Mat& buf)
{
#ifdef __LIB_OPENCV
  cv::Mat image = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  bool ok = preprocess_cvMat(image, pp);
  if (!ok || !copy_cvMat_to_buf(image, buf, pp)) {
    _LBANN_DEBUG_MSG("Image preprocessing or copying failed.");
    return false;
  }
  Width  = image.cols;
  Height = image.rows;
  Type   = image.type();
  return true;
#else
  return false;
#endif // __LIB_OPENCV
}

bool lbann::image_utils::save_image(const std::string& filename, const int Width, const int Height, const int Type, const cvMat_proc_params& pp, const ::Mat& buf)
{
#ifdef __LIB_OPENCV
  cv::Mat image = copy_buf_to_cvMat(buf, Width, Height, Type, pp);
  bool ok = !image.empty() && postprocess_cvMat(image, pp);
  if (!ok) {
    _LBANN_DEBUG_MSG("Either the image is empty or postprocessing has failed.");
    return false;
  }
  return (ok && cv::imwrite(filename, image));
#else
  return false;
#endif // __LIB_OPENCV
}
