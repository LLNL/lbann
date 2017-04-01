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
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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
    Mat image = imread(Imagefile, CV_LOAD_IMAGE_COLOR);
    if (image.empty())
        return false;

    Width = image.cols;
    Height = image.rows;

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            Pixels[offset] = pixel[0];
            Pixels[offset + Height*Width] = pixel[1];
            Pixels[offset + 2*Height*Width] = pixel[2];
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
    Mat image = Mat(Height, Width, CV_8UC3);

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            Vec3b pixel;
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            pixel[0] = Pixels[offset];
            pixel[1] = Pixels[offset + Height*Width];
            pixel[2] = Pixels[offset + 2*Height*Width];
            image.at<Vec3b>(y, x) = pixel;
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
    Mat image = imread(Imagefile, CV_LOAD_IMAGE_COLOR);
    if (image.empty())
        return false;

    Width = image.cols;
    Height = image.rows;

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            Pixels[offset] = pixel[0];
            Pixels[offset + Height*Width] = pixel[1];
            Pixels[offset + 2*Height*Width] = pixel[2];
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
    Mat image = Mat(Height, Width, CV_8UC3);

    for (int y = 0; y < Height; y++) {
        for (int x = 0; x < Width; x++) {
            Vec3b pixel;
            int offset = (Flip) ? ((Height - 1 - y) * Width + x) : (y * Width + x);
            pixel[0] = Pixels[offset];
            pixel[1] = Pixels[offset + Height*Width];
            pixel[2] = Pixels[offset + 2*Height*Width];
            image.at<Vec3b>(y, x) = pixel;
        }
    }
    imwrite(Imagefile, image);

    return true;
#else
    return false;
#endif
}
