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
// datatype.h - basic data type, macro, inline function
////////////////////////////////////////////////////////////////////////////////

#ifndef DATATYPE_HPP_INCLUDED
#define DATATYPE_HPP_INCLUDED

#include <math.h>

#ifndef NULL
#define NULL 0
#endif

#define __PI                                    3.14159265359
#define __PI_SQRT                               1.77245385091

#define __MIN(a, b)                             ((a < b) ? a : b)
#define __MAX(a, b)                             ((a > b) ? a : b)

#define __RAD2DEG(rad)                          ((double)rad * 180 / __PI)
#define __DEG2RAD(deg)                          ((double)deg * __PI / 180)

#define __LOG(a, base)                          (log(double(a)) / log(double(base)))
#define __LOG2(a)                               (log(double(a)) / log(2.0))

#define __OFFSET_2D(x, y, width)                (y * width + x)
#define __OFFSET_3D(x, y, z, width, height)     (z * width * height + y * width + x)

#define __LENGTH_2D(x1, y1, x2, y2)             (sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)))
#define __LENGTH_3D(x1, y1, z1, x2, y2, z2)     (sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)))
#define __LENGTH_3D_2(x1, y1, z1, x2, y2, z2)   ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))

#define __ORIENTATION(x1, y1, x2, y2, x3, y3)   ((x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3))
#define __SLOPE(x1, y1, x2, y2)         ((y2 - y1) / (x2 - x1))
#define __COS2TH(a, b, c)           ((a * a + c * c - b * b) / (2.0 * a * c))

#define __SWAP16(a)     ( (((a) & 0x00ff) << 8) | (((a)&0xff00) >> 8) )
#define __SWAP32(a)     ( (((a) & 0x000000ff) << 24) | (((a) & 0x0000ff00) << 8 ) | \
                              (((a) & 0x00ff0000) >> 8 ) | (((a) & 0xff000000) >> 24) )

#ifdef WIN32
#define __DIR_DELIMITER        "\\"
#else
#define __DIR_DELIMITER        "/"
#endif


typedef unsigned char   uchar;
typedef unsigned short  ushort;
typedef unsigned int    uint;

typedef struct _CHAR2 {
  char x;
  char y;
} CHAR2;
typedef struct _CHAR3 {
  char x;
  char y;
  char z;
} CHAR3;
typedef struct _CHAR4 {
  char x;
  char y;
  char z;
  char w;
} CHAR4;

typedef struct _UCHAR2 {
  uchar x;
  uchar y;
} UCHAR2;
typedef struct _UCHAR3 {
  uchar x;
  uchar y;
  uchar z;
} UCHAR3;
typedef struct _UCHAR4 {
  uchar x;
  uchar y;
  uchar z;
  uchar w;
} UCHAR4;

typedef struct _SHORT2 {
  short x;
  short y;
} SHORT2;
typedef struct _SHORT3 {
  short x;
  short y;
  short z;
} SHORT3;
typedef struct _SHORT4 {
  short x;
  short y;
  short z;
  short w;
} SHORT4;

typedef struct _USHORT2 {
  ushort x;
  ushort y;
} USHORT2;
typedef struct _USHORT3 {
  ushort x;
  ushort y;
  ushort z;
} USHORT3;
typedef struct _USHORT4 {
  ushort x;
  ushort y;
  ushort z;
  ushort w;
} USHORT4;


typedef struct _INT2 {
  int x;
  int y;
  bool operator < ( const _INT2& other) const {
    return (this->y == other.y) ? (this->x < other.x) : (this->y < other.y);
  }
} INT2;
typedef struct _INT3 {
  int x;
  int y;
  int z;
  bool operator < ( const _INT3& other) const {
    if (this->z == other.z) {
      return (this->y == other.y) ? (this->x < other.x) : (this->y < other.y);
    } else {
      return (this->z < other.z);
    }
  }
} INT3;
typedef struct _INT4 {
  int x;
  int y;
  int z;
  int w;
} INT4;

typedef struct _FLOAT2 {
  float x;
  float y;
} FLOAT2;
typedef struct _FLOAT3 {
  float x;
  float y;
  float z;
} FLOAT3;
typedef struct _FLOAT4 {
  float x;
  float y;
  float z;
  float w;
} FLOAT4;

typedef struct _DOUBLE2 {
  double x;
  double y;
} DOUBLE2;
typedef struct _DOUBLE3 {
  double x;
  double y;
  double z;
} DOUBLE3;
typedef struct _DOUBLE4 {
  double x;
  double y;
  double z;
  double w;
} DOUBLE4;

typedef struct _BOX2_CHAR   {
  char   xmin;
  char   ymin;
  char   xmax;
  char   ymax;
} BOX2_CHAR;
typedef struct _BOX2_UCHAR  {
  uchar  xmin;
  uchar  ymin;
  uchar  xmax;
  uchar  ymax;
} BOX2_UCHAR;
typedef struct _BOX2_SHORT  {
  short  xmin;
  short  ymin;
  short  xmax;
  short  ymax;
} BOX2_SHORT;
typedef struct _BOX2_USHORT {
  ushort xmin;
  ushort ymin;
  ushort xmax;
  ushort ymax;
} BOX2_USHORT;
typedef struct _BOX2_INT    {
  int    xmin;
  int    ymin;
  int    xmax;
  int    ymax;
} BOX2_INT;
typedef struct _BOX2_FLOAT  {
  float  xmin;
  float  ymin;
  float  xmax;
  float  ymax;
} BOX2_FLOAT;
typedef struct _BOX2_DOUBLE {
  double xmin;
  double ymin;
  double xmax;
  double ymax;
} BOX2_DOUBLE;

typedef struct _BOX3_CHAR   {
  char   xmin;
  char   ymin;
  char   zmin;
  char   xmax;
  char   ymax;
  char   zmax;
} BOX3_CHAR;
typedef struct _BOX3_UCHAR  {
  uchar  xmin;
  uchar  ymin;
  uchar  zmin;
  uchar  xmax;
  uchar  ymax;
  uchar  zmax;
} BOX3_UCHAR;
typedef struct _BOX3_SHORT  {
  short  xmin;
  short  ymin;
  short  zmin;
  short  xmax;
  short  ymax;
  short  zmax;
} BOX3_SHORT;
typedef struct _BOX3_USHORT {
  ushort xmin;
  ushort ymin;
  ushort zmin;
  ushort xmax;
  ushort ymax;
  ushort zmax;
} BOX3_USHORT;
typedef struct _BOX3_INT    {
  int    xmin;
  int    ymin;
  int    zmin;
  int    xmax;
  int    ymax;
  int    zmax;
} BOX3_INT;
typedef struct _BOX3_FLOAT  {
  float  xmin;
  float  ymin;
  float  zmin;
  float  xmax;
  float  ymax;
  float  zmax;
} BOX3_FLOAT;
typedef struct _BOX3_DOUBLE {
  double xmin;
  double ymin;
  double zmin;
  double xmax;
  double ymax;
  double zmax;
} BOX3_DOUBLE;


inline UCHAR2 MAKE_UCHAR2(uchar x, uchar y) {
  UCHAR2 value;
  value.x = x, value.y = y;
  return value;
}
inline UCHAR3 MAKE_UCHAR3(uchar x, uchar y, uchar z) {
  UCHAR3 value;
  value.x = x, value.y = y, value.z = z;
  return value;
}
inline UCHAR4 MAKE_UCHAR4(uchar x, uchar y, uchar z, uchar w) {
  UCHAR4 value;
  value.x = x, value.y = y, value.z = z, value.w = w;
  return value;
}

inline USHORT2 MAKE_USHORT2(ushort x, ushort y) {
  USHORT2 value;
  value.x = x, value.y = y;
  return value;
}
inline USHORT3 MAKE_USHORT3(ushort x, ushort y, ushort z) {
  USHORT3 value;
  value.x = x, value.y = y, value.z = z;
  return value;
}
inline USHORT4 MAKE_USHORT4(ushort x, ushort y, ushort z, ushort w) {
  USHORT4 value;
  value.x = x, value.y = y, value.z = z, value.w = w;
  return value;
}

inline INT2 MAKE_INT2(int x, int y) {
  INT2 value;
  value.x = x, value.y = y;
  return value;
}
inline INT3 MAKE_INT3(int x, int y, int z) {
  INT3 value;
  value.x = x, value.y = y, value.z = z;
  return value;
}
inline INT4 MAKE_INT4(int x, int y, int z, int w) {
  INT4 value;
  value.x = x, value.y = y, value.z = z, value.w = w;
  return value;
}

inline FLOAT2 MAKE_FLOAT2(float x, float y) {
  FLOAT2 value;
  value.x = x, value.y = y;
  return value;
}
inline FLOAT3 MAKE_FLOAT3(float x, float y, float z) {
  FLOAT3 value;
  value.x = x, value.y = y, value.z = z;
  return value;
}
inline FLOAT4 MAKE_FLOAT4(float x, float y, float z, float w) {
  FLOAT4 value;
  value.x = x, value.y = y, value.z = z, value.w = w;
  return value;
}

inline DOUBLE2 MAKE_DOUBLE2(double x, double y) {
  DOUBLE2 value;
  value.x = x, value.y = y;
  return value;
}
inline DOUBLE3 MAKE_DOUBLE3(double x, double y, double z) {
  DOUBLE3 value;
  value.x = x, value.y = y, value.z = z;
  return value;
}
inline DOUBLE4 MAKE_DOUBLE4(double x, double y, double z, float w) {
  DOUBLE4 value;
  value.x = x, value.y = y, value.z = z, value.w = w;
  return value;
}

inline float  DOT_FLOAT2(FLOAT2& a, FLOAT2& b) {
  return (a.x * b.x + a.y * b.y);
}
inline float  DOT_FLOAT3(FLOAT3& a, FLOAT3& b) {
  return (a.x * b.x + a.y * b.y + a.z * b.z);
}
inline float  DOT_FLOAT4(FLOAT4& a, FLOAT4& b) {
  return (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}

inline double LENGTH_INT2(INT2& a) {
  return sqrt((double)(a.x * a.x + a.y * a.y));
}
inline double LENGTH_INT3(INT3& a) {
  return sqrt((double)(a.x * a.x + a.y * a.y + a.z * a.z));
}
inline double LENGTH_INT4(INT4& a) {
  return sqrt((double)(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w));
}
inline double LENGTH_FLOAT2(FLOAT2& a) {
  return sqrt((double)(a.x * a.x + a.y * a.y));
}
inline double LENGTH_FLOAT3(FLOAT3& a) {
  return sqrt((double)(a.x * a.x + a.y * a.y + a.z * a.z));
}
inline double LENGTH_FLOAT4(FLOAT4& a) {
  return sqrt((double)(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w));
}

inline INT2   ADD_INT2(INT2& a, INT2& b) {
  INT2 result;
  result.x = a.x + b.x, result.y = a.y + b.y;
  return result;
}
inline INT3   ADD_INT3(INT3& a, INT3& b) {
  INT3 result;
  result.x = a.x + b.x, result.y = a.y + b.y, result.z = a.z + b.z;
  return result;
}
inline INT4   ADD_INT4(INT4& a, INT4& b) {
  INT4 result;
  result.x = a.x + b.x, result.y = a.y + b.y, result.z = a.z + b.z, result.w = a.w + b.w;
  return result;
}
inline FLOAT2 ADD_FLOAT2(FLOAT2& a, FLOAT2& b) {
  FLOAT2 result;
  result.x = a.x + b.x, result.y = a.y + b.y;
  return result;
}
inline FLOAT3 ADD_FLOAT3(FLOAT3& a, FLOAT3& b) {
  FLOAT3 result;
  result.x = a.x + b.x, result.y = a.y + b.y, result.z = a.z + b.z;
  return result;
}
inline FLOAT4 ADD_FLOAT4(FLOAT4& a, FLOAT4& b) {
  FLOAT4 result;
  result.x = a.x + b.x, result.y = a.y + b.y, result.z = a.z + b.z, result.w = a.w + b.w;
  return result;
}

inline INT2   SUBTRACT_INT2(INT2& a, INT2& b) {
  INT2 result;
  result.x = a.x - b.x, result.y = a.y - b.y;
  return result;
}
inline INT3   SUBTRACT_INT3(INT3& a, INT3& b) {
  INT3 result;
  result.x = a.x - b.x, result.y = a.y - b.y, result.z = a.z - b.z;
  return result;
}
inline INT4   SUBTRACT_INT4(INT4& a, INT4& b) {
  INT4 result;
  result.x = a.x - b.x, result.y = a.y - b.y, result.z = a.z - b.z, result.w = a.w - b.w;
  return result;
}
inline FLOAT2 SUBTRACT_FLOAT2(FLOAT2& a, FLOAT2& b) {
  FLOAT2 result;
  result.x = a.x - b.x, result.y = a.y - b.y;
  return result;
}
inline FLOAT3 SUBTRACT_FLOAT3(FLOAT3& a, FLOAT3& b) {
  FLOAT3 result;
  result.x = a.x - b.x, result.y = a.y - b.y, result.z = a.z - b.z;
  return result;
}
inline FLOAT4 SUBTRACT_FLOAT4(FLOAT4& a, FLOAT4& b) {
  FLOAT4 result;
  result.x = a.x - b.x, result.y = a.y - b.y, result.z = a.z - b.z, result.w = a.w - b.w;
  return result;
}

inline BOX2_CHAR   MAKE_BOX2_CHAR(char xmin, char ymin, char xmax, char ymax)           {
  BOX2_CHAR   value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}
inline BOX2_UCHAR  MAKE_BOX2_UCHAR(uchar xmin, uchar ymin, uchar xmax, uchar ymax)      {
  BOX2_UCHAR  value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}
inline BOX2_SHORT  MAKE_BOX2_SHORT(short xmin, short ymin, short xmax, short ymax)      {
  BOX2_SHORT  value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}
inline BOX2_USHORT MAKE_BOX2_USHORT(ushort xmin, ushort ymin, ushort xmax, ushort ymax) {
  BOX2_USHORT value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}
inline BOX2_INT    MAKE_BOX2_INT(int xmin, int ymin, int xmax, int ymax)                {
  BOX2_INT    value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}
inline BOX2_FLOAT  MAKE_BOX2_FLOAT(float xmin, float ymin, float xmax, float ymax)      {
  BOX2_FLOAT  value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}
inline BOX2_DOUBLE MAKE_BOX2_DOUBLE(double xmin, double ymin, double xmax, double ymax) {
  BOX2_DOUBLE value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.xmax = xmax;
  value.ymax = ymax;
  return value;
}

inline BOX3_CHAR   MAKE_BOX3_CHAR(char xmin, char ymin, char zmin, char xmax, char ymax, char zmax)               {
  BOX3_CHAR   value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}
inline BOX3_UCHAR  MAKE_BOX3_UCHAR(uchar xmin, uchar ymin, uchar zmin, uchar xmax, uchar ymax, uchar zmax)        {
  BOX3_UCHAR  value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}
inline BOX3_SHORT  MAKE_BOX3_SHORT(short xmin, short ymin, short zmin, short xmax, short ymax, short zmax)        {
  BOX3_SHORT  value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}
inline BOX3_USHORT MAKE_BOX3_USHORT(ushort xmin, ushort ymin, ushort zmin, ushort xmax, ushort ymax, ushort zmax) {
  BOX3_USHORT value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}
inline BOX3_INT    MAKE_BOX3_INT(int xmin, int ymin, int zmin, int xmax, int ymax, int zmax)                      {
  BOX3_INT    value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}
inline BOX3_FLOAT  MAKE_BOX3_FLOAT(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)        {
  BOX3_FLOAT  value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}
inline BOX3_DOUBLE MAKE_BOX3_DOUBLE(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax) {
  BOX3_DOUBLE value;
  value.xmin = xmin;
  value.ymin = ymin;
  value.zmin = zmin;
  value.xmax = xmax;
  value.ymax = ymax;
  value.zmax = zmax;
  return value;
}



inline void __swapByteOrder16(ushort& us) {
  us = ((us >> 8) | (us << 8));
}

inline void __swapByteOrder32(uint& ui) {
  ui = ((ui >> 24) | ((ui<<8) & 0x00FF0000) | ((ui>>8) & 0x0000FF00) | (ui << 24));
}

inline unsigned long __seedmix(unsigned long a, unsigned long b, unsigned long c) {
  a=a-b;
  a=a-c;
  a=a^(c >> 13);
  b=b-c;
  b=b-a;
  b=b^(a << 8);
  c=c-a;
  c=c-b;
  c=c^(b >> 13);
  a=a-b;
  a=a-c;
  a=a^(c >> 12);
  b=b-c;
  b=b-a;
  b=b^(a << 16);
  c=c-a;
  c=c-b;
  c=c^(b >> 5);
  a=a-b;
  a=a-c;
  a=a^(c >> 3);
  b=b-c;
  b=b-a;
  b=b^(a << 10);
  c=c-a;
  c=c-b;
  c=c^(b >> 15);
  return c;
}

inline double __factorial(int n) {
  if (n == 0 || n == 1) {
    return 1;
  }

  double fac = 1;
  for (int m = n; m > 0; m--) {
    fac *= m;
  }
  return fac;
}

inline double __gamma(int n) { // for integer value only, otherwise use std::tgamma
  return __factorial(n - 1);
}

inline double __uincgamma(double x, int n) {
  double esum = 0;
  for (int k = 0; k < n; k++) {
    esum += pow(x, (double)k) / (double)__factorial(k);
  }
  return ((__factorial(n - 1) * exp(-x) * esum) / __gamma(n));
}

inline double __lincgamma(double x, int n) {
  double esum = 0;
  for (int k = 0; k < n; k++) {
    esum += pow(x, (double)k) / (double)__factorial(k);
  }
  return ((__factorial(n - 1) * (1 - (exp(-x) * esum))) / __gamma(n));
}

template <class T>
inline void __normalizeVector(T *vector, int n) {
  double sum = 0;
  for (int k = 0; k < n; k++) {
    sum += vector[k];
  }
  for (int k = 0; k < n; k++) {
    vector[k] = vector[k] / sum;
  }
}



#endif // DATATYPE_HPP_INCLUDED
