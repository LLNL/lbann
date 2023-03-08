////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_TRANSFORMS_VISION_UNIT_TEST_HELPER
#define LBANN_TRANSFORMS_VISION_UNIT_TEST_HELPER

inline void
apply_elementwise(El::Matrix<uint8_t>& mat,
                  El::Int height,
                  El::Int width,
                  El::Int channels,
                  std::function<void(uint8_t&, El::Int, El::Int, El::Int)> f)
{
  uint8_t* buf = mat.Buffer();
  for (El::Int channel = 0; channel < channels; ++channel) {
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        f(buf[channels * (col + row * width) + channel], row, col, channel);
      }
    }
  }
}

inline void identity(El::Matrix<uint8_t>& mat,
                     El::Int height,
                     El::Int width,
                     El::Int channels = 1)
{
  mat.Resize(height * width * channels, 1);
  apply_elementwise(mat,
                    height,
                    width,
                    channels,
                    [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                      x = (row == col) ? 1 : 0;
                    });
}

inline void zeros(El::Matrix<uint8_t>& mat,
                  El::Int height,
                  El::Int width,
                  El::Int channels = 1)
{
  mat.Resize(height * width * channels, 1);
  uint8_t* buf = mat.Buffer();
  for (El::Int i = 0; i < height * width * channels; ++i) {
    buf[i] = 0;
  }
}

inline void ones(El::Matrix<uint8_t>& mat,
                 El::Int height,
                 El::Int width,
                 El::Int channels = 1)
{
  mat.Resize(height * width * channels, 1);
  uint8_t* buf = mat.Buffer();
  for (El::Int i = 0; i < height * width * channels; ++i) {
    buf[i] = 1;
  }
}

inline void print(const El::Matrix<uint8_t>& mat,
                  El::Int height,
                  El::Int width,
                  El::Int channels = 1)
{
  const uint8_t* buf = mat.LockedBuffer();
  for (El::Int channel = 0; channel < channels; ++channel) {
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        std::cout << ((int)buf[channels * (col + row * width) + channel])
                  << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "--" << std::endl;
  }
}

#endif // LBANN_TRANSFORMS_VISION_UNIT_TEST_HELPER
