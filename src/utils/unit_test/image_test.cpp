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
// MUST include this
#include "Catch2BasicSupport.hpp"

// File being tested
#include <lbann/utils/image.hpp>

// Hide by default because this will create a file.
TEST_CASE("Testing image utils", "[.image-utils][utilities]")
{
  SECTION("JPEG")
  {
    std::string filename = "test.jpg";
    lbann::CPUMat image;
    // Make this a 3-channel image.
    image.Resize(3 * 32 * 32, 1);
    {
      lbann::DataType* buf = image.Buffer();
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t col = 0; col < 32; ++col) {
          for (size_t row = 0; row < 32; ++row) {
            const size_t i = channel * 32 * 32 + row + col * 32;
            if (row == col) {
              buf[i] = 1.0f;
            }
            else {
              buf[i] = 0.0f;
            }
          }
        }
      }
    }
    SECTION("save image")
    {
      std::vector<size_t> dims = {3, 32, 32};
      REQUIRE_NOTHROW(lbann::save_image(filename, image, dims));
    }
    SECTION("load image")
    {
      El::Matrix<uint8_t> loaded_image;
      std::vector<size_t> dims;
      REQUIRE_NOTHROW(lbann::load_image(filename, loaded_image, dims));
      REQUIRE(dims.size() == 3);
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 32);
      REQUIRE(dims[2] == 32);
      const uint8_t* buf = loaded_image.LockedBuffer();
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t col = 0; col < 32; ++col) {
          for (size_t row = 0; row < 32; ++row) {
            const size_t i = 3 * (col + row * 32) + channel;
            if (row == col) {
              REQUIRE(buf[i] == 255);
            }
            // Turns out JPEG doesn't encode every pixel to exactly 0.
            else {
              REQUIRE(buf[i] <= 1);
            }
          }
        }
      }
    }
  }
}
