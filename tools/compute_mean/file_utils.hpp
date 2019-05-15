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
////////////////////////////////////////////////////////////////////////////////

#ifndef _TOOLS_COMPUTE_MEAN_FILE_UTILS_HPP_
#define _TOOLS_COMPUTE_MEAN_FILE_UTILS_HPP_
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "lbann/utils/file_utils.hpp"

namespace tools_compute_mean {

/**
 * Write the contents in a given buffer into a binary file.
 * T: type of the buf, which can be std::vector<unsigned char> for binary file
 *    std::string for text file.
 */
template<typename T = std::vector<unsigned char> >
inline void write_file(const std::string filename, const T& buf) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  file.write((const char *) buf.data(), buf.size() * sizeof(unsigned char));
  file.close();
}


/**
 * Write the contents in a given buffer into a binary file.
 * The buffer is given as a pointer and the size in bytes.
 */
inline void write_file(const std::string filename, const unsigned char *buf, const size_t size) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  file.write((const char *) buf, size);
  file.close();
}

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_FILE_UTILS_HPP_
