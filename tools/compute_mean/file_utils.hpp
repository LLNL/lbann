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
////////////////////////////////////////////////////////////////////////////////

#ifndef _TOOLS_COMPUTE_MEAN_FILE_UTILS_HPP_
#define _TOOLS_COMPUTE_MEAN_FILE_UTILS_HPP_
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>

#define __DIR_DELIMITER "/"

bool parse_path(const std::string& path, std::string& dir, std::string& basename);
std::string get_ext_name(const std::string file_name);
std::string get_basename_without_ext(const std::string file_name);
std::string add_delimiter(const std::string dir);

bool check_if_file_exists(const std::string& filename);
bool create_dir(const std::string output_dir);


/**
 * Read a binary file into a buffer.
 * T: type of the buf, which can be std::vector<unsigned char> for binary file
 *    std::string for text file.
 */
template <typename T = std::vector<unsigned char> >
inline bool read_file(const std::string filename, T& buf) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    return false;
  }

  file.unsetf(std::ios::skipws);

  file.seekg(0, std::ios::end);
  const std::streampos file_size = file.tellg();

  file.seekg(0, std::ios::beg);

  buf.reserve(file_size);

  buf.insert(buf.begin(),
             std::istream_iterator<unsigned char>(file),
             std::istream_iterator<unsigned char>());

  return true;
}


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

#endif // _TOOLS_COMPUTE_MEAN_FILE_UTILS_HPP_
