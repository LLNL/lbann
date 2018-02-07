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

#ifndef _LBANN_FILE_UTILS_HPP_
#define _LBANN_FILE_UTILS_HPP_
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>

namespace lbann {

struct path_delimiter {
  static const std::string characters;
  static std::string preferred() {
    return std::string(1, characters[0]);
  }
  static bool check(const char ch) {
    return (characters.find(ch) != std::string::npos);
  }
  bool operator()(const char ch) const {
    return (characters.find(ch) != std::string::npos);
  }
};

/// Tokenize a string into integers by an ordered sequence of delimiter characters.
std::vector<int> get_tokens(std::string str, const std::vector<char> delims);
/// Tokenize a string into substrings by set of delimiter characters.
std::vector<std::string> get_tokens(const std::string str, const std::string delims = " :;\t\r\n");

bool parse_path(const std::string& path, std::string& dir, std::string& basename);
std::string get_ext_name(const std::string file_name);
std::string get_basename_without_ext(const std::string file_name);
std::string add_delimiter(const std::string dir);
std::string modify_file_name(const std::string file_name, const std::string tag, const std::string new_ext="");

bool check_if_file_exists(const std::string& filename);
bool create_dir(const std::string output_dir);


/// Load a file into a buffer
template <typename T = std::vector<unsigned char> >
inline bool load_file(const std::string filename, T& buf) {
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

inline void __swapEndianInt(unsigned int& ui) {
  ui = ((ui >> 24) | ((ui<<8) & 0x00FF0000) | ((ui>>8) & 0x0000FF00) | (ui << 24));
}

} // end of namespace lbann
#endif // _LBANN_FILE_UTILS_HPP_
