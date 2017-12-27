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

bool parse_path(const std::string& path, std::string& dir, std::string& basename);
std::string get_ext_name(const std::string file_name);
std::string get_basename_without_ext(const std::string file_name);
std::string add_delimiter(const std::string dir);

bool check_if_file_exists(const std::string& filename);
bool create_dir(const std::string output_dir);

} // end of namespace lbann
#endif // _LBANN_FILE_UTILS_HPP_
