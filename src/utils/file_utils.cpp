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
////////////////////////////////////////////////////////////////////////////////

/// @todo Rename this file to file.cpp.

#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/exception.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>

namespace lbann {

const std::string path_delimiter::characters = "/";

std::vector<int> get_tokens(std::string str, const std::vector<char> delims) {
  std::vector<int> tokens;
  size_t pos;

  for (const auto d : delims) {
    pos = str.find_first_of(d);
    if (pos == std::string::npos) {
     // std::cerr << "Not able to split " << str << " by " << d << std::endl;
      return std::vector<int>();
    }
    tokens.push_back(atoi(str.substr(0, pos).c_str()));
    str = str.substr(pos+1, str.size());
  }

  return tokens;
}

std::vector<std::string> get_tokens(const std::string str, const std::string delims) {
  std::vector<std::string> parsed;
  size_t pos_start = 0u;
  size_t pos_end = 0u;

  while ((pos_end != std::string::npos) && (pos_start != std::string::npos)) {
    pos_start = str.find_first_not_of(delims, pos_end);
    if (pos_start != std::string::npos) {
      pos_end = str.find_first_of(delims, pos_start);
      parsed.push_back(str.substr(pos_start, (pos_end-pos_start)));
    }
  }
  return parsed;
}

/// @todo Deprecated.
bool parse_path(const std::string& path, std::string& dir, std::string& basename) {
  dir = file::extract_parent_directory(path);
  basename = file::extract_base_name(path);
  return !basename.empty();
}

/// Return file extention name.
std::string get_ext_name(const std::string file_name) {
  std::string dir;
  std::string basename;
  parse_path(file_name, dir, basename);

  size_t pos = basename.find_last_of('.');
  if (pos >= basename.size()) {
    return "";  // hidden file
  }
  return basename.substr(pos+1, basename.size());
}


/// Return basename without extention.
std::string get_basename_without_ext(const std::string file_name) {
  std::string dir;
  std::string basename;
  parse_path(file_name, dir, basename);

  size_t pos = basename.find_last_of('.');
  if (pos == 0u) {
    return basename;
  }
  return basename.substr(0, pos);
}


/**
 * This automatically attaches the directory deliminator at the end of the given
 * directory as necessary.
 * If "" is given, it will do nothing
 */
std::string add_delimiter(const std::string dir) {
  if (dir.empty()) {
    return "";
  }
  std::string new_dir(dir);

  if (!path_delimiter::check(new_dir.back())) {
    new_dir.append(path_delimiter::preferred());
  }

  return new_dir;
}


/**
 * Add tag to a file name and/or change the extension.
 * i.e., given a file_name as "name.ext", a new name "name_tag.new_ext" is returned
 * If a new extension is not specified, it assumes the same.
 * To change the extension without adding a tag, set tag to a null string.
 */
std::string modify_file_name(const std::string file_name, const std::string tag, const std::string new_ext) {
  std::string dir;
  std::string name;
  bool ok = parse_path(file_name, dir, name);
  if (!ok) {
    return std::string();
  }
  std::string ext = (new_ext.empty() ? get_ext_name(name) : new_ext);
  name = get_basename_without_ext(name);

  if (!tag.empty()) {
    name = name + '_' + tag;
  }

  dir = add_delimiter(dir);
  if(!ext.empty()) {
    return (dir + name + '.' + ext);
  }else {
    return (dir + name);
  }
}

/// @todo Deprecated.
bool check_if_file_exists(const std::string& filename) {
  return file::file_exists(filename);
}

/// @todo Deprecated.
bool check_if_dir_exists(const std::string& dirname) {
  return file::directory_exists(dirname);
}

/// @todo Deprecated.
bool create_dir(const std::string dirname) {
  file::make_directory(dirname);
  return true;
}

/// Load a file into a buffer
bool load_file(const std::string filename, std::vector<char>& buf) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    return false;
  }

  file.unsetf(std::ios::skipws);

  file.seekg(0, std::ios::end);
  const std::streampos file_size = file.tellg();

  file.seekg(0, std::ios::beg);

  buf.resize(file_size);

  file.read(buf.data(), file_size);

  return true;
}

namespace file {

std::string extract_parent_directory(const std::string& path) {
  std::vector<char> buffer(path.size()+1);
  path.copy(buffer.data(), path.size());
  buffer.back() = '\0';
  return ::dirname(buffer.data());
}

std::string extract_base_name(const std::string& path) {
  std::vector<char> buffer(path.size()+1);
  path.copy(buffer.data(), path.size());
  buffer.back() = '\0';
  return ::basename(buffer.data());
}

bool file_exists(const std::string& path) {
  if (path.empty() || path == "." || path == "/") {
    return true;
  }
  struct ::stat buffer;
  return (::stat(path.c_str(), &buffer) == 0);
}

bool directory_exists(const std::string& path) {
  if (path.empty() || path == "." || path == "/") {
    return true;
  }
  struct ::stat buffer;
  return (::stat(path.c_str(), &buffer) == 0
          && S_ISDIR(buffer.st_mode));
}

void make_directory(const std::string& path) {
  if (directory_exists(path)) { return; }

  // Create parent directory if needed
  const auto& parent = extract_parent_directory(path);
  make_directory(parent);

  // Create directory
  // Note: Don't complain if the directory already exists.
  auto status = ::mkdir(path.c_str(), S_IRWXU | S_IRWXG); // chmod 770
  if (status != 0 && errno != EEXIST) {
    std::stringstream err;
    err << "failed to create directory (" << path << ") "
        << "with error " << errno << " "
        << "(" << strerror(errno) << ")";
    LBANN_ERROR(err.str());
  }

}

} // namespace file

} // namespace lbann
