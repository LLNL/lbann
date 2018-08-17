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

#include "lbann/utils/file_utils.hpp"
#include <algorithm>
#include <fstream>
//#include <iostream> // std::cerr

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

/// Divide a given path into dir and basename.
bool parse_path(const std::string& path, std::string& dir, std::string& basename) {
  std::string::const_iterator nb =
    std::find_if(path.rbegin(), path.rend(), path_delimiter()).base();
  dir =  std::string(path.begin(), nb);
  basename = std::string(nb, path.end());
  if (basename.empty()) {
    return false;
  }

  return true;
}


/// Return file extention name.
std::string get_ext_name(const std::string file_name) {
  std::string dir;
  std::string basename;
  parse_path(file_name, dir, basename);

  size_t pos = basename.find_last_of('.');
  if (pos == 0u) {
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
 * i.e., given a file_name as "name.ext", a new name "name.tag.new_ext" is returned
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
    name = name + '.' + tag;
  }
  return (dir + name + '.' + ext);
}


/// Return true if a file with the given name exists.
bool check_if_file_exists(const std::string& filename) {
  std::ifstream ifile(filename);
  return static_cast<bool>(ifile);
}


#ifdef _POSIX_SOURCE
#include <sys/stat.h>
#include <errno.h>
/// Return true if a directory with the given name exists.
bool check_if_dir_exists(const std::string& dirname) {
  struct stat sb;

  return (stat(dirname.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
}
#else
bool check_if_dir_exists(const std::string& dirname) {
  return check_if_file_exists(dirname);
}
#endif


#include <cstdlib>
/**
 * Create a directory, and return true if successful.
 * If a directory with the same name already exists, simply return true.
 */
bool create_dir(const std::string dirname) {
  std::string dir = dirname;
  if (!dir.empty() && path_delimiter::check(dir.back())) {
    dir.pop_back();
  }

  if (dir.empty()) {
    return true;
  }

  const bool file_exists = check_if_file_exists(dir);

  if (file_exists) {
    if (!check_if_dir_exists(dir)) {
      return false;
    } else {
      return true;
    }
  }

  std::string cmd = std::string("mkdir -p ") + dir;
#if 1 // for ray
  char cmdstr[cmd.size()+1];
  std::copy(cmd.begin(), cmd.end(), &cmdstr[0]);
  cmdstr[cmd.size()] = '\0';
  const int r = ::system(&cmdstr[0]);
#else
  const int r = ::system(cmd.c_str());
#endif

  if (WEXITSTATUS(r) == 0x10) {
    return true;
  } else if (!check_if_dir_exists(dir)) {
    return false;
  }
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

} // end of namespace lbann
