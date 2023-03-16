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

/// @todo Rename this file to file.hpp

#ifndef LBANN_UTILS_FILE_HPP_INCLUDED
#define LBANN_UTILS_FILE_HPP_INCLUDED

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace lbann {

struct path_delimiter
{
  static const std::string characters;
  static std::string preferred() { return std::string(1, characters[0]); }
  static bool check(const char ch)
  {
    return (characters.find(ch) != std::string::npos);
  }
  bool operator()(const char ch) const
  {
    return (characters.find(ch) != std::string::npos);
  }
};

/// Tokenize a string into integers by an ordered sequence of delimiter
/// characters.
std::vector<int> get_tokens(std::string str, const std::vector<char> delims);
/// Tokenize a string into substrings by set of delimiter characters.
std::vector<std::string> get_tokens(const std::string str,
                                    const std::string delims = " :;\t\r\n");

/** @todo Deprecated. Use @c lbann::file::extract_parent_directory and
 *  @c lbann::file::extract_base_name instead. */
bool parse_path(const std::string& path,
                std::string& dir,
                std::string& basename);
std::string get_ext_name(const std::string file_name);
std::string get_basename_without_ext(const std::string file_name);
std::string add_delimiter(const std::string dir);
std::string modify_file_name(const std::string file_name,
                             const std::string tag,
                             const std::string new_ext = "");

/** @todo Deprecated. Use @c lbann::file::file_exists instead. */
bool check_if_file_exists(const std::string& filename);
/** @todo Deprecated. Use @c lbann::file::directory_exists instead. */
bool check_if_dir_exists(const std::string& dirname);
/** @todo Deprecated. Use @c lbann::file::make_directory instead. */
bool create_dir(const std::string output_dir);

bool load_file(const std::string filename,
               std::vector<char>& buf,
               bool append = false);

inline void __swapEndianInt(unsigned int& ui)
{
  ui = ((ui >> 24) | ((ui << 8) & 0x00FF0000) | ((ui >> 8) & 0x0000FF00) |
        (ui << 24));
}

// The generic approach
template <typename T>
std::basic_string<T> pad(const std::basic_string<T>& s,
                         typename std::basic_string<T>::size_type n,
                         T c)
{
  if (n > s.length()) {
    std::string t = s;
    t.insert(t.begin(), n - t.length(), c);
    return t;
  }
  else {
    return s;
  }
}

namespace file {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace details {

template <typename BaseTypeT, typename... PathRepresentationType>
std::string join_path_impl(BaseTypeT const& base,
                           PathRepresentationType&&... rest);

// Base cases
inline std::string const& join_path_impl(std::string const& x) { return x; }
inline std::string join_path_impl(char const* const x)
{
  return std::string(x);
}

// Recursive cases
template <typename... PathRepresentationType>
std::string join_path_impl(std::string const& base,
                           PathRepresentationType&&... rest)
{
  return base + "/" +
         join_path_impl(std::forward<PathRepresentationType>(rest)...);
}

template <typename BaseType, typename... PathRepresentationType>
std::string join_path_impl(BaseType const& base,
                           PathRepresentationType&&... rest)
{
  return std::string(base) + "/" +
         join_path_impl(std::forward<PathRepresentationType>(rest)...);
}

} // namespace details
#endif // !defined(DOXYGEN_SHOULD_SKIP_THIS)

/** @brief Concatenate all paths, injecting path separators between
 *         each argument.
 *  @todo Assert every
 */
template <typename... PathNameType>
std::string join_path(PathNameType&&... paths)
{
  return details::join_path_impl(std::forward<PathNameType>(paths)...);
}

/** @brief Wrapper around @c dirname.
 *
 *  Deletes any suffix beginning with the last '/' (ignoring trailing
 *  slashes).
 */
std::string extract_parent_directory(const std::string& path);

/** @brief Wrapper around @c basename.
 *
 *  Deletes any prefix up to the last '/' (ignoring trailing slashes).
 */
std::string extract_base_name(const std::string& path);

/** @brief Check if file exists. */
bool file_exists(const std::string& path);

/** @brief Check if directory exists. */
bool directory_exists(const std::string& path);

/** @brief Create directory if needed.
 *
 *  Does nothing if directory already exists. Parent directories are
 *  created recursively if needed. This is thread-safe (provided @c
 *  mkdir is thread-safe).
 */
void make_directory(const std::string& path);

void remove_multiple_slashes(std::string& str);

} // namespace file

} // namespace lbann

#endif // LBANN_UTILS_FILE_HPP_INCLUDED
