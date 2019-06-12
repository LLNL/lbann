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

#ifndef _TOOLS_COMPUTE_MEAN_TEXT_READ_HPP_
#define _TOOLS_COMPUTE_MEAN_TEXT_READ_HPP_
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <exception>


namespace tools_compute_mean {

template <typename T, typename F>
inline bool read_text_stream(const std::string& fileName, std::istream& textStream,
                             T& container, F func, const unsigned int numHeaderLines = 0u);
/**
 * Main interface to read a formatted text data file.
 * User provide the file name and the line parsing function, and then
 * the reader reads each line and put an item into the container.
 * Each field in a line must be separated by comma, tabs, or spaces.
 * The character '#' is reserved for commenting. The part of a line that
 * follows '#' will be ignored.
 *
 * input:
 * - fileName: name of the file to read
 * - func: line parsing function with the interface as:
 *         bool func(const std::string &line, ContainerType& container)
 * - numHeaderLines: the number of header lines to skip
 *
 * output:
 * - container: container such as a vector or list. The line parsing
 *   function provide by a user will add an entry as it reads a line
 */
template <typename T, typename F>
inline bool read_text_file(const std::string& fileName, T& container, F func,
                           const unsigned int numHeaderLines = 0u) {
  std::ifstream textFile(fileName.c_str(), std::ios_base::in);
  bool ok = read_text_stream<T,F>(fileName, textFile, container, func, numHeaderLines);
  textFile.close();
  return ok;
}

/**
 * Similar to read_text_file except that this reads text from a std::string,
 * which is preloaded but not yet parsed.
 */
template <typename T, typename F>
inline bool read_text(const std::string& text, T& container, F func,
                      const unsigned int numHeaderLines = 0u) {
  std::istringstream iss(text);
  bool ok = read_text_stream<T,F>(std::string("text buffer"), iss, container, func, numHeaderLines);
  return ok;
}


template <typename T, typename F>
inline bool parse_line(std::string& line, T& container, F func) {
  std::size_t pos = line.find_first_not_of(", \t\r\n");
  if ((pos == std::string::npos) || (line[pos] == '#')) {
    return true; // skipping empty lines or comments
  }
  return func(line, container);
}

/// Reads each line from an input stream, and parse the content out into container using func.
template <typename T, typename F>
inline bool read_text_stream(const std::string& fileName, std::istream& textStream,
                             T& container, F func, const unsigned int numHeaderLines) {
  std::string line;

  for(unsigned int i=0u; i <= numHeaderLines; ++i) {
    if (!std::getline(textStream, line)) {
      std::string msg = "Failed to read " + fileName + "\n";
      throw(msg.c_str());
      return false;
    }
  }

  while (textStream) { // parse the data file one line at a time until the end of the file
    if (!(parse_line<T,F>(line, container, func))) {
      std::string msg = "Failed to parse the line '" + line + "'\n";
      throw(msg.c_str());
      return false;
    }
    std::getline(textStream, line);
  }

  return true;
}

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_TEXT_READ_HPP_
