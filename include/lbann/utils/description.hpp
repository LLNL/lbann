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

#ifndef LBANN_UTILS_DESCRIPTION_HPP
#define LBANN_UTILS_DESCRIPTION_HPP

#include <string>
#include <vector>
#include <ostream>
#include <sstream>

namespace lbann {

/** @brief Generates nicely formatted description messages.
 *
 *  Messages have hanging indentation and can be output to an output
 *  stream like @c std::cout. For example:
 *
@verbatim
Title
  Some numerical field: 12.3
  A boolean parameter: true
  Miscellaneous statement
@endverbatim
 */
class description {
public:

  /** @param title  First line in description message.
   */
  description(std::string title = "");

  /** Set first line in description message. */
  void set_title(std::string title);

  /** Print description to stream. */
  friend std::ostream& operator<<(std::ostream& os,
                                  const description& desc);

  /** Add new line. */
  void add(std::string line);

  /** Add new line describing a field value.
   *
   *  The line is formatted:
   *
   *  @verbatim <field>: <value> @endverbatim
   */
  template <typename T>
  void add(std::string field, T value) {
    std::stringstream ss;
    ss.setf(std::ios_base::boolalpha);
    ss << field << ": " << value;
    add(ss.str());
  }

  /** Insert a nested @c description.
   *
   *  The indentation in @c desc is combined with the current
   *  indentation. For instance:
   *
@verbatim
Outer description
  Some numerical field: 12.3
    Nested description
      This: abc
      That: 123
@endverbatim
   */
  void add(const description& desc);

private:

  /** First line of message.
   *
   *  When printed, this line isn't indented.
   */
  std::string m_title;

  /** Lines in message (excluding first line).
   *
   *  When printed, each line is indented.
   */
  std::vector<std::string> m_lines;

};

} // namespace lbann

#endif // LBANN_UTILS_DESCRIPTION_HPP
