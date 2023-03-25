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
#ifndef LBANN_UTILS_OUTPUT_HELPERS_HPP_INCLUDED
#define LBANN_UTILS_OUTPUT_HELPERS_HPP_INCLUDED

#include <iostream>

namespace lbann {

/** @brief Roughly determines if the stream points to a nice
 *         terminal (is a terminal, supports color).
 */
bool is_good_terminal(std::ostream& os) noexcept;

/** @brief Gets the dimensions of the terminal, if available.
 *
 *  If the stream can be determined to be using the terminal for
 *  output, this will further try to determine the dimensions in
 *  characters of the the terminal window. The method for determining
 *  this is unspecified and likely to not be generally portable.
 *
 *  If the stream cannot be determined to be a terminal, or if its
 *  dimensions cannot be resolved, the returned size is {0,0}.
 *
 *  Note that the dimensions are returned as {num_rows, num_cols}.
 */
std::pair<unsigned short, unsigned short>
get_window_size(std::ostream& os) noexcept;

/** @brief A simple utility to replace the tail end of a long string
 *         with an ellipsis.
 */
std::string truncate_to_width(std::string const& str, size_t max_len);

/** @defgroup ansi_stream ANSI stream manipulations
 *
 *  These utilities manipulate streams using ANSI CSIs to "modify
 *  streams". Note that these just inject the CSI characters into the
 *  stream; nothing about the stream itself has actually changed.
 *
 *  Also note that these are supposed to be stupid and simple. It is
 *  the user's responsibility to make sure the stream in question will
 *  process the ANSI CSIs correctly; otherwise, weird character
 *  sequences will appear in the stream output. That is, these should
 *  not be used to write to files, and they should not be used for
 *  terminal outputs if the terminal does not support color.
 */
/// @{

/** @brief Remove ANSI CSIs from the string. */
std::string strip_ansi_csis(std::string const& input);

/** @brief Turn the ANSI foreground color output black. */
std::ostream& black(std::ostream&);

/** @brief Turn the ANSI foreground color output red. */
std::ostream& red(std::ostream&);

/** @brief Turn the ANSI foreground color output green. */
std::ostream& green(std::ostream&);

/** @brief Turn the ANSI foreground color output yellow. */
std::ostream& yellow(std::ostream&);

/** @brief Turn the ANSI foreground color output blue. */
std::ostream& blue(std::ostream&);

/** @brief Turn the ANSI foreground color output magenta. */
std::ostream& magenta(std::ostream&);

/** @brief Turn the ANSI foreground color output cyan. */
std::ostream& cyan(std::ostream&);

/** @brief Turn the ANSI foreground color output white. */
std::ostream& white(std::ostream&);

/** @brief Turn the ANSI background color output black. */
std::ostream& bgblack(std::ostream&);

/** @brief Turn the ANSI background color output red. */
std::ostream& bgred(std::ostream&);

/** @brief Turn the ANSI background color output green. */
std::ostream& bggreen(std::ostream&);

/** @brief Turn the ANSI background color output yellow. */
std::ostream& bgyellow(std::ostream&);

/** @brief Turn the ANSI background color output blue. */
std::ostream& bgblue(std::ostream&);

/** @brief Turn the ANSI background color output magenta. */
std::ostream& bgmagenta(std::ostream&);

/** @brief Turn the ANSI background color output cyan. */
std::ostream& bgcyan(std::ostream&);

/** @brief Turn the ANSI background color output white. */
std::ostream& bgwhite(std::ostream&);

/** @brief Reset the ANSI color to the default. */
std::ostream& nocolor(std::ostream&);

/** @brief Clear remaining characters in the line. */
std::ostream& clearline(std::ostream&);

///@}

} // namespace lbann

#endif // LBANN_UTILS_OUTPUT_HELPERS_HPP_INCLUDED
