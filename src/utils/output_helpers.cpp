////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include "lbann/utils/output_helpers.hpp"

#include <regex>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <unistd.h> // fileno, isatty
#if __has_include(<sys/ioctl.h>)
#define LBANN_HAS_SYS_IOCTL
#include <sys/ioctl.h>
#endif

#define LBANN_ANSI_FG_BLACK "\x1b[30m"
#define LBANN_ANSI_FG_RED "\x1b[31m"
#define LBANN_ANSI_FG_GREEN "\x1b[32m"
#define LBANN_ANSI_FG_YELLOW "\x1b[33m"
#define LBANN_ANSI_FG_BLUE "\x1b[34m"
#define LBANN_ANSI_FG_MAGENTA "\x1b[35m"
#define LBANN_ANSI_FG_CYAN "\x1b[36m"
#define LBANN_ANSI_FG_WHITE "\x1b[37m"

#define LBANN_ANSI_BG_BLACK "\x1b[40m"
#define LBANN_ANSI_BG_RED "\x1b[41m"
#define LBANN_ANSI_BG_GREEN "\x1b[42m"
#define LBANN_ANSI_BG_YELLOW "\x1b[43m"
#define LBANN_ANSI_BG_BLUE "\x1b[44m"
#define LBANN_ANSI_BG_MAGENTA "\x1b[45m"
#define LBANN_ANSI_BG_CYAN "\x1b[46m"
#define LBANN_ANSI_BG_WHITE "\x1b[47m"

#define LBANN_ANSI_RESET_COLOR "\x1b[0m"
#define LBANN_ANSI_CLEAR_TO_LINE_END "\x1b[K"

// This makes some assumptions about how std::{cout,cerr,clog} have
// been handled. In particular, it assumes that their streambuf
// objects have not been replaced.
static FILE* guess_file_obj(std::ostream& os) noexcept
{
  auto const os_buf = os.rdbuf();
  if (os_buf == std::cout.rdbuf())
    return stdout;
  if ((os_buf == std::cerr.rdbuf()) || (os_buf == std::clog.rdbuf()))
    return stderr;
  return nullptr;
}

static bool is_terminal(std::ostream& os) noexcept
{
  if (auto* f = guess_file_obj(os))
    return isatty(fileno(f));
  return false;
}

// This makes some mild assumptions about the capabilities of some
// terminal families. It also assumes that environment variable is
// correct. The worst case scenario here is that someone has some
// ASCII control sequences in their output -- what tragedy!
static bool terminal_supports_color() noexcept
{
  // Don't need more than 16 colors, so there are many options.
  std::unordered_set<std::string> const terminals_that_support_color = {
    "linux",
    "rxvt",
    "rxvt-16color",
    "xterm",
    "xterm-16color",
    "xterm-256color",
    "xterm-88color",
    "xterm-color",
  };
  if (const char* term_var = std::getenv("TERM"))
    return terminals_that_support_color.count(term_var);
  return false;
}

bool lbann::is_good_terminal(std::ostream& os) noexcept
{
  return is_terminal(os) && terminal_supports_color();
}

std::pair<unsigned short, unsigned short>
lbann::get_window_size(std::ostream& os) noexcept
{
  using PairType = std::pair<unsigned short, unsigned short>;
#ifdef LBANN_HAS_SYS_IOCTL
  if (auto* f = guess_file_obj(os)) {
    winsize w;
    if (0 == ioctl(fileno(f), TIOCGWINSZ, &w))
      return std::make_pair(w.ws_row, w.ws_col);
  }
#endif
  return PairType{0, 0};
}

/** @brief A simple utility to replace the tail end of a long string
 *         with an ellipsis.
 */
std::string lbann::truncate_to_width(std::string const& str, size_t max_len)
{
  if (str.length() > max_len)
    return str.substr(0, max_len - 3) + "...";
  else
    return str;
}

// See:
// https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_(Control_Sequence_Introducer)_sequences
// (Note: 0x7F is NOT a valid terminal character.)
std::string lbann::strip_ansi_csis(std::string const& input)
{
  static std::regex const csi_re("\x1b\[[\x20-\x3f]*[\x40-\x7e]");
  return std::regex_replace(input, csi_re, "");
}

std::ostream& lbann::black(std::ostream& os)
{
  return os << LBANN_ANSI_FG_BLACK;
}

std::ostream& lbann::red(std::ostream& os) { return os << LBANN_ANSI_FG_RED; }

std::ostream& lbann::green(std::ostream& os)
{
  return os << LBANN_ANSI_FG_GREEN;
}

std::ostream& lbann::yellow(std::ostream& os)
{
  return os << LBANN_ANSI_FG_YELLOW;
}

std::ostream& lbann::blue(std::ostream& os) { return os << LBANN_ANSI_FG_BLUE; }

std::ostream& lbann::magenta(std::ostream& os)
{
  return os << LBANN_ANSI_FG_MAGENTA;
}

std::ostream& lbann::cyan(std::ostream& os) { return os << LBANN_ANSI_FG_CYAN; }

std::ostream& lbann::white(std::ostream& os)
{
  return os << LBANN_ANSI_FG_WHITE;
}

std::ostream& lbann::bgblack(std::ostream& os)
{
  return os << LBANN_ANSI_FG_BLACK;
}

std::ostream& lbann::bgred(std::ostream& os) { return os << LBANN_ANSI_FG_RED; }

std::ostream& lbann::bggreen(std::ostream& os)
{
  return os << LBANN_ANSI_FG_GREEN;
}

std::ostream& lbann::bgyellow(std::ostream& os)
{
  return os << LBANN_ANSI_FG_YELLOW;
}

std::ostream& lbann::bgblue(std::ostream& os)
{
  return os << LBANN_ANSI_FG_BLUE;
}

std::ostream& lbann::bgmagenta(std::ostream& os)
{
  return os << LBANN_ANSI_FG_MAGENTA;
}

std::ostream& lbann::bgcyan(std::ostream& os)
{
  return os << LBANN_ANSI_FG_CYAN;
}

std::ostream& lbann::bgwhite(std::ostream& os)
{
  return os << LBANN_ANSI_FG_WHITE;
}

std::ostream& lbann::nocolor(std::ostream& os)
{
  return os << LBANN_ANSI_RESET_COLOR;
}

std::ostream& lbann::clearline(std::ostream& os)
{
  return os << LBANN_ANSI_CLEAR_TO_LINE_END;
}
