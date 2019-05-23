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

#ifndef LBANN_UTILS_GLOB_HPP
#define LBANN_UTILS_GLOB_HPP

#include <glob.h>
#include <vector>
#include <string>
#include "lbann/utils/exception.hpp"

namespace lbann {

/**
 * Wrapper around glob, which searches for paths matching pattern according to
 * the shell.
 * Note this does not do tilde expansion.
 */
inline std::vector<std::string> glob(const std::string& pattern) {
  glob_t pglob;
  int r = glob(pattern.c_str(), 0, nullptr, &pglob);
  if (r != 0) {
    // Either an error or no match.
    if (r == GLOB_NOMATCH) {
      return {};
    } else {
      throw lbann_exception("glob error");
    }
  }
  std::vector<std::string> results(pglob.gl_pathc);
  for (size_t i = 0; i < pglob.gl_pathc; ++i) {
    results[i] = std::string(pglob.gl_pathv[i]);
  }
  globfree(&pglob);
  return results;
}

}  // namespace lbann

#endif  // LBANN_UTILS_GLOB_HPP
