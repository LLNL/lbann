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

#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/exception.hpp"

#include <clara.hpp>

#include <iostream>
#include <string>

namespace {
/** @brief Check a raw argument for a given token.
 *
 *  Returns "true" when EITHER:
 *
 *    - testtoken == rawargument is true.
 *    - rawargument has an equals sign and the substring up to (but not
 *      including) the equals sign compares equal to testtoken.
 */
bool token_match(std::string const& testtoken, std::string const& rawargument)
{
  return (rawargument.compare(0, rawargument.find('='), testtoken) == 0);
}
} // namespace

namespace lbann {
namespace utils {
void strict_parsing::handle_error(clara::detail::InternalParseResult result,
                                  clara::Parser&,
                                  std::vector<char const*>&)
{
  throw parse_error(
    lbann::build_string("Arguments could not be parsed.\n\nMessage: ",
                        result.errorMessage()));
}

void allow_extra_parameters::handle_error(
  clara::detail::InternalParseResult parse_result,
  clara::Parser& parser_,
  std::vector<char const*>& newargv)
{
  do {
    std::string const base_text = "Unrecognised token: ";
    std::string const err_text = parse_result.errorMessage();
    if ((err_text.size() > base_text.size()) &&
        (err_text.substr(0, base_text.size()).compare(base_text) == 0)) {
      auto const token = err_text.substr(base_text.size());
      auto iter = std::find_if(
        newargv.cbegin(),
        newargv.cend(),
        [&token](char const* str) { return token_match(token, str); });
      newargv.erase(newargv.cbegin() + 1,
                    (iter == newargv.cend() ? iter : iter + 1));
      parse_result = parser_.parse(clara::Args(newargv.size(), newargv.data()));
    }
    else {
      throw parse_error(
        lbann::build_string("Arguments could not be parsed.\n\nMessage: ",
                            parse_result.errorMessage()));
    }
  } while (!parse_result);
}

} // namespace utils

default_arg_parser_type& global_argument_parser()
{
  static default_arg_parser_type args;
  return args;
}

} // namespace lbann
