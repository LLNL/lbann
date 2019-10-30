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

#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/any.hpp"
#include "lbann/utils/exception.hpp"

#include <clara.hpp>

#include <iostream>
#include <string>

namespace lbann
{
namespace utils
{

argument_parser::argument_parser()
{
  params_["print help"] = false;
  parser_ |= clara::ExeName(exe_name_);
  parser_ |= clara::Help(utils::any_cast<bool&>(params_["print help"]));

  // Work around a bug in Clara logic
  parser_.m_exeName.set(exe_name_);
}

void argument_parser::parse(int argc, char const* const argv[])
{
  parse_no_finalize(argc, argv);
  finalize();
}

void argument_parser::parse_no_finalize(int argc, char const* const argv[])
{
  clara::Args args(argc, argv);

  auto parse_result = parser_.parse(clara::Args(argc, argv));
  if (!parse_result)
    throw parse_error(
      lbann::build_string(
        "Arguments could not be parsed.\n\nMessage: ",
        parse_result.errorMessage()));
}

void argument_parser::finalize() const
{
  if (!help_requested() && required_.size())
    throw missing_required_arguments(required_);
}

auto argument_parser::add_flag(
  std::string const& name,
  std::initializer_list<std::string> cli_flags,
  std::string const& description)
  -> readonly_reference<bool>
{
  params_[name] = false;
  auto& param_ref = any_cast<bool&>(params_[name]);
  clara::Opt option(param_ref);
  for (auto const& f : cli_flags)
    option[f];
  parser_ |= option(description).optional();
  return param_ref;
}

std::string const& argument_parser::get_exe_name() const noexcept
{
  return exe_name_;
}

bool argument_parser::help_requested() const
{
  return utils::any_cast<bool>(params_.at("print help"));
}

void argument_parser::print_help(std::ostream& out) const
{
  out << parser_ << std::endl;
}

}// namespace utils

utils::argument_parser& global_argument_parser()
{
  static utils::argument_parser args;
  return args;
}

}// namespace lbann

std::ostream& operator<<(std::ostream& os,
                         lbann::utils::argument_parser const& parser)
{
  parser.print_help(os);
  return os;
}
