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

#ifndef LBANN_UTILS_ARGUMENT_PARSER_HPP_INCLUDED
#define LBANN_UTILS_ARGUMENT_PARSER_HPP_INCLUDED

#include "lbann/utils/environment_variable.hpp"
#include "lbann/utils/exception.hpp"

#include <clara.hpp>

#include <any>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lbann {
namespace utils {

/** @class parse_error
 *  @brief std::exception subclass that is thrown if the parser
 *         can not parse the arguments.
 */
struct parse_error : std::runtime_error
{
  /** @brief Construct the exception with the string to be
   *         return by what()
   */
  template <typename T>
  parse_error(T&& what_arg) : std::runtime_error{std::forward<T>(what_arg)}
  {}
}; // parse_error

/** @class strict_parsing
 *
 *  Allows any valid subset of parameters. This will throw an
 *  exception for any error raised by the underlying parser.
 */
struct strict_parsing
{
  void handle_error(clara::detail::InternalParseResult result,
                    clara::Parser& parser,
                    std::vector<char const*>& argv);
}; // struct strict_parsing

/** @class allow_extra_parameters
 *
 *  Ignores "unknown token" errors raised by the parser and attempts
 *  to proceed until all tokens are processed or another error is
 *  detected.
 */
struct allow_extra_parameters
{
  void handle_error(clara::detail::InternalParseResult result,
                    clara::Parser& parser,
                    std::vector<char const*>& argv);
}; // struct allow_extra_parameters

/** @class argument_parser
 *  @brief Basic argument parsing with automatic help messages.
 *
 *  @section arg_parser_params Supported parameter types
 *
 *  The argument parser supports 3 types of command line parameters:
 *  flags, options, and arguments.
 *
 *  @subsection arg_parser_flags Flags
 *
 *  Flags default to "false" and toggle to "true" when they are given
 *  on the command line. It is an error to provide a value to a flag
 *  on the command line (e.g., "-flag 0"). If a flag called "-v" is
 *  tied to a variable called `verbose`, `verbose` will have default
 *  value `false`. Passing "-v" on the command line, `a.out -v`, will
 *  result in `verbose` having post-parse value `true`.
 *
 *  @subsection arg_parser_options Options
 *
 *  Options represent key-value pairs. They must take only a single
 *  value (e.g. `a.out -key value`). It is an error to omit a value
 *  for a parameter of option type (e.g., `a.out -key`). Options are
 *  strongly typed to match their default values. The string passed on
 *  the command line must be convertible to the type of the default
 *  value provided by the developer programmatically.
 *
 *  @subsection arg_parser_arguments Arguments
 *
 *  Arguments (or "positional arguments") do not name a key on the
 *  command line and are implicitly keyed by their index in the
 *  argument list. A corollary to this is that required arguments must
 *  appear before optional arguments. Arguments with each category
 *  ("required" and "optional") are keyed in the order in which they
 *  are added.
 *
 *  On command line, "optional" arguments are ordered after the
 *  "required" arguments, in the order in which they are added. For
 *  example, adding an (optional) argument called "A", then adding
 *  a required argument called "B", then adding an (optioinal)
 *  argument called "C" will require that these arguments be passed
 *  as `a.out B A C`. Since "A" and "C" are optional, it is also
 *  valid to pass `a.out B` or `a.out B A`. It is undefined
 *  behavior to pass `a.out B C`.
 *
 *  Erroneously passing `a.out B C` might be accepted by the parser
 *  if "A" and "C" have the same (or sufficiently compatible)
 *  types, but the output will not be as unexpected (the variable
 *  bound to "A" will have the value expected in "C", and the
 *  variable bound to "C" will have its default value). If "A" and
 *  "C" are not compatible types, an exception will be thrown. In
 *  the first case, the parser cannot read your mind to know if you
 *  passed things in the right order; it is the application
 *  developer's responsibility to ensure that all arguments have
 *  been added before the help message is printed, and it is the
 *  user's responsibility to consult the help message for the
 *  runtime ordering of arguments.
 *
 *  @section arg_parser_finalize Finalization
 *
 *  To accomodate the presence of required arguments with the
 *  maintenance-intensive practice of adding arguments willy-nilly
 *  (because I don't believe a PR without said terrifying
 *  capability would ever make it through), parsing of the
 *  arguments can be done two ways: with or without finalization.
 *
 *  If there are no required arguments registered in the parser,
 *  these should be equivalent. If there are required arguments,
 *  they must all have been registered with the parser and seen in
 *  the arguments given to the parse functions before
 *  finalization. Semantically, the parser must be finalized before
 *  attempting to use any of the required arguments.
 */
template <typename ErrorHandler>
class argument_parser : ErrorHandler
{
public:
  /** @name Public types */
  ///@{

  /** @brief A proxy class representing the current value associated
   *         with an option.
   *
   *  This class is best manipulated generically, through `auto`
   *  variables.
   *
   *  @tparam T The type of the held object.
   */
  template <typename T>
  class readonly_reference
  {
  public:
    readonly_reference(T& val) noexcept : ref_(val) {}
    T const& get() const noexcept { return ref_; }
    operator T const&() const noexcept { return this->get(); }

    template <typename S>
    bool operator==(S const& y) const noexcept
    {
      return this->get() == y;
    }

  private:
    T& ref_;
  }; // class readonly_reference<T>

  /** @class parse_error
   *  @brief std::exception subclass that is thrown if the parser
   *         can not parse the arguments.
   */
  struct parse_error : std::runtime_error
  {
    /** @brief Construct the exception with the string to be
     *         return by what()
     */
    template <typename T>
    parse_error(T&& what_arg) : std::runtime_error{std::forward<T>(what_arg)}
    {}
  };

  /** @class missing_required_arguments
   *  @brief std::exception subclass that is thrown if a required
   *         argument is not found.
   */
  struct missing_required_arguments : std::runtime_error
  {
    /** @brief Construct the exception with a list of the missing
     *         argument names.
     *
     *  @param[in] missing_args A container that holds the names
     *             of the missing arguments.
     */
    template <typename Container>
    missing_required_arguments(Container const& missing_args)
      : std::runtime_error{build_what_string_(missing_args)}
    {}

  private:
    template <typename Container>
    std::string build_what_string_(Container const& missing_args)
    {
      std::ostringstream oss;
      oss << "The following required arguments are missing: {";
      for (auto const& x : missing_args)
        oss << " \"" << x << "\"";
      oss << " }";
      return oss.str();
    }
  };

  ///@}

public:
  /** @name Constructors */
  ///@{

  /** @brief Create the parser */
  argument_parser();

  /** @brief Copy construction is disabled.
   *
   *  The Clara parser is not actually copyable because the references
   *  are bound to memory in the parameter map. They are not re-bound
   *  during the default copy, so the copied parser would refer to
   *  memory in the source parser. If the source parser is destroyed,
   *  these references are left dangling. It would be very hard to
   *  rebind the references on copy (it would have to be done through
   *  "std::any" object, making it all the more complicated). Rather,
   *  we don't have the other machinery to do this at this time, so
   *  copy is disabled. This issue should not apply to move.
   */
  argument_parser(argument_parser const&) = delete;

  /** @brief Copy assignment is disabled. */
  argument_parser& operator=(argument_parser const&) = delete;

  /** @brief Move constructor */
  argument_parser(argument_parser&&) = default;

  /** @brief Move assignment operator */
  argument_parser& operator=(argument_parser&&) = default;

  ///@}
  /** @name Adding options and arguments */
  ///@{

  /** @brief Add a flag (i.e. a boolean parameter that is "true" if
   *         given and "false" if not given).
   *
   *  The value of a flag defaults to `false`. If, for some strange
   *  reason, users should be forced to type the boolean value on
   *  the command line, e.g., "my_exe -b 1", use add_option()
   *  instead. If a flag with default value `true` is desired,
   *  invert the logic and use this instead.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] cli_flags The valid command line flags to trigger
   *             this flag to `true`. At least one must be given.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *
   *  @return A read-only reference to the value pointed to by this
   *          flag.
   */
  readonly_reference<bool>
  add_flag(std::string const& name,
           std::initializer_list<std::string> cli_flags,
           std::string const& description);

  /** @brief Add a flag with environment variable override.
   *
   *  The value of a flag defaults to `false`. The flag may be set to
   *  `true` by passing the flag on the command line. Alternatively,
   *  it may be set to `true` if the environment variable `env` is
   *  defined and has a value that converts to `true`.
   *
   *  @tparam AccessPolicy The access method for the environment
   *          variable. (Deduced.)
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] cli_flags The valid command line flags to trigger
   *             this flag to `true`. At least one must be given.
   *  @param[in] env The environment variable to prefer over the
   *             default parameter value.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *
   *  @return A read-only reference to the value pointed to by this
   *          flag.
   */
  template <typename AccessPolicy>
  readonly_reference<bool>
  add_flag(std::string const& name,
           std::initializer_list<std::string> cli_flags,
           EnvVariable<AccessPolicy> env,
           std::string const& description)
  {
    if (env.exists() && env.template value<bool>())
      return add_flag_impl_(name,
                            std::move(cli_flags),
                            description + "\nENV: {" + env.name() + "}",
                            true);
    else
      return add_flag(name,
                      std::move(cli_flags),
                      description + "\nENV: {" + env.name() + "}");
  }

  /** @brief Add an additional named option.
   *
   *  Currently, named options are all optional. This could be
   *  expanded if needed.
   *
   *  @tparam T The type associated with the option. Deduced if a
   *          default value is given. If the default value is not
   *          given, the template parameter must be named explicitly
   *          and the default value will be default-constructed.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] cli_flags The valid command line flags to identify
   *             this option and its value. At least one must be
   *             given.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *  @param[in] default_value The default value to be returned if
   *             the option is not passed to the command line.
   *
   *  @return A read-only reference to the value pointed to by this
   *          option.
   */
  template <typename T>
  readonly_reference<T> add_option(std::string const& name,
                                   std::initializer_list<std::string> cli_flags,
                                   std::string const& description,
                                   T default_value = T());

  /** @brief Add an additional named option.
   *
   *  Currently, named options are all optional. This could be
   *  expanded if needed.
   *
   *  @tparam T The type associated with the option. Deduced if a
   *          default value is given. If the default value is not
   *          given, the template parameter must be named explicitly
   *          and the default value will be default-constructed.
   *  @tparam AccessPolicy The access method for the environment
   *          variable. (Deduced.)
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] cli_flags The valid command line flags to identify
   *             this option and its value. At least one must be
   *             given.
   *  @param[in] env The environment variable to prefer over the
   *             default parameter value.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *  @param[in] default_value The default value to be returned if
   *             the option is not passed to the command line.
   *
   *  @return A read-only reference to the value pointed to by this
   *          option.
   */
  template <typename T, typename AccessPolicy>
  readonly_reference<T> add_option(std::string const& name,
                                   std::initializer_list<std::string> cli_flags,
                                   EnvVariable<AccessPolicy> env,
                                   std::string const& description,
                                   T default_value = T())
  {
    if (env.exists())
      return add_option(name,
                        std::move(cli_flags),
                        description + "\nENV: {" + env.name() + "}",
                        env.template value<T>());
    else
      return add_option(name,
                        std::move(cli_flags),
                        description + "\nENV: {" + env.name() + "}",
                        std::move(default_value));
  }

  /** @brief Add an additional named option; overloaded for "char
   *         const*" parameters.
   *
   *  The value will be stored as an `std::string`. Its value must
   *  be extracted using `get<std::string>(name)`.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] cli_flags The valid command line flags to trigger
   *             this flag to `true`. At least one must be given.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *  @param[in] default_value The default value to be returned if
   *             the option is not passed to the command line.
   *
   *  @return A read-only reference to the value pointed to by this
   *          option.
   */
  readonly_reference<std::string>
  add_option(std::string const& name,
             std::initializer_list<std::string> cli_flags,
             std::string const& description,
             char const* default_value)
  {
    return add_option(name,
                      std::move(cli_flags),
                      description,
                      std::string(default_value));
  }

  /** @brief Add an additional named option; overloaded for "char
   *         const*" parameters.
   *
   *  The value will be stored as an `std::string`. Its value must
   *  be extracted using `get<std::string>(name)`.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] cli_flags The valid command line flags to trigger
   *             this flag to `true`. At least one must be given.
   *  @param[in] env The environment variable to prefer over the
   *             default parameter value.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *  @param[in] default_value The default value to be returned if
   *             the option is not passed to the command line.
   *
   *  @return A read-only reference to the value pointed to by this
   *          option.
   */
  template <typename AccessPolicy>
  readonly_reference<std::string>
  add_option(std::string const& name,
             std::initializer_list<std::string> cli_flags,
             EnvVariable<AccessPolicy> env,
             std::string const& description,
             char const* default_value)
  {
    return add_option(name,
                      cli_flags,
                      std::move(env),
                      description + "\nENV: {" + env.name() + "}",
                      std::string(default_value));
  }

  /** @brief Add an optional positional argument.
   *
   *  These are essentially defaulted positional arguments. They must
   *  be given on the command line in the order in which they are
   *  added to the parser. If the arguments have all been added by the
   *  time the help message is produced, the help message will display
   *  the correct ordering.
   *
   *  @tparam T The type to which the argument maps.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *  @param[in] default_value The value to use for this argument if
   *             not detected in the formal argument list.
   *
   *  @return A read-only reference to the value pointed to by this
   *          argument.
   */
  template <typename T>
  readonly_reference<T> add_argument(std::string const& name,
                                     std::string const& description,
                                     T default_value = T());

  /** @brief Add a positional argument; char const* overload
   *
   *  The data is stored in an std::string object internally and
   *  must be accessed using `get<std::string>(name)`.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *  @param[in] default_value The value to use for this argument if
   *             not detected in the formal argument list.
   *
   *  @return A read-only reference to the value pointed to by this
   *          argument.
   */
  readonly_reference<std::string> add_argument(std::string const& name,
                                               std::string const& description,
                                               char const* default_value)
  {
    return add_argument(name, description, std::string(default_value));
  }

  /** @brief Add a "required" positional argument.
   *
   *  @tparam T The type to which the argument maps.
   *
   *  @param[in] name The name to be used to refer to the argument.
   *  @param[in] description A brief description of the argument,
   *             used for the help message.
   *
   *  @return A read-only reference to the value pointed to by this
   *          argument.
   */
  template <typename T>
  readonly_reference<T> add_required_argument(std::string const& name,
                                              std::string const& description);

  /** @brief Clear all state in the parser.
   *
   *  The resulting state is as though the parser had been newly
   *  constructed.
   */
  void clear() noexcept;

  ///@}
  /** @name Command-line-like parsing */
  ///@{

  /** @brief Parse the command line arguments and finalize the
   *         arguments.
   *
   *  This is equivalent to calling parse_no_finalize() followed
   *  immediately by finalize().
   *
   *  @param[in] argc The number of arguments
   *  @param[in] argv The list of arguments
   *
   *  @throws parse_error if an internal parsing error is detected.
   */
  void parse(int argc, char const* const argv[]);

  /** @brief Parse the command line arguments but do not finalize
   *         the parser.
   *
   *  This parses command-line-like arguments but does no checks for
   *  required arguments. Users should call finalize() before
   *  attempting to use the values associated with any required
   *  arguments.
   *
   *  @param[in] argc The number of arguments
   *  @param[in] argv The list of arguments
   *
   *  @throws parse_error if an internal parsing error is detected.
   */
  void parse_no_finalize(int argc, char const* const argv[]);

  /** @brief Assert that all required components are set properly.
   *

   *  This should be called sometime after parse_no_finalize() and
   *  before using the values. This is implicitly called by parse().
   *
   *  @throws missing_required_arguments If a missing argument is
   *          detected.
   */
  void finalize() const;

  ///@}
  /** @name Queries */
  ///@{

  /** @brief Get the executable name.
   *
   *  This is only meaningful after calling either parse() or
   *  parse_no_finalize().
   *
   *  @return The name of the executable.
   */
  std::string get_exe_name() const noexcept;

  /** @brief Test if an option exists in the parser.
   *
   *  This only tests whether the argument or option is known to the
   *  parser, not whether it has been set or modified by the parser.
   *
   *  @param[in] option_name The name of the option/argument.
   */
  bool option_is_defined(std::string const& option_name) const;

  /** @brief Test if help has been requested. */
  bool help_requested() const;

  /** @brief Get the requested value from the argument list.
   *  @tparam T The type of the requested parameter.
   *  @param option_name The name given to the option or argument.
   *  @return A const-reference to the held value.
   */
  template <typename T>
  T const& get(std::string const& option_name) const;

  ///@}
  /** @name Output */
  ///@{

  /** @brief Print a help string to a stream.
   *  @param[in] stream The ostream to print the help message to.
   */
  void print_help(std::ostream& stream) const;

  ///@}

private:
  /** @brief Reinitialize the parser. */
  void init() noexcept;

  /** @brief Implementation of add_flag */
  readonly_reference<bool>
  add_flag_impl_(std::string const& name,
                 std::initializer_list<std::string> cli_flags,
                 std::string const& description,
                 bool default_value);

private:
  /** @brief Dictionary of arguments to their values */
  std::unordered_map<std::string, std::any> params_;
  /** @brief Patch around in-progress clara limitation */
  std::unordered_set<std::string> required_;
  /** @brief The underlying clara object */
  clara::Parser parser_;
};

template <typename ErrorHandler>
inline bool argument_parser<ErrorHandler>::option_is_defined(
  std::string const& option_name) const
{
  return params_.count(option_name);
}

template <typename ErrorHandler>
template <typename T>
inline T const&
argument_parser<ErrorHandler>::get(std::string const& option_name) const
{
  if (!option_is_defined(option_name)) {
    LBANN_ERROR("Invalid option: ", option_name);
  }
  return std::any_cast<T const&>(params_.at(option_name));
}

template <typename ErrorHandler>
template <typename T>
inline auto argument_parser<ErrorHandler>::add_option(
  std::string const& name,
  std::initializer_list<std::string> cli_flags,
  std::string const& description,
  T default_value) -> readonly_reference<T>
{
  params_[name] = std::move(default_value);
  auto& param_ref = std::any_cast<T&>(params_[name]);
  clara::Opt option(param_ref, name);
  for (auto const& f : cli_flags)
    option[f];
  parser_ |= option(description).optional();
  return param_ref;
}

template <typename ErrorHandler>
template <typename T>
inline auto
argument_parser<ErrorHandler>::add_argument(std::string const& name,
                                            std::string const& description,
                                            T default_value)
  -> readonly_reference<T>
{
  params_[name] = std::move(default_value);
  auto& param_ref = std::any_cast<T&>(params_[name]);
  parser_ |= clara::Arg(param_ref, name)(description).optional();
  return param_ref;
}

template <typename ErrorHandler>
template <typename T>
inline auto argument_parser<ErrorHandler>::add_required_argument(
  std::string const& name,
  std::string const& description) -> readonly_reference<T>
{
  // Add the reference to bind to
  params_[name] = T{};
  auto& param_any = params_[name];
  auto& param_ref = std::any_cast<T&>(param_any);

  required_.insert(name);

  // Make sure the required arguments are all grouped together.
  auto iter = parser_.m_args.cbegin(), invalid = parser_.m_args.cend();
  while (iter != invalid && !iter->isOptional())
    ++iter;

  // Create the argument
  auto ret = parser_.m_args.emplace(
    iter,
    [name, &param_ref, this](std::string const& value) {
      auto result = clara::detail::convertInto(value, param_ref);
      if (result)
        required_.erase(name);
      return result;
    },
    name);
  ret->operator()(description).required();
  return param_ref;
}

template <typename ErrorHandler>
argument_parser<ErrorHandler>::argument_parser()
{
  init();
}

template <typename ErrorHandler>
void argument_parser<ErrorHandler>::clear() noexcept
{
  std::unordered_map<std::string, std::any>{}.swap(params_);
  std::unordered_set<std::string>{}.swap(required_);
  parser_ = clara::Parser{};
  init();
}

template <typename ErrorHandler>
void argument_parser<ErrorHandler>::init() noexcept
{
  params_["print help"] = false;
  parser_ |= clara::ExeName();
  parser_ |= clara::Help(std::any_cast<bool&>(params_["print help"]));
}

template <typename ErrorHandler>
void argument_parser<ErrorHandler>::parse(int argc, char const* const argv[])
{
  parse_no_finalize(argc, argv);
  finalize();
}

template <typename ErrorHandler>
void argument_parser<ErrorHandler>::parse_no_finalize(int argc,
                                                      char const* const argv[])
{
  std::vector<char const*> newargv(argv, argv + argc);
  auto parse_result =
    parser_.parse(clara::Args(newargv.size(), newargv.data()));

  if (!parse_result)
    this->handle_error(parse_result, parser_, newargv);
}

template <typename ErrorHandler>
void argument_parser<ErrorHandler>::finalize() const
{
  if (!help_requested() && required_.size())
    throw missing_required_arguments(required_);
}

template <typename ErrorHandler>
auto argument_parser<ErrorHandler>::add_flag(
  std::string const& name,
  std::initializer_list<std::string> cli_flags,
  std::string const& description) -> readonly_reference<bool>
{
  return add_flag_impl_(name, std::move(cli_flags), description, false);
}

template <typename ErrorHandler>
std::string argument_parser<ErrorHandler>::get_exe_name() const noexcept
{
  return parser_.m_exeName.name();
}

template <typename ErrorHandler>
bool argument_parser<ErrorHandler>::help_requested() const
{
  return std::any_cast<bool>(params_.at("print help"));
}

template <typename ErrorHandler>
void argument_parser<ErrorHandler>::print_help(std::ostream& out) const
{
  out << parser_ << std::endl;
}

template <typename ErrorHandler>
auto argument_parser<ErrorHandler>::add_flag_impl_(
  std::string const& name,
  std::initializer_list<std::string> cli_flags,
  std::string const& description,
  bool default_value) -> readonly_reference<bool>
{
  params_[name] = default_value;
  auto& param_ref = std::any_cast<bool&>(params_[name]);
  clara::Opt option(param_ref);
  for (auto const& f : cli_flags)
    option[f];
  parser_ |= option(description).optional();
  return param_ref;
}

} // namespace utils

using default_arg_parser_type = utils::argument_parser<utils::strict_parsing>;

default_arg_parser_type& global_argument_parser();

} // namespace lbann

/** @brief Write the parser's help string to the given @c ostream */
template <typename ErrorHandler>
std::ostream&
operator<<(std::ostream& os,
           lbann::utils::argument_parser<ErrorHandler> const& parser)
{
  parser.print_help(os);
  return os;
}

#endif /* LBANN_UTILS_ARGUMENT_PARSER_HPP_INCLUDED */
