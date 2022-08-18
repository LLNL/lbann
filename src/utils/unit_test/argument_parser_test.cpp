///////////////////////////////////////////////////////////////////////////////
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
#include "Catch2BasicSupport.hpp"

#include "lbann/utils/argument_parser.hpp"

#include "lbann/utils/environment_variable.hpp"

#include "stubs/preset_env_accessor.hpp"

TEMPLATE_TEST_CASE ("Testing the argument parser", "[parser][utilities]",
                    lbann::utils::strict_parsing,
                    lbann::utils::allow_extra_parameters)
{
  using error_handler = TestType;
  using parser_type = lbann::utils::argument_parser<error_handler>;

  parser_type parser;
  SECTION("Passing default arguments")
  {
      char const* argv[] = { "argument_parser_test.exe" };
      int const argc = sizeof(argv) / sizeof(argv[0]);
      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(
        parser.get_exe_name() == "argument_parser_test.exe");
  }

  SECTION("Short help flag")
  {
    char const* argv[] = {"argument_parser_test.exe", "-h"};
    int const argc = sizeof(argv) / sizeof(argv[0]);

    CHECK_FALSE(parser.help_requested());
    REQUIRE_NOTHROW(parser.parse(argc, argv));
    CHECK(parser.help_requested());

    REQUIRE_NOTHROW(parser.clear());
    CHECK_FALSE(parser.help_requested());
  }

  SECTION("Long help flag")
  {
    char const* argv[] = {"argument_parser_test.exe", "--help"};
    int const argc = sizeof(argv) / sizeof(argv[0]);

    CHECK_FALSE(parser.help_requested());
    REQUIRE_NOTHROW(parser.parse(argc, argv));
    CHECK(parser.help_requested());

    REQUIRE_NOTHROW(parser.clear());
    CHECK_FALSE(parser.help_requested());
  }

  SECTION("Boolean flags are false by default")
  {
    auto flag_v =
      parser.add_flag(
        "flag v", {"-v", "--flag-v"}, "print verbosely");

    CHECK(parser.option_is_defined("flag v"));
    CHECK_FALSE(flag_v);

    SECTION("Clear removes the flag")
    {
      REQUIRE_NOTHROW(parser.clear());
      CHECK_FALSE(parser.option_is_defined("flag v"));
    }

    SECTION("Short flag sets flag to true")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "-v"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      CHECK_FALSE(parser.template get<bool>("flag v"));

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<bool>("flag v"));
      CHECK(flag_v);
    }

    SECTION("Long flag sets flag to true")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "--flag-v"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      CHECK_FALSE(parser.template get<bool>("flag v"));

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<bool>("flag v"));
      CHECK(flag_v);
    }
  }

  SECTION("Numeric option")
  {
    auto param_t =
      parser.add_option("parameter t", {"-t", "--param-t"},
                        "Docstring for \"parameter t\"", 1);

    CHECK(parser.option_is_defined("parameter t"));
    CHECK(parser.template get<int>("parameter t") == 1);
    CHECK(param_t == 1);

    SECTION ("Short flag")
    {
      char const* argv[] = {"argument_parser_test.exe", "-t", "9"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(
        parser.template get<int>("parameter t") == 9);
      CHECK(param_t == 9);
    }

    SECTION ("Long flag")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "--param-t", "13"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<int>("parameter t") == 13);
      CHECK(param_t == 13);
    }
  }

  SECTION("String-valued option")
  {
    auto param_n =
      parser.add_option("parameter n", {"-n", "--param-n", "--parameter-n"},
                        "Docstring for \"parameter t\"",
                        "<unregistered parameter value>");

    CHECK(parser.option_is_defined("parameter n"));
    CHECK(parser.template get<std::string>("parameter n")
          == "<unregistered parameter value>");

    SECTION("The short option is passed on the command line")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "-n", "short form of param n"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(
        parser.template get<std::string>("parameter n")
        == "short form of param n");
      CHECK(param_n == "short form of param n");
    }

    SECTION ("First long option")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "--param-n",
           "first long form of param n"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(
        parser.template get<std::string>("parameter n")
        == "first long form of param n");
      CHECK(param_n == "first long form of param n");
    }

    SECTION ("Second long option")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "--parameter-n",
           "second long form of param n"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(
        parser.template get<std::string>("parameter n")
        == "second long form of param n");
      CHECK(param_n == "second long form of param n");
    }
  }

  SECTION ("Required numeric argument")
  {
    auto required_int =
      parser.template add_required_argument<int>(
        "required", "This argument is required.");

    CHECK(parser.option_is_defined("required"));

    SECTION("Missing required argument")
    {
      char const* argv[] = {"argument_parser_test.exe"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      parser.parse_no_finalize(argc,argv);
      REQUIRE_THROWS_AS(
        parser.finalize(),
        typename parser_type::missing_required_arguments);
    }

    SECTION("Required argument is passed")
    {
      char const* argv[] = {"argument_parser_test.exe","13"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(required_int == 13);
    }

    SECTION("Another is added option and passed in the arguments")
    {
      auto required_string =
        parser.template add_required_argument<std::string>(
          "required string", "This argument is also required.");

      char const* argv[] = {"argument_parser_test.exe","13","bananas"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(required_int == 13);
      CHECK(required_string == "bananas");
    }
  }

  SECTION("More complex argument relationships")
  {
    auto optional_int =
      parser.add_argument(
        "optional", "This argument is optional.", -1);

    CHECK(parser.option_is_defined("optional"));
    CHECK(parser.template get<int>("optional") == -1);
    CHECK(optional_int == -1);

    SECTION("Only default arguments passed")
    {
      int const argc = 1;
      char const* argv[] = {"argument_parser_test.exe"};

      REQUIRE_NOTHROW(parser.parse(argc,argv));
      CHECK(parser.template get<int>("optional") == -1);
      CHECK(optional_int == -1);
    }

    SECTION("Option is passed in the arguments")
    {
      int const argc = 2;
      char const* argv[] = {"argument_parser_test.exe","13"};

      REQUIRE_NOTHROW(parser.parse(argc,argv));
      CHECK(parser.template get<int>("optional") == 13);
      CHECK(optional_int == 13);
    }

    SECTION("Another optional argument is added")
    {
      auto optional_string =
        parser.add_argument(
          "optional string", "This argument is also optional.",
          "pickles");

      SECTION("Parsing both arguments works")
      {
        int const argc = 3;
        char const* argv[] = {"argument_parser_test.exe","42","bananas"};

        CHECK(optional_int == -1);
        CHECK(optional_string == "pickles");
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        CHECK(optional_int == 42);
        CHECK(optional_string == "bananas");
      }

      SECTION("A required argument is added and passed in the arguments")
      {
        auto required_string =
          parser.template add_required_argument<std::string>(
            "required string", "This argument is required.");

        SECTION("Bad ordering of the arguments")
        {
          int const argc = 3;
          char const* argv[] = {
            "argument_parser_test.exe","42","bananas"};

          REQUIRE_THROWS(parser.parse(argc,argv));
          CHECK(required_string == "42");
        }

        SECTION("Correct ordering of the arguments")
        {
          int const argc = 3;
          char const* argv[] = {
            "argument_parser_test.exe","bananas","42"};

          CHECK(optional_int == -1);
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          CHECK(optional_int == 42);
          CHECK(required_string == "bananas");
        }
      }
    }
  }

  SECTION("A flag with env variable override is added")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    auto verbose =
      parser.add_flag("verbose", {"-v"},
                      TestENV("VALUE_IS_TRUE"), "");

    CHECK(parser.option_is_defined("verbose"));
    CHECK(verbose);

    SECTION("Command line flag overrides environment variable")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "-v"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<bool>("verbose"));
      CHECK(verbose);
    }
  }

  SECTION("A flag with false-valued env variable override is added")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    auto verbose =
      parser.add_flag("verbose", {"-v"},
                      TestENV("VALUE_IS_FALSE"), "");

    CHECK(parser.option_is_defined("verbose"));
    CHECK_FALSE(verbose);

    SECTION("Command line flag overrides environment variable")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "-v"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<bool>("verbose"));
      CHECK(verbose);
    }
  }

  SECTION("A flag with false-valued env variable override is added")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    auto verbose =
      parser.add_flag("verbose", {"-v"},
                      TestENV("VALUE_IS_UNDEFINED"), "");

    CHECK(parser.option_is_defined("verbose"));
    CHECK_FALSE(verbose);

    SECTION("Command line flag overrides environment variable")
    {
      char const* argv[]
        = {"argument_parser_test.exe", "-v"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<bool>("verbose"));
      CHECK(verbose);
    }
  }

  SECTION("A defined environment varible is added")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    parser.add_option(
      "apple", {"-a"}, TestENV("APPLE"),
      "Apple pie tastes good.", 1.23);

    CHECK(parser.option_is_defined("apple"));

    SECTION("The option is not passed in the arguments")
    {
      int const argc = 1;
      char const* argv[] = {"argument_parser_test.exe"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<double>("apple") == 3.14);
    }

    SECTION("The option is passed in the arguments")
    {
      int const argc = 3;
      char const* argv[] = {"argument_parser_test.exe", "-a", "5.0"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<double>("apple") == 5.0);
    }
  }

  SECTION("An undefined environment varible is added")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    parser.add_option(
      "platypus", {"-p"}, TestENV("DOESNT_EXIST"),
      "This variable won't exist.", 1.23);

    CHECK(parser.option_is_defined("platypus"));

    SECTION("The option is not passed in the arguments")
    {
      int const argc = 1;
      char const* argv[] = {"argument_parser_test.exe"};

          REQUIRE_NOTHROW(parser.parse(argc, argv));
          CHECK(parser.template get<double>("platypus") == 1.23);
    }
    SECTION("The option is passed in the arguments")
    {
      int const argc = 3;
      char const* argv[] = {"argument_parser_test.exe", "-p", "2.0"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<double>("platypus") == 2.0);
    }
  }

  SECTION("A defined string environment varible is added")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    parser.add_option(
      "pizza", {"-p"}, TestENV("PIZZA"),
      "Mmmm pizza.", "mushroom");

    CHECK(parser.option_is_defined("pizza"));

    SECTION("The option is not passed in the arguments")
    {
      int const argc = 1;
      char const* argv[] = {"argument_parser_test.exe"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<std::string>("pizza") == "pepperoni");
    }

    SECTION("The option is passed in the arguments")
    {
      int const argc = 3;
      char const* argv[] = {"argument_parser_test.exe", "-p", "hawaiian"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<std::string>("pizza") == "hawaiian");
    }
  }

  SECTION("An undefined environment varible is added to a string option")
  {
    using namespace lbann::utils::stubs;
    using TestENV = lbann::utils::EnvVariable<PresetEnvAccessor>;

    parser.add_option(
      "parameter p", {"-p"}, TestENV("DOESNT_EXIST"),
      "This variable won't exist.", "parameter p test string");

    CHECK(parser.option_is_defined("parameter p"));

    SECTION("The option is not passed in the arguments")
    {
      int const argc = 1;
      char const* argv[] = {"argument_parser_test.exe"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<std::string>("parameter p")
            == "parameter p test string");
    }

    SECTION("The option is passed in the arguments")
    {
      int const argc = 3;
      char const* argv[] = {"argument_parser_test.exe", "-p",
                            "parameter p argument value"};

      REQUIRE_NOTHROW(parser.parse(argc, argv));
      CHECK(parser.template get<std::string>("parameter p")
            == "parameter p argument value");
    }
  }
}

// Testing for cases of ignored arguments
TEST_CASE("Partial argument parsing", "[parser][utilities]")
{
  lbann::utils::argument_parser<lbann::utils::allow_extra_parameters> parser;
  auto flag_v =
    parser.add_flag(
      "flag v", {"-v", "--flag-v"}, "Docstring for \"flag v\"");
  auto param_s =
    parser.add_option("parameter s", {"-s", "--param-s"},
                      "Docstring for \"parameter s\"",
                      "default value of s");
  auto param_t =
    parser.add_option("parameter t", {"-t", "--param-t"},
                      "Docstring for \"parameter t\"", 1);

  CHECK(parser.option_is_defined("parameter t"));
  CHECK(parser.option_is_defined("flag v"));

  SECTION("Incomplete argument sets are fine.")
  {
    char const* argv[] = {"argument_parser_test.exe", "-t", "9"};
    int const argc = sizeof(argv) / sizeof(argv[0]);
    REQUIRE_NOTHROW(parser.parse(argc, argv));

    CHECK_FALSE(parser.template get<bool>("flag v"));
    CHECK_FALSE(flag_v);

    CHECK(parser.template get<int>("parameter t") == 9);
    CHECK(param_t == 9);
  }

  SECTION("Unknown arguments are ok.")
  {
    char const* argv[] = {"argument_parser_test.exe",
                          "-o", "-v", "-a", "-t", "2", "-p", "13"};
    int const argc = sizeof(argv) / sizeof(argv[0]);

    REQUIRE_NOTHROW(parser.parse(argc, argv));

    CHECK(parser.template get<bool>("flag v"));
    CHECK(flag_v);

    CHECK(parser.template get<int>("parameter t") == 2);
    CHECK(param_t == 2);
  }

  SECTION("Final argument is unknown flag is ok.")
  {
    char const* argv[] = {"argument_parser_test.exe",
                          "-o", "-v", "-a", "-t", "2", "-p", "13", "-flag"};
    int const argc = sizeof(argv) / sizeof(argv[0]);

    REQUIRE_NOTHROW(parser.parse(argc, argv));

    CHECK(parser.template get<bool>("flag v"));
    CHECK(flag_v);

    CHECK(parser.template get<int>("parameter t") == 2);
    CHECK(param_t == 2);
  }

  SECTION("Arguments with equals assignments are ok")
  {
    SECTION("Short form")
    {
      char const* argv[] = {"argument_parser_test.exe", "-t=32"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));

      CHECK(parser.template get<int>("parameter t") == 32);
      CHECK(param_t == 32);
    }

    SECTION("Long form")
    {
      char const* argv[] = {"argument_parser_test.exe", "--param-t=121"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));

      CHECK(parser.template get<int>("parameter t") == 121);
      CHECK(param_t == 121);
    }

    SECTION("String parameter with equals sign in it")
    {
      char const* argv[] = {"argument_parser_test.exe",
                            "--param-s=something=other"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));

      CHECK(parser.template get<std::string>("parameter s") == "something=other");
      CHECK(param_s == "something=other");
    }

    SECTION("Unknown parameters may also use equals signs")
    {
      char const* argv[] = {"argument_parser_test.exe", "--flag-v",
                            "--param-q=121", "--param-t=21"};
      int const argc = sizeof(argv) / sizeof(argv[0]);

      REQUIRE_NOTHROW(parser.parse(argc, argv));

      CHECK(parser.template get<int>("parameter t") == 21);
      CHECK(flag_v);
    }
  }
}
