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
#include <catch2/catch.hpp>

#include "lbann/utils/argument_parser.hpp"

SCENARIO ("Testing the argument parser", "[parser][utilities]")
{
  GIVEN ("An argument parser")
  {
    lbann::utils::argument_parser parser;
    WHEN ("The default arguments are passed")
    {
      int const argc = 1;
      char const* argv[] = { "argument_parser_test.exe" };
      THEN ("The parser recognizes the executable name")
      {
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        REQUIRE(
          parser.get_exe_name() == "argument_parser_test.exe");
      }
    }
    WHEN ("The short help flag is passed")
    {
      int const argc = 2;
      char const* argv[] = {"argument_parser_test.exe", "-h"};
      THEN ("The parser notes that help has been requested.")
      {
        REQUIRE_FALSE(parser.help_requested());
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        REQUIRE(parser.help_requested());
      }
    }
    WHEN ("The long help flag is passed")
    {
      int const argc = 2;
      char const* argv[argc] = {"argument_parser_test.exe", "--help"};
      THEN ("The parser notes that help has been requested.")
      {
        REQUIRE_NOTHROW(parser.parse(argc, argv));
        REQUIRE(parser.help_requested());
      }
    }
    WHEN ("A boolean flag is added")
    {
      parser.add_flag(
        "verbose", {"-v", "--verbose"}, "print verbosely");
      THEN ("The flag's option name is known")
      {
        REQUIRE(parser.option_is_defined("verbose"));
      }
      AND_WHEN("The flag is passed")
      {
        int const argc = 2;
        char const* argv[]
          = {"argument_parser_test.exe", "--verbose"};
        REQUIRE_FALSE(parser.get<bool>("verbose"));
        THEN ("The verbose flag is registered")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(parser.get<bool>("verbose"));
        }
      }
    }
    WHEN ("An option is added")
    {
      parser.add_option("number of threads", {"-t", "--num_threads"},
                        "The number of threads to use in this test.", 1);
      THEN ("The option is registered with the parser.")
      {
        REQUIRE(parser.option_is_defined("number of threads"));
        REQUIRE(parser.template get<int>("number of threads") == 1);
      }
      AND_WHEN ("The short option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[] = {"argument_parser_test.exe", "-t", "9"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<int>("number of threads") == 9);
        }
      }
      AND_WHEN ("The long option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "--num_threads", "13"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<int>("number of threads") == 13);
        }
      }
    }
    WHEN ("A string-valued option is added")
    {
      parser.add_option("my name", {"-n", "--name", "--my_name"},
                        "The number of threads to use in this test.",
                        "<unregistered name>");
      THEN ("The option is registered with the parser.")
      {
        REQUIRE(parser.option_is_defined("my name"));
        REQUIRE(parser.template get<std::string>("my name")
                == "<unregistered name>");
      }
      AND_WHEN ("The short option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "-n", "Banana Joe"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<std::string>("my name")
            == "Banana Joe");
        }
      }
      AND_WHEN ("The first long option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "--name", "Plantain Pete"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<std::string>("my name")
            == "Plantain Pete");
        }
      }
      AND_WHEN ("The second long option is passed on the command line")
      {
        int const argc = 3;
        char const* argv[]
          = {"argument_parser_test.exe", "--my_name",
             "Jackfruit Jill"};
        THEN ("The new value is registered.")
        {
          REQUIRE_NOTHROW(parser.parse(argc, argv));
          REQUIRE(
            parser.template get<std::string>("my name")
            == "Jackfruit Jill");
        }
      }
    }
    WHEN ("A required argument is added")
    {
      parser.add_required_argument<int>(
        "required", "This argument is required.");
      THEN ("The option is recognized")
      {
        REQUIRE(parser.option_is_defined("required"));
      }
      AND_WHEN("The option is not passed in the arguments")
      {
        int const argc = 1;
        char const* argv[argc] = {"argument_parser_test.exe"};

        THEN ("Finalization fails.")
        {
          parser.parse_no_finalize(argc,argv);
          REQUIRE_THROWS_AS(
            parser.finalize(),
            lbann::utils::argument_parser::missing_required_arguments);
        }
      }
    }
  }
}
