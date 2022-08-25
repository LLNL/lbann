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

#include "Catch2BasicSupport.hpp"

#include "lbann/utils/accumulating_timer.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/timer_map.hpp"

#include <iostream>
#include <thread>

TEST_CASE("Sanity-checking the timer", "[utils][timer]")
{
  using namespace std::chrono_literals;
  auto& parser = lbann::global_argument_parser();
  parser = lbann::default_arg_parser_type{};

  lbann::Timer timer;

  // The timer should report zero in this case.
  CHECK(timer.stop() == 0.0);

  timer.start();
  std::this_thread::sleep_for(50ms);
  double sleep_time = timer.check();

  // This is a VERY forgiving tolerance
  {
    INFO("Sleep time = " << sleep_time);
    CHECK((sleep_time >= 0.05 && sleep_time < 0.1));
  }

  // Resets the counter
  timer.start();
  double startstop_time = timer.stop();
  {
    INFO("Start/stop time = " << startstop_time);
    CHECK(startstop_time > 0.);
    CHECK(startstop_time < 0.05);
    CHECK_FALSE(timer.running());
    CHECK(timer.check() == 0.);
  }

  {
    INFO("Timer reset");
    timer.start();

    CHECK(timer.running());
    CHECK(timer.check() > 0.);

    timer.reset();

    CHECK_FALSE(timer.running());
    CHECK(timer.check() == 0.);
    CHECK(timer.stop() == 0.);
  }
}

TEST_CASE("Accumulating timer", "[utils][timer]")
{
  lbann::AccumulatingTimer timer;

  SECTION("Default state is correct.")
  {
    CHECK_FALSE(timer.running());
    CHECK(timer.check() == 0.);
    CHECK(timer.min() == 0.);
    CHECK(timer.max() == 0.);
    CHECK(timer.mean() == 0.);
    CHECK(timer.stddev() == 0.);
    CHECK(timer.total_time() == 0.);
    CHECK(timer.samples() == 0);
  }

  SECTION("Running timer.")
  {
    timer.start();
    CHECK(timer.running());
    CHECK(timer.check() > 0.);

    auto time = timer.stop();
    CHECK(time > 0.);
    CHECK_FALSE(timer.running());

    CHECK(timer.samples() == 1);
    CHECK(timer.min() == time);
    CHECK(timer.max() == time);
    CHECK(timer.mean() == time);
    CHECK(timer.total_time() == time);
    CHECK(timer.stddev() == 0.);
  }

  SECTION("Reset does not commit the duration.")
  {
    timer.start();
    timer.reset();
    CHECK_FALSE(timer.running());
    CHECK(timer.samples() == 0UL);
  }

  SECTION("Multiple starts does not commit multiple durations.")
  {
    timer.start();
    timer.start();
    timer.start();
    CHECK(timer.running());
    CHECK(timer.samples() == 0UL);

    timer.stop();
    CHECK(timer.samples() == 1UL);
  }

  SECTION("Stop without a start does not commit a duration.")
  {
    timer.stop();
    timer.stop();
    CHECK(timer.samples() == 0UL);
  }
}
