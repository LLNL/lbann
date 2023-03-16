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

#include "Catch2BasicSupport.hpp"

#include "lbann/utils/running_statistics.hpp"

#include <iostream>
#include <thread>

TEST_CASE("Running statistics accumulation", "[utils][stats]")
{
  using StatsType = lbann::RunningStats;
  StatsType stats;
  CHECK(stats.samples() == 0UL);
  CHECK(stats.min() == StatsType::default_min);
  CHECK(stats.max() == StatsType::default_max);
  CHECK(stats.mean() == 0.);
  CHECK(stats.total() == 0.);
  CHECK(stats.stddev() == 0.);

  stats.insert(2.);
  CHECK(stats.samples() == 1UL);
  CHECK(stats.min() == 2.);
  CHECK(stats.max() == 2.);
  CHECK(stats.mean() == 2.);
  CHECK(stats.total() == 2.);
  CHECK(stats.variance() == 0.);
  CHECK(stats.stddev() == 0.);

  stats.insert(1.);
  CHECK(stats.samples() == 2UL);
  CHECK(stats.min() == 1.);
  CHECK(stats.max() == 2.);
  CHECK(stats.mean() == 1.5);
  CHECK(stats.total() == 3.);
  CHECK(stats.variance() == Approx(0.5));
  CHECK(stats.stddev() == Approx(std::sqrt(0.5)));

  stats.insert(3.);
  CHECK(stats.samples() == 3UL);
  CHECK(stats.min() == 1.);
  CHECK(stats.max() == 3.);
  CHECK(stats.mean() == 2.);
  CHECK(stats.total() == 6.);
  CHECK(stats.variance() == 1.);
  CHECK(stats.stddev() == 1.);

  stats.reset();

  CHECK(stats.samples() == 0UL);
  CHECK(stats.min() == StatsType::default_min);
  CHECK(stats.max() == StatsType::default_max);
  CHECK(stats.mean() == 0.);
  CHECK(stats.total() == 0.);
  CHECK(stats.variance() == 0.);
  CHECK(stats.stddev() == 0.);

  stats.insert(5.);
  stats.insert(4.);
  stats.insert(5.);
  stats.insert(6.);
  stats.insert(4.5);
  stats.insert(5.5);

  CHECK(stats.samples() == 6UL);
  CHECK(stats.min() == 4.);
  CHECK(stats.max() == 6.);
  CHECK(stats.mean() == 5.);
  CHECK(stats.total() == 30.);
  CHECK(stats.variance() == Approx(0.5));
  CHECK(stats.stddev() == Approx(std::sqrt(0.5)));
}
