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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _TOOLS_COMPUTE_MEAN_WALLTIMES_HPP_
#define _TOOLS_COMPUTE_MEAN_WALLTIMES_HPP_
#include <chrono>
#include <vector>
#include "mpi_states.hpp"


namespace tools_compute_mean {

/** Return time in fractional seconds since an epoch. */
inline double get_time() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(
           steady_clock::now().time_since_epoch()).count();
}

class walltimes {
 public:
  double m_total;
  double m_load;
  double m_decode;
  double m_preprocess;
  double m_write;

  walltimes() :
    m_total(0.0),
    m_load(0.0),
    m_decode(0.0),
    m_preprocess(0.0),
    m_write(0.0) {}

  std::vector<double> get() const {
    return {m_total, m_load, m_decode, m_preprocess, m_write};
  }

  std::vector<std::string> get_names() const {
    return {"total", "load", "decode", "preprocess", "write"};
  }
};

void collect_times(const std::vector<double>& localTimes,
                   std::vector<double>& avgTimes, std::vector<double>& minTimes,
                   std::vector<double>& maxTimes, std::vector<double>& stdTimes,
                   const mpi_states& ms);

void summarize_walltimes(walltimes& wt, mpi_states& ms);

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_WALLTIMES_HPP_
