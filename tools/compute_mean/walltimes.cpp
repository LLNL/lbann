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

#include <mpi.h>
#include <cmath>
#include <iostream>
#include "walltimes.hpp"


namespace tools_compute_mean {

/**
 *  Collects the average, the min, the max and the stdev of various timing
 *  measured over the processes. Each processors supposed to have the same
 *  number of timing values in the localTimes input vector, each of which
 *  represent a different timing type.
 */
void collect_times(const std::vector<double>& localTimes,
                   std::vector<double>& avgTimes,
                   std::vector<double>& minTimes,
                   std::vector<double>& maxTimes,
                   std::vector<double>& stdTimes,
                   const mpi_states& ms) {
  const unsigned int numTimes = localTimes.size();
  if (numTimes == 0u) {
    return;
  }

  avgTimes.clear();
  avgTimes.resize(numTimes);
  minTimes.clear();
  minTimes.resize(numTimes);
  maxTimes.clear();
  maxTimes.resize(numTimes);
  stdTimes.clear();
  stdTimes.resize(numTimes);
  std::vector<double> diffTimes(numTimes);

  int mc = MPI_SUCCESS;
  mc = MPI_Allreduce(const_cast<void*>(_ConstVoidP(&localTimes[0])), _VoidP(&avgTimes[0]),
                     numTimes, MPI_DOUBLE, MPI_SUM, ms.get_comm());
  ms.check_mpi(mc);

  for (unsigned int i=0u; i < numTimes; ++i) {
    avgTimes[i] = avgTimes[i]/ms.get_effective_num_ranks();
    double diff = (avgTimes[i] - localTimes[i]);
    diffTimes[i] = diff*diff;
  }

  mc = MPI_Reduce(const_cast<void*>(_ConstVoidP(&localTimes[0])), _VoidP(&minTimes[0]),
                  numTimes, MPI_DOUBLE, MPI_MIN, ms.m_root, ms.get_comm());
  ms.check_mpi(mc);

  mc = MPI_Reduce(const_cast<void*>(_ConstVoidP(&localTimes[0])), _VoidP(&maxTimes[0]),
                  numTimes, MPI_DOUBLE, MPI_MAX, ms.m_root, ms.get_comm());
  ms.check_mpi(mc);

  mc = MPI_Reduce(const_cast<void*>(_ConstVoidP(&diffTimes[0])), _VoidP(&stdTimes[0]),
                  numTimes, MPI_DOUBLE, MPI_SUM, ms.m_root, ms.get_comm());
  ms.check_mpi(mc);

  for (unsigned int i=0u; i < numTimes; ++i) {
    stdTimes[i] = sqrt(stdTimes[i]/ms.get_effective_num_ranks());
  }
}


/// print out the wallclock times
void summarize_walltimes(walltimes& wt, mpi_states& ms) {
  if (ms.is_serial_run()) {
    const std::vector<std::string> names = wt.get_names();
    const std::vector<double> times = wt.get();
    std::cout << "name\ttime" << std::endl;
    for (size_t i=0u; i < times.size(); ++i) {
      std::cout << names[i] << '\t' << times[i] << std::endl;
    }
  } else {
    std::vector<double> avgTimes, minTimes, maxTimes, stdTimes;
    collect_times(wt.get(), avgTimes, minTimes, maxTimes, stdTimes, ms);
    const std::vector<std::string> names = wt.get_names();
    if (ms.is_root()) {
      const size_t num_times = avgTimes.size();
      std::cout << "name\tavg\tmin\tmax\tstddev" << std::endl;
      for (size_t i=0u; i < num_times; ++i) {
        std::cout << names[i] << '\t' << avgTimes[i] << '\t' << minTimes[i] << '\t' << maxTimes[i] << '\t' << stdTimes[i] << std::endl;
      }
    }
  }
}

} // end of namespace tools_compute_mean
