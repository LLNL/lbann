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

#ifndef _TOOLS_COMPUTE_MEAN_MPI_STATES_HPP_
#define _TOOLS_COMPUTE_MEAN_MPI_STATES_HPP_

#include <string>
#include <mpi.h>
#include <vector>

#if defined(__bgq__) || (defined(MPICH2_NUMVERSION) && (MPICH2_NUMVERSION <= 10700300))
#define _ConstVoidP(_P_) ((void*) (_P_))
#define _VoidP(_P_)      ((void*) (_P_))
#else
#define _ConstVoidP(_P_) reinterpret_cast<const void*>(_P_)
#define _VoidP(_P_)      reinterpret_cast<void*>(_P_)
#endif

namespace tools_compute_mean {

class mpi_states {
 public:
  MPI_Comm m_comm;
  int m_effective_num_ranks;
  int m_num_ranks;
  int m_my_rank;
  int m_root;

  mpi_states();
  bool check_mpi(const int mpi_code) const;
  void abort(const std::string err, const int report_rank) const;
  void abort(const std::string err) const {
    abort(err, m_root);
  }
  void abort_by_me(const std::string err) const {
    abort(err, m_my_rank);
  }
  void initialize(int& argc, char **& argv);
  void finalize() {
    MPI_Finalize();
  }
  std::string description() const;
  bool is_root() const {
    return (m_my_rank == m_root);
  }
  bool is_serial_run() const {
    return (m_num_ranks == 1);
  }
  void split_over_ranks(const unsigned int num_total,
                        std::vector<unsigned int>& num_per_rank) const;
  int get_num_ranks() const {
    return m_num_ranks;
  }
  void set_effective_num_ranks(unsigned int n) {
    m_effective_num_ranks = static_cast<int>(n);
  }
  int get_effective_num_ranks() const {
    return m_effective_num_ranks;
  }
  int get_my_rank() const {
    return m_my_rank;
  }
  const MPI_Comm& get_comm() const {
    return m_comm;
  }
};

} // end of namespace tools_compute_mean
#endif /// _TOOLS_COMPUTE_MEAN_MPI_STATES_HPP_
