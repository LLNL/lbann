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

#include "mpi_states.hpp"
#include <iostream>


namespace tools_compute_mean {

mpi_states::mpi_states()
  : m_comm(MPI_COMM_WORLD), m_effective_num_ranks(1), m_num_ranks(1), m_my_rank(0), m_root(0) {}

/// Check the return code of an MPI call, and abort if not successful.
bool mpi_states::check_mpi(const int mpi_code) const {
  const bool ok = (mpi_code == MPI_SUCCESS);
  if (!ok) {
    MPI_Abort(m_comm, mpi_code);
  }
  return ok;
};


/// Abort MPI. Only a designated rank prints out a message.
void mpi_states::abort(const std::string err, const int report_rank) const {
  if (m_my_rank == report_rank) {
    std::cerr << "rank " << m_my_rank << ": " << err << std::endl;
  }
  MPI_Abort(m_comm, 1);
}


/**
 * Initialize MPI, and obtain the total number of ranks and my rank.
 */
void mpi_states::initialize(int& argc, char **& argv) {
  int mpi_code = MPI_SUCCESS;
  mpi_code = MPI_Init(&argc, &argv);
  check_mpi(mpi_code);
  mpi_code = MPI_Comm_size(m_comm, &m_num_ranks);
  m_effective_num_ranks = m_num_ranks;
  check_mpi(mpi_code);
  mpi_code = MPI_Comm_rank(m_comm, &m_my_rank);
  check_mpi(mpi_code);
  if (m_my_rank == m_root) {
    std::cout << "Number of ranks: " << m_num_ranks << std::endl;
  }
}


/// Return a string to print out my rank and the total number of ranks.
std::string mpi_states::description() const {
  return std::to_string(m_my_rank) + '/' + std::to_string(m_num_ranks);
}


/**
 * Split the given amount into chunks as evenly as possible among ranks.
 * The i_th element of num_per_rank contains the amount for i_th rank.
 */
void mpi_states::split_over_ranks(const unsigned int num_total,
                                  std::vector<unsigned int>& num_per_rank) const {
  num_per_rank.clear();
  if (m_num_ranks <= 0) {
    return;
  }
  const unsigned int num_ranks = static_cast<unsigned int>(m_num_ranks);
  const unsigned int chunkSz = num_total/num_ranks;
  const unsigned int rest = num_total - num_ranks*chunkSz;
  const unsigned int first_with_extra = num_ranks - rest;
  num_per_rank.assign(num_ranks+1, chunkSz);
  num_per_rank[num_ranks] = 0u;

  for (unsigned int i = num_ranks; i > first_with_extra ; --i) {
    num_per_rank[i-1] ++;
  }

  unsigned int prev = 0u;
  unsigned int sumSoFar = 0u;
  for (unsigned int i=0; i < num_ranks; ++i) {
    const unsigned int cur = num_per_rank[i];
    num_per_rank[i] = sumSoFar = sumSoFar + prev;
    prev = cur;
  }
  num_per_rank[num_ranks] = num_total;
}

} // end of namespace tools_compute_mean
