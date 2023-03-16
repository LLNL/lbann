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

#include "lbann/comm_impl.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/jag_utils.hpp"

namespace lbann {

void read_filelist(lbann_comm* comm,
                   const std::string& fn,
                   std::vector<std::string>& filelist_out)
{
  const int rank = comm->get_rank_in_world();
  std::string f; // concatenated, space separated filelist
  int f_size;

  // P_0 reads filelist
  if (!rank) {

    std::ifstream in(fn.c_str());
    if (!in) {
      throw lbann_exception(std::string{} + __FILE__ + " " +
                            std::to_string(__LINE__) + " :: failed to open " +
                            fn + " for reading");
    }

    std::stringstream s;
    std::string line;
    while (getline(in, line)) {
      if (line.size()) {
        s << line << " ";
      }
    }
    in.close();

    f = s.str();
    f_size = s.str().size();
  }

  // bcast concatenated filelist
  comm->world_broadcast<int>(0, &f_size, 1);
  f.resize(f_size);
  comm->world_broadcast<char>(0, &f[0], f_size);

  // unpack filelist into vector
  std::stringstream s2(f);
  std::string filename;
  while (s2 >> filename) {
    if (filename.size()) {
      filelist_out.push_back(filename);
    }
  }
  if (!rank)
    std::cerr << "num files: " << filelist_out.size() << "\n";
}

} // namespace lbann
