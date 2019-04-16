// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef NUMPY_CONDUIT_CACHE_HPP
#define NUMPY_CONDUIT_CACHE_HPP

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT

#include <cnpy.h>
#include "lbann/comm.hpp"
#include <unordered_map>
#include "conduit/conduit.hpp"

namespace lbann {

/**
 * A numpy_conduit_cache has functionality to read numpy files,
 * convert them to conduit, and return the conduit::Node
 * upon request.
 */

class numpy_conduit_cache {
 public:

  numpy_conduit_cache(lbann_comm *comm) : m_comm(comm)
  {}

  numpy_conduit_cache(const numpy_conduit_cache&);

  numpy_conduit_cache& operator=(const numpy_conduit_cache&);

  ~numpy_conduit_cache() {}

  //! Load a numpy array from file; assign it the given data_id;
  //! and cache it internally. For now we assume there is 
  //! a single ndarray per file; this may change in the future
  void load(const std::string filename, int data_id);

  const conduit::Node & get_data(int data_id) const;

protected :

  void copy_members(const numpy_conduit_cache &rhs);

  lbann_comm *m_comm;

  std::unordered_map<int, conduit::Node> m_data;

  // we keep the numpy data to avoid a deep copy in m_data;
  std::unordered_map<int, std::map<std::string, cnpy::NpyArray> > m_numpy;

};

}  // namespace lbann

#endif  // #ifdef LBANN_HAS_CONDUIT

#endif  // NUMPY_CONDUIT_CACHE_HPP
