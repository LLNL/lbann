////////////////////////////////////////////////////////////////////////////////
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

#ifndef __GENERIC_DATA_STORE_HPP__
#define __GENERIC_DATA_STORE_HPP__


#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include <vector>
#include <ostream>

namespace lbann {

class generic_data_reader;
class lbann_comm;

/**
 * todo
 */

class generic_data_store {
 public:

  //! ctor
  generic_data_store(lbann_comm *comm, generic_data_reader *reader); 

  //! copy ctor
  generic_data_store(const generic_data_store&) = default;

  //! operator=
  generic_data_store& operator=(const generic_data_store&) = default;

  //! dtor
  virtual ~generic_data_store() {}

  virtual generic_data_store * copy() const = 0;

  //! returns a pointer to the data reader; may not be necessary
  generic_data_reader * get_data_reader() {
    return m_reader;
  }

  virtual void setup();

  void set_shuffled_indices(const std::vector<int> *indices) {
    m_shuffled_indices = indices;
    m_cur_idx = 0;
  }

  virtual void get_data_buf(std::string dir, std::string filename, std::vector<unsigned char> *&buf, int tid) = 0;

  void update() {
    m_cur_idx = 0;
  }

  /// for use during development and debugging
  void print_indices(std::ostream &out);

 protected :

  bool m_in_memory;

  lbann_comm *m_comm;
  bool m_master;
  generic_data_reader *m_reader;

  int m_cur_idx;

  //std::vector<std::vector<int> > *m_my_indices;

  const std::vector<int> *m_shuffled_indices;

  /// fills in m_my_indices; these are indices into the shuffled
  /// indices; they do not change!
  void get_my_indices();

  std::vector<std::vector<unsigned char> > m_buffers;
};

}  // namespace lbann

#endif  // __GENERIC_DATA_STORE_HPP__
