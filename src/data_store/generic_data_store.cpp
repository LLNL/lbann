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

#include "lbann/data_store/generic_data_store.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/utils/options.hpp"

namespace lbann {

generic_data_store::generic_data_store(lbann_comm *comm, generic_data_reader *reader) :
    m_in_memory(false),
    m_comm(comm), m_master(comm->am_world_master()), m_reader(reader),
    m_cur_idx(0)
  {
    if (options::get()->has_bool("ds_in_memory")) {
      m_in_memory = options::get()->get_bool("ds_in_memory");
    }
  }

void generic_data_store::setup() {
  m_reader->save_initial_state();

  set_shuffled_indices( &(m_reader->get_shuffled_indices()) );
  //get_my_indices();

  m_reader->restore_initial_state();
}

void generic_data_store::get_my_indices() {
  std::vector<int> indices;
  bool is_active_reader = true; //@todo needs fixing for distributed 

  /*
  std::stringstream s;
  s << "data_store_debug_" << m_comm->get_rank_in_world() << ".txt";
  std::ofstream out(s.str().c_str());
  */

  //int j = 1;
  while (true) {
    int n = m_reader->fetch_data_indices(indices);
    //std::cout << "n: " << n << std::endl;
    if (n != (int)indices.size()) {
      std::stringstream s2;
      s2 << __FILE__ << " " << __LINE__ << " :: "
         << " something is badly wrong!";
      throw lbann_exception(s2.str());
    }  

    /*
    out << "call #" << j++ << " returned " << n << " indices: ";
    for (size_t k=0; k<indices.size(); k++) {
      out << indices[k] << " ";
    }
    out << std::endl;
    */
    bool is_done = m_reader->update(is_active_reader);
    if (!is_done) {
      break;
    }
  }

  //out.close();

}

}  // namespace lbann
