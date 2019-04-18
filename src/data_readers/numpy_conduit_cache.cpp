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

#include "lbann/data_readers/numpy_conduit_cache.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_CONDUIT

namespace lbann {


void numpy_conduit_cache::load(const std::string filename, int data_id) {

  try {
    m_numpy[data_id] = cnpy::npz_load(filename);
    conduit::Node &n = m_data[data_id];
    std::map<std::string, cnpy::NpyArray> &a = m_numpy[data_id];
    for (auto &&t : a) {
      cnpy::NpyArray &b = t.second;
      n[std::to_string(data_id) + "/" + t.first + "/word_size"] = b.word_size;
      n[std::to_string(data_id) + "/" + t.first + "/fortran_order"] = b.fortran_order;
      n[std::to_string(data_id) + "/" + t.first + "/num_vals"] = b.num_vals;
      n[std::to_string(data_id) + "/" + t.first + "/shape"] = b.shape;
      std::shared_ptr<std::vector<char>> data = b.data_holder;
      n[std::to_string(data_id) + "/" + t.first + "/data"].set_external_char_ptr(b.data_holder->data());
    }
  } catch (...) {
    //note: npz_load throws std::runtime_error, but I don't want to assume
    //      that won't change in the future
    LBANN_ERROR("failed to open " + filename + " during cnpy::npz_load");
  }
}

const conduit::Node & numpy_conduit_cache::get_conduit_node(int data_id) const {
  std::unordered_map<int, conduit::Node>::const_iterator it = m_data.find(data_id);
  if (it == m_data.end()) {
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_data");
  }
  return it->second;
}



} // end of namespace lbann

#endif // LBANN_HAS_CONDUIT
