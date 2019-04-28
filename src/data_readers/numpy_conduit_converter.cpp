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

#include "lbann/data_readers/numpy_conduit_converter.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

//static
void numpy_conduit_converter::load_conduit_node(const std::string filename, int data_id, conduit::Node &output, bool reset_conduit_node) {

  try {
    if (reset_conduit_node) {
      output.reset();
    }
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filename);
    for (auto &&t : a) {
      cnpy::NpyArray &b = t.second;
      output[std::to_string(data_id) + "/" + t.first + "/word_size"] = b.word_size;
      output[std::to_string(data_id) + "/" + t.first + "/fortran_order"] = b.fortran_order;
      output[std::to_string(data_id) + "/" + t.first + "/num_vals"] = b.num_vals;
      output[std::to_string(data_id) + "/" + t.first + "/shape"] = b.shape;

      // conduit makes a copy of the data, and owns the data, hence it
      // will be properly deleted when then conduit::Node is deleted
      char *data = b.data_holder->data();
      output[std::to_string(data_id) + "/" + t.first + "/data"].set_char_ptr(data, b.word_size*b.num_vals);
    }
  } catch (...) {
    //note: npz_load throws std::runtime_error, but I don't want to assume
    //      that won't change in the future
    LBANN_ERROR("failed to open " + filename + " during cnpy::npz_load");
  }
}

} // end of namespace lbann
