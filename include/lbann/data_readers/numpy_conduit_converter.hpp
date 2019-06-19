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

#ifndef NUMPY_CONDUIT_CONVERTER_HPP
#define NUMPY_CONDUIT_CONVERTER_HPP

#include "lbann_config.hpp"
#include "conduit/conduit.hpp"

namespace lbann {

/**
 * The numpy_conduit_converter class contains static method(s) for
 * reading numpy files and copying the contents to a conduit file.
 *
 * In general the schema for npz files, after conversion to conduit, is:
 *
 * @code{.unparsed}
 * {
 *   data_id (int) :
 *   // one or more of the following sections
 *   {
 *     section_name :
 *     {
 *       "word_size": <int>,
 *       "fortran_order: <0|1>,
 *       "num_vals": <int>,
 *       "shape": <[ vector ]>,
 *       "data": <char*>
 *     }
 *   }
 * }
 * @endcode
 *
 * cosmoflow has the following sections:
 * @code{.unparsed}
 *    "data":
 *    "frm":
 *    "responses":
 * @endcode
 */

class numpy_conduit_converter {
 public:

  static void load_conduit_node(const std::string filename, int data_id, conduit::Node &output, bool reset_conduit_node = true);

};

}  // namespace lbann

#endif  // NUMPY_CONDUIT_CONVERTER_HPP
