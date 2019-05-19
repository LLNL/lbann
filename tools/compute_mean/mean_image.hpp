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

#ifndef _TOOLS_COMPUTE_MEAN_MEAN_IMAGE_HPP_
#define _TOOLS_COMPUTE_MEAN_MEAN_IMAGE_HPP_
#include <string>
#include "mpi_states.hpp"
#include "lbann/data_readers/opencv.hpp"
#include "lbann/data_readers/cv_process.hpp"

namespace tools_compute_mean {

bool write_mean_image(const lbann::cv_process& pp, const int mean_extractor_idx, const mpi_states& ms, const std::string& out_dir);

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_MEAN_IMAGE_HPP_
