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

#include "lbann/callbacks/callback_gpu_memory_usage.hpp"
#include <iomanip>

namespace lbann {

void lbann_callback_gpu_memory_usage::on_epoch_begin(model *m) {
#ifdef LBANN_HAS_CUDA
  size_t available;
  size_t total;
  FORCE_CHECK_CUDA(cudaMemGetInfo(&available, &total));
  size_t used = total - available;
  std::cout << "GPU memory usage at epoch " << m->get_cur_epoch()
            << " of model " << m->get_comm()->get_model_rank()
            << " at rank " << m->get_comm()->get_rank_in_model()
            << ": " << used << " bytes ("
            << std::setprecision(3)
            << (used / 1024.0 / 1024.0 / 1024.0) << " GiB) used out of "
            << total << " bytes ("
            << std::setprecision(3)      
            << (total / 1024.0 / 1024.0 / 1024.0)
            << " GiB)" << std::endl;
#endif
}

}  // namespace lbann
