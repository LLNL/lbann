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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LIBRARY_HPP
#define LBANN_LIBRARY_HPP

#include "lbann/models/model.hpp"
#include "lbann/proto/proto_common.hpp"

namespace lbann {

const int lbann_default_random_seed = 42;

std::unique_ptr<thread_pool> construct_io_thread_pool(lbann_comm *comm);

std::unique_ptr<model> build_model_from_prototext(
    int argc, char **argv,
    lbann_data::LbannPB &pb,
    lbann_comm *comm,
    std::shared_ptr<thread_pool> io_thread_pool,
    bool first_model);

void print_lbann_configuration(
    lbann_data::Model *pb_model, lbann_comm *comm,
    int io_threads_per_process, int io_threads_offset);

} // namespace lbann

#endif // LBANN_LIBRARY_HPP
