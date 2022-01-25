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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/execution_algorithms/batch_functional_inference_algorithm.hpp"

namespace lbann {

El::Matrix<int, El::Device::CPU>
batch_functional_inference_algorithm::infer(observer_ptr<model> model,
        size_t mbs) {
    if (mbs <= 0) {
      LBANN_ERROR("mini-batch size must be larger than 0");
    }

    El::Matrix<int, El::Device::CPU> labels(mbs, 1);

    auto c = SGDExecutionContext(execution_mode::inference, mbs);
    model->reset_mode(c, execution_mode::inference);
    get_trainer().get_data_coordinator().reset_mode(c);

    get_trainer().get_data_coordinator().fetch_data(execution_mode::inference);
    model->forward_prop(execution_mode::inference);
    get_labels(*model, labels);
    // Create an SGD_execution_context so that layer.forward_prop can get the
    // mini_batch_size - This should be fixed in the future, when SGD is not so
    // hard-coded into the model & layers
    /*
    auto c = SGDExecutionContext(execution_mode::inference, mbs);
    model->reset_mode(c, execution_mode::inference);

    model->forward_prop(execution_mode::inference);
    */

    return labels;
  }

} // namespace lbann
