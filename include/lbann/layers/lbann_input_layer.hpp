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

#ifndef LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/data_readers/lbann_data_reader.hpp"

namespace lbann
{
  class input_layer : public Layer {
  public:
    input_layer(lbann_comm* comm, uint mini_batch_size, DataReader* training_data_reader, DataReader* testing_data_reader,
                std::vector<regularizer*> regs={});
    input_layer(lbann_comm* comm, uint mini_batch_size, DataReader* training_data_reader);
    DistMat *bp_output();
    DataReader *select_data_reader();
    DataReader *set_training_data_reader(DataReader *data_reader);
    DataReader *set_testing_data_reader(DataReader *data_reader);
    long update_num_samples_processed(long num_samples);

    long get_num_samples_trained() { return m_num_training_samples_processed; }
    long get_num_samples_tested() { return m_num_testing_samples_processed; }
    long get_total_num_training_samples() { return m_total_training_samples; }
    long get_total_num_testing_samples() { return m_total_testing_samples; }

  public:
    DataReader *m_training_data_reader;
    DataReader *m_testing_data_reader;
    long m_num_training_samples_processed;
    long m_total_training_samples;
    long m_num_testing_samples_processed;
    long m_total_testing_samples;
  };
}

#endif  // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
