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
// lbann_layer_softmax .hpp .cpp - Softmax layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYER_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include <string>

namespace lbann
{
  class SoftmaxLayer: public Layer
  {
  public:
    SoftmaxLayer(data_layout data_dist,
                 uint index,
                 int numPrevNeurons,
                 uint numNeurons,
                 uint minim_batch_size,
                 weight_initialization init,
                 lbann_comm* comm,
                 optimizer *opt);
    ~SoftmaxLayer();
    void initialize_model_parallel_distribution();
    void initialize_data_parallel_distribution();
    void setup(int numPrevNeurons);
    bool update();
    DataType checkGradient(Layer& PrevLayer, const DataType Epsilon=1e-4);
    //        void updateMB(const float LearnRate);
    DataType WBL2norm();

  protected:
    void fp_linearity();
    void bp_linearity();
    void fp_nonlinearity();
    void bp_nonlinearity();
    void fp_set_std_matrix_view();

  public:
    DataType   WBL2NormSum;

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes);
    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes);
    bool saveToCheckpointShared(persist& p);
    bool loadFromCheckpointShared(persist& p);

  private:
    weight_initialization m_weight_initialization;
    AbsDistMat *m_workspace;
    AbsDistMat *m_workspace_v;
  };
}


#endif // LBANN_LAYER_SOFTMAX_HPP_INCLUDED
