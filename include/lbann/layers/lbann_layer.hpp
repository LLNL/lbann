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
// lbann_layer .h .cpp - Parent class for all layer types
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_HPP_INCLUDED
#define LBANN_LAYER_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_summary.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"
#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include <string>
#include <vector>

namespace lbann
{

// Forward-declare this.
class regularizer;

  class Layer {
  public:
    Layer(const uint index, lbann_comm* comm, Optimizer *optimizer,
          uint mbsize);
    Layer(const uint index, lbann_comm* comm, Optimizer *optimizer,
          uint mbsize, std::vector<regularizer*> regs);
    virtual ~Layer();
    virtual DataType forwardProp(DataType prev_WBL2NormSum);
    virtual void backProp();
    virtual bool update() { return false; };
    virtual void summarize(lbann_summary& summarizer, int64_t step);
    /**
     * Print information at the end of an epoch.
     * This is always called on the model masters and should synchronize
     * printing if needed.
     */
    virtual void epoch_print() const {}
    virtual DataType checkGradientMB(Layer& PrevLayer, const DataType Epsilon=1e-4) {
      return 0.0;
    };

    virtual void setup(int);

    /** Return the index of this layer. */
    inline uint get_index() const { return Index; }
    /** Return (a view of) the weights/biases matrix for this layer. */
    virtual ElMat& get_weights_biases() { return *WB; }
    /** Return (a view of) the weights/biases gradient matrix for this layer. */
    virtual ElMat& get_weights_biases_gradient() { return *WB_D; }
    /** Return the layer's optimizer. */
    virtual Optimizer* get_optimizer() const { return optimizer; }

    /**
     * Get the "effective" size of a mini-batch.
     * This is for backward propagation, etc. when there are more updates being
     * contributed than the local mini-batch size implies (e.g. when doing
     * inter-model updates).
     */
    virtual uint get_effective_minibatch_size() const {
      return m_effective_mbsize;
    }
    /** Set the effective size of a mini-batch to size. */
    virtual void set_effective_minibatch_size(uint size) {
      m_effective_mbsize = size;
    }

    ElMat *fp_output();
    ElMat *bp_output();
    void setup_fp_input(ElMat *fp_input);
    void setup_bp_input(ElMat *bp_input);

    /* void updateMB(const float LearnRate); */
    //    virtual double computeCost(DistMat &deltas) = 0;
    //    { return 0.0;}

    bool saveToFile(int fd, const char* filename);
    bool loadFromFile(int fd, const char* filename);

    bool saveToCheckpoint(int fd, const char* filename, uint64_t* bytes);
    bool loadFromCheckpoint(int fd, const char* filename, uint64_t* bytes);

    bool saveToCheckpointShared(const char* dir, uint64_t* bytes);
    bool loadFromCheckpointShared(const char* dir, uint64_t* bytes);

  public:
    uint               Index;                  // Layer index (start with 0)
    uint 		NumNeurons; 	// # neurons
    execution_mode  m_execution_mode;

    // TODO: move to lbann_layer_fully_connected.hpp
    ElMat *WB;             // Weight and Bias Set ((# neurons + 1) x (# previous layer's neurons + 1))
    ElMat *WB_D;           // Weights and Bias Gradient ((# neurons + 1) x (# previous layer's neurons + 1))
    ElMat *Zs;             // Zs ((# neurons + 1) x mini-batch size)
    ElMat *Ds;             // Deltas ((# neurons + 1) x mini-batch size)

    ElMat *Ds_Temp;        // Temporary deltas for computation ((# neurons + 1) x mini-batch size)
    ElMat *Acts;           // Activations ((# neurons + 1) x mini-batch size)

    Optimizer *optimizer;

    ElMat *fp_input;
    ElMat *bp_input;

    lbann_comm* comm;
  protected:
    /** Apply the layer's linear update in forward propagation. */
    virtual void fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y) {}
    /** Handle the layer's linearity in backward propagation. */
    virtual void bp_linearity() {}
    /** Apply the layer's nonlinearity in forward propagation. */
    virtual void fp_nonlinearity() {}
    /** Handle the layer's nonlinearity in backward propagation. */
    virtual void bp_nonlinearity() {}

    /** Regularizers being applied to the layer. */
    std::vector<regularizer*> regularizers;
    /** Size of the local mini-batch. */
    uint m_mini_batch_size;
    /** "Effective" mini-batch size for backward propagation, etc.. */
    uint m_effective_mbsize;

    /** Time spent in forward propagation. */
    double fp_time;
    /** Time spent in backward propagation. */
    double bp_time;
  };
}


#endif // LBANN_LAYER_HPP_INCLUDED
