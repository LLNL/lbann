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
// lbann_callback_imcomm .hpp .cpp - Send gradient updates between models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "lbann/callbacks/lbann_callback.hpp"
#include "lbann/utils/lbann_quantizer.hpp"

namespace lbann {

/**
 * Support inter-model communication after each mini-batch to synchronize
 * gradient updates.
 * This optionally supports quantizing the gradient updates before communication
 * in order to reduce bandwidth requirements.
 */
class lbann_callback_imcomm : public lbann_callback {
public:
  enum comm_type {
    NONE,  /** Do no gradient updates. */
    NORMAL,  /** Simply sum gradient updates. */
    ONEBIT_QUANTIZATION,  /** Do one-bit quantization with AdaGrad. */
    THRESH_QUANTIZATION,  /** Do thresholded one-bit quantization. */
    COMPRESSED_THRESH_QUANTIZATION,  /** Do compressed thresholded one-bit quantization. */
    ADAPTIVE_THRESH_QUANTIZATION,  /** Do adaptive thresholded one-bit quantization. */
    COMPRESSED_ADAPTIVE_THRESH_QUANTIZATION,  /** Do compressed adaptive thresholded one-bit quantization. */
    NORMAL_AR  /** Sum gradient updates but use the custom allreduce. */
  };
  /** Do inter-model gradient updates of the given type. */
  lbann_callback_imcomm(comm_type ct = NONE, lbann_summary* _summarizer = nullptr);
  /**
   * Do inter-model gradient updates of the given type, only for the layers in
   * the layers set.
   */
  lbann_callback_imcomm(comm_type ct, std::unordered_set<uint> _layers,
                        lbann_summary* _summarizer = nullptr);
  /** Do initialization for this model. */
  void setup(model* m);
  /** Clear out remaining error if needed. */
  void on_epoch_end(model* m);
  /** Do inter-model gradient updates. */
  void on_backward_prop_end(model* m);
private:
  /** Communication type. */
  comm_type ct;
  /** Quantizer for quantization of updates, if needed. */
  lbann_quantizer quantizer;
  /** Per-layer quantization errors. */
  std::unordered_map<uint, Mat> quantization_errors;
  /** Per-layer inter-model sum quantization errors. */
  std::unordered_map<uint, Mat> im_quantization_errors;
  /** Per-layer gradient history when using one-bit quantization. */
  std::unordered_map<uint, Mat> gradhistories;
  /** Layers indicies to quantize. */
  std::unordered_set<uint> layer_indices;

  /** Return true if the comm type does quantization. */
  inline bool ct_does_quantization() const {
    return (ct == ONEBIT_QUANTIZATION ||
            ct == THRESH_QUANTIZATION ||
            ct == COMPRESSED_THRESH_QUANTIZATION ||
            ct == ADAPTIVE_THRESH_QUANTIZATION ||
            ct == COMPRESSED_ADAPTIVE_THRESH_QUANTIZATION);
  }
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_IMCOMM_HPP_INCLUDED
