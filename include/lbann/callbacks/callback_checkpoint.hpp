//////////////////////////////////////////////////////////////////////////////
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
// lbann_callback_checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/io/persist.hpp"

namespace lbann {

/**
 *  * Checkpoint at given interval in given directory
 *   */
class lbann_callback_checkpoint : public lbann_callback {
 public:

  /**
 * @param checkpoint_dir directory to save checkpoint files
 * @param checkpoint_epochs interval to checkpoint
 * @param checkpoint_steps interval to checkpoint
 * @param checkpoint_secs interval to checkpoint
 * @param checkpoint_per_rank true to save/load a file per mpi rank
 */
  lbann_callback_checkpoint(std::string checkpoint_dir, 
                            int checkpoint_epochs, int checkpoint_steps, int checkpoint_secs, std::string per_rank_dir, int ckpt_dist_epochs, int ckpt_dist_steps) : 
    lbann_callback(),
    m_checkpoint_dir(checkpoint_dir),
    m_checkpoint_epochs(checkpoint_epochs), 
    m_checkpoint_steps(checkpoint_steps), 
    m_checkpoint_secs(checkpoint_secs), 
    m_per_rank_dir(per_rank_dir),
    m_ckpt_dist_epochs(ckpt_dist_epochs), 
    m_ckpt_dist_steps(ckpt_dist_steps) {}
  lbann_callback_checkpoint(const lbann_callback_checkpoint&) = default;
  lbann_callback_checkpoint& operator=(const lbann_callback_checkpoint&) = default;
  lbann_callback_checkpoint* copy() const override { return new lbann_callback_checkpoint(*this); }
  void setup(model *m) override;
  void on_epoch_end(model *m) override;
  void on_batch_end(model *m) override;
  void on_validation_end(model *m) override;

  inline void set_checkpoint_dir(std::string dir){
    m_checkpoint_dir= dir;
  }

  inline void set_checkpoint_epochs(int epochs){
    m_checkpoint_epochs= epochs;
  }

  inline void set_checkpoint_steps(int steps){
    m_checkpoint_steps= steps;
  }

  inline void set_checkpoint_secs(EvalType secs){
    m_checkpoint_secs= secs;
  }
  
  inline void set_per_rank_dir(std::string dir){
    m_per_rank_dir = dir;
  }

  inline void set_ckpt_dist_epochs(int ckpt_dist_epochs){                                                                  
    m_ckpt_dist_epochs = ckpt_dist_epochs;
  }  

  inline void set_ckpt_dist_steps(int ckpt_dist_steps){
    m_ckpt_dist_steps = ckpt_dist_steps;
  }

  bool need_checkpoint(model *m);
  bool checkpoint(model *m);
  bool restart(model *m);
  std::string name() const override { return "checkpoint"; }
 protected:
  std::string m_checkpoint_dir;
  int m_checkpoint_epochs;
  int m_checkpoint_steps;
  EvalType m_checkpoint_secs;
  std::string m_per_rank_dir;
  int m_ckpt_dist_epochs;
  int m_ckpt_dist_steps;
  EvalType m_checkpoint_last;
  bool m_epoch_end;
  bool m_val_end;
  bool m_mb_end;
  bool m_checkpoint_dist;
  bool m_checkpoint_shared;
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
