////////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
//// Produced at the Lawrence Livermore National Laboratory.
//// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
//// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
////
//// LLNL-CODE-697807.
//// All rights reserved.
////
//// This file is part of LBANN: Livermore Big Artificial Neural Network
//// Toolkit. For details, see http://software.llnl.gov/LBANN or
//// https://github.com/LLNL/LBANN.
////
//// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
//// may not use this file except in compliance with the License.  You may
//// obtain a copy of the License at:
////
//// http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//// implied. See the License for the specific language governing
//// permissions and limitations under the license.
////
//// lbann_callback_checkpoint .hpp .cpp - Callback hooks to checkpoint model
//////////////////////////////////////////////////////////////////////////////////
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
 */
  lbann_callback_checkpoint(std::string checkpoint_dir, int checkpoint_epochs, int checkpoint_steps, int checkpoint_secs) : 
    lbann_callback(), m_checkpoint_dir(checkpoint_dir), m_checkpoint_epochs(checkpoint_epochs), m_checkpoint_steps(checkpoint_steps), m_checkpoint_secs(checkpoint_secs) {}
  lbann_callback_checkpoint(const lbann_callback_checkpoint&) = default;
  lbann_callback_checkpoint& operator=(const lbann_callback_checkpoint&) = default;
  lbann_callback_checkpoint* copy() const override { return new lbann_callback_checkpoint(*this); }
  void setup(model *m) override;
  void on_epoch_end(model *m) override;
  void on_batch_end(model *m) override;

  inline void set_checkpoint_dir(std::string dir){
    m_checkpoint_dir= dir;
  }

  inline void set_checkpoint_epochs(int epochs){
    m_checkpoint_epochs= epochs;
  }

  inline void set_checkpoint_steps(int steps){
    m_checkpoint_steps= steps;
  }

  inline void set_checkpoint_secs(double secs){
    m_checkpoint_secs= secs;
  }
  
  virtual bool at_epoch_start() {
    return true;
  }
  
  bool need_checkpoint(model *m);
  bool checkpointShared(model *m);
  bool restartShared(model *m);

  std::string name() const override { return "checkpoint"; }
 protected:
  std::string m_checkpoint_dir;
  int m_checkpoint_epochs;
  int m_checkpoint_steps;
  double m_checkpoint_secs;
  double m_checkpoint_last;
 
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECKPOINT_HPP_INCLUDED
