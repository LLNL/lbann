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
// weights .hpp .cpp - Layer weights class
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_WEIGHTS_HPP
#define LBANN_WEIGHTS_HPP

#include <string>

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/io/persist.hpp"
namespace lbann {

// Forward declaration
class optimizer;

/** Layer weights.
 *  Similar to Tensorflow "variables."
 */
class weights {

  friend class optimizer;

 public:

  /** Constructor. */
  weights(lbann_comm* comm,
          cudnn::cudnn_manager* cudnn = nullptr);

  /** Copy constructor. */
  weights(const weights& other);
  /** Copy assignment operator. */
  weights& operator=(const weights& other);
  /** Destructor. */
  virtual ~weights();

  /** Set weights name.
   *  Each set of weights in a model should have a unique name.
   */
  void set_name(std::string name) { m_name = name; }

  /** Get weights name. */
  std::string get_name() const { return m_name; }

  /** Create a copy of the weights. */
  virtual weights* copy() const { return new weights(*this); }

  /** Setup weights. */
  virtual void setup(int height,
                     int width,
                     El::Distribution col_dist,
                     El::Distribution row_dist);
  /** Setup GPU objects for weights. */
  virtual void setup_gpu();

  /** Get height of weights matrix. */
  int get_height() const { return m_height; }
  /** Get width of weights matrix. */
  int get_width() const { return m_width; }

  /** Get cuDNN manager. */
  cudnn::cudnn_manager* get_cudnn_manager() { return m_cudnn; }

  /** Get weights initializer. */
  weights_initializer& get_initializer() { return *m_initializer; }
  /** Get weights initializer (const). */
  const weights_initializer& get_initializer() const { return *m_initializer; }
  /** Set weights initializer.
   *  This takes ownership of the initializer and deallocates it
   *  during destruction.
   */
  void set_initializer(weights_initializer* initializer);

  /** Get weights optimizer. */
  optimizer* get_optimizer() { return m_optimizer; }
  /** Get weights optimizer (const). */
  const optimizer* get_optimizer() const { return m_optimizer; }
  /** Set weights optimizer.
   *  This takes ownership of the optimizer and deallocates it during
   *  destruction.
   */
  void set_optimizer(optimizer* opt);

  /** Get the weights matrix. */
  const AbsDistMat& get_values();
  /** Set the weights matrix. */
  void set_values(const AbsDistMat& values);
  /** Set an entry in the weights matrix. */
  void set_value(int row, int col, DataType value);

  /** Get a view into the weights matrix.
   *  If values_v has a different matrix distribution than the
   *  weights matrix, the matrix values are copied into values_v.
   */
  void get_values_view(AbsDistMat& values_v);

#ifdef __LIB_CUDNN
  /** Get the weights matrix on GPU. */
  std::vector<DataType*> get_values_gpu();
#endif // __LIB_CUDNN

  bool save_to_checkpoint_shared(persist& p);
  bool load_from_checkpoint_shared(persist& p);
 protected:

  /** Weights name.
   *  Each set of weights in a model should have a unique name.
   */
  std::string m_name;

  /** LBANN communicator. */
  lbann_comm* m_comm;
  /** cuDNN manager. */
  cudnn::cudnn_manager* m_cudnn;

  /** Height of weights matrix. */
  int m_height;
  /** Width of weights matrix. */
  int m_width;

  /** Weights matrix. */
  AbsDistMat* m_values;

  /** Weights initializer.
   *  Default is zero initialization.
   */
  weights_initializer* m_initializer;
  /** Weights optimizer.
   *  Default is nullptr, which corresponds to no optimizer.
   */
  optimizer* m_optimizer;

#ifdef __LIB_CUDNN
  /** GPU memory for weights matrix. */
  std::vector<DataType*> m_values_d;
#endif // __LIB_CUDNN

};

} // namespace lbann

#endif // LBANN_WEIGHTS_HPP
