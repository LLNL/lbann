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
// variable .hpp .cpp - Layer variable class
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_VARIABLE_HPP
#define LBANN_VARIABLE_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include <string>

namespace lbann {

// Class prototype
class optimizer;

/** Layer variable class. */
class variable {

 public:

  /** Constructor. */
  variable(lbann_comm* comm,
           cudnn::cudnn_manager* cudnn = nullptr);

  /** Copy constructor. */
  variable(const variable& other);
  /** Copy assignment operator. */
  variable& operator=(const variable& other);
  /** Destructor. */
  virtual ~variable();

  /** Set variable name.
   *  Each variable in a model should have a unique name.
   */
  void set_name(std::string name) { m_name = name; }

  /** Get variable name. */
  std::string get_name() const { return m_name; }

  /** Create a copy of the variable. */
  virtual variable* copy() const { return new variable(*this); }

  /** Setup variable. */
  virtual void setup(int height,
                     int width,
                     El::Distribution col_dist,
                     El::Distribution row_dist);
  /** Setup GPU objects for variable. */
  virtual void setup_gpu();

  /** Get variable initializer. */
  variable_initializer& get_initializer() { return *m_initializer; }
  /** Get variable initializer (const). */
  const variable_initializer& get_initializer() const { return *m_initializer; }
  /** Set variable initializer.
   *  The variable takes ownership of the initializer and deallocates
   *  it during destruction.
   */
  void set_initializer(variable_initializer* initializer);

  /** Get variable optimizer. */
  optimizer* get_optimizer() { return m_optimizer; }
  /** Get variable optimizer (const). */
  const optimizer* get_optimizer() const { return m_optimizer; }
  /** Set variable optimizer.
   *  The variable takes ownership of the optimizer and deallocates it
   *  during destruction.
   */
  void set_optimizer(optimizer* opt);

  /** Get the variable matrix. */
  AbsDistMat& get_values();
  /** Get the variable matrix (const). */
  const AbsDistMat& get_values() const;
  /** Set the variable matrix. */
  void set_values(const AbsDistMat& values);

  /** Get a view into the variable matrix.
   *  If values_v has a different matrix distribution than the
   *  variable matrix, the matrix values are copied into values_v.
   */
  void get_values_view(AbsDistMat& values_v) const;

 protected:

  /** Variable name.
   *  Each variable in a model should have a unique name.
   */
  std::string m_name;

  /** LBANN communicator. */
  lbann_comm* m_comm;
  /** cuDNN manager. */
  cudnn::cudnn_manager* m_cudnn;

  /** Variable matrix. */
  AbsDistMat* m_values;

  /** Variable initializer.
   *  Default is zero initialization.
   */
  variable_initializer* m_initializer;
  /** Variable optimizer.
   *  Default is nullptr, which corresponds to no optimizer.
   */
  optimizer* m_optimizer;

};

} // namespace lbann

#endif // LBANN_VARIABLE_HPP
