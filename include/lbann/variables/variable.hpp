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

/// Layer variable class
class variable {

 public:

  /// Constructor
  variable(lbann_comm* comm,
           variable_initializer* initializer = nullptr,
           cudnn::cudnn_manager* cudnn = nullptr);
  variable(const variable& other);
  variable& operator=(const variable& other);

  /// Destructor
  virtual ~variable();

  void set_name(const std::string name) { m_name = name; }

  std::string get_name() const { return m_name; }

  virtual variable* copy() const { return new variable(*this); }

  virtual void setup(int height,
                     int width,
                     El::Distribution col_dist,
                     El::Distribution row_dist);
  
  virtual void setup_gpu();

  variable_initializer& get_initializer() { return *m_initializer; }
  const variable_initializer& get_initializer() const { return *m_initializer; }
  void set_initializer(variable_initializer* initializer);

  optimizer& get_optimizer() { return *m_optimizer; }
  const optimizer& get_optimizer() const { return *m_optimizer; }
  void set_optimizer(optimizer* opt);

  AbsDistMat& get_values() { return *m_values; }
  const AbsDistMat& get_values() const { return *m_values; }
  void set_values(const AbsDistMat& values);

  void get_values_view(AbsDistMat& values_v) const;

 protected:

  std::string m_name;

  /// LBANN communicator
  lbann_comm* m_comm;
  /// cuDNN manager
  cudnn::cudnn_manager* m_cudnn;

  AbsDistMat* m_values;

  variable_initializer* m_initializer;
  optimizer* m_optimizer;
  

};

} // namespace lbann

#endif // LBANN_VARIABLE_HPP
