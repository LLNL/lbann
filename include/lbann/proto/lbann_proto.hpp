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

/**
 * the lbann_proto class is a singleton that provides functionality
 * for reading/writing lbann models and associated data to/from
 * protocol buffers
 */

#ifndef LBANN_PROTO_HPP_INCLUDED
#define LBANN_PROTO_HPP_INCLUDED

#include "lbann/lbann_params.hpp"
#include "lbann/proto/lbann.pb.h"
#include <string>

namespace lbann
{

class lbann_proto {
public :

  /// returns a pointer to the lbann_proto singleton
  static lbann_proto * get() {
    return s_instance;
  }

  /// read messages from a prototxt file. Call this method if you want to
  /// load an lbann model from file. After calling this you should call the
  /// appropriate getXXX() method(s) below.
  void init(const std::string &filename);

  /// Returns a TrainingParam object. If init() was not called, the returned object
  /// will contain whatever default values are defined in the TrainingParam ctor.
  /// If init() was called, some or all defaults will be overriden by the contents
  /// of the prototxt file
  TrainingParams & getTrainingParams();

  PerformanceParams & getPerformanceParams();

  NetworkParams & getNetworkParams();

  SystemParams & getSystemParams();

  /// Sets internal prototcol buffer fields from the TrainingParams object.
  /// You should call this prior to calling write()
  void load(const TrainingParams &training);

  void load(const PerformanceParams &performance);

  void load(const NetworkParams &network);

  void load(const SystemParams &System);

  /// Writes a prototxt file that saves an lbann model. 
  /// You should call the appropriate load() methods prior to calling write()
  void write(const std::string &filename);

private:

  lbann_proto();
  ~lbann_proto();
  lbann_proto(lbann_proto &) {}
  lbann_proto operator=(lbann_proto&) { return *this; }

  static lbann_proto *s_instance;

  bool m_init_from_prototxt;
};

}

#endif //LBANN_PROTO_HPP_INCLUDED

