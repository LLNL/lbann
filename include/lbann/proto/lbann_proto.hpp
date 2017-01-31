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

//#include "lbann/lbann_params.hpp"
#include "lbann/proto/lbann.pb.h"
#include <string>
#include <iostream>

namespace lbann
{

class lbann_proto {
public :

  /// returns a pointer to the lbann_proto singleton
  static lbann_proto * get() {
    return s_instance;
  }

  /// for testing during development
  lbann_data::LbannPB & getLbannPB() { return m_pb; }

  void init(const char *filename = 0);

  void test() {
     std::cerr << "in test!\n\n";
  }

  void writePrototextFile(const char *filename);

  void DataReaderMNIST_ctor(
    int batch_size, 
    bool shuffle); 

  void DataReaderMNIST_load(
    std::string file_dir, 
    std::string image_file, 
    std::string label_file); 

  void Model_ctor(
    std::string name, 
    std::string objective_function, 
    std::string optimizer);

  void Model_train(
    int num_epochs, 
    int evaluation_frequency);

  void Layer_InputDistributedMiniBatchParallelIO_ctor(
    int num_parallel_readers, 
    int mini_batch_size); 

  void Layer_FullyConnected_ctor(
    int num_prev_neurons, 
    int num_neurons, 
    int mini_batch_size, 
    std::string activation_type, 
    std::string weight_initialization, 
    std::string optimizer);



  /// Returns a TrainingParam object. If init() was not called, the returned object
  /// will contain whatever default values are defined in the TrainingParam ctor.
  /// If init() was called, some or all defaults will be overriden by the contents
  /// of the prototxt file
  //TrainingParams & getTrainingParams();

  //PerformanceParams & getPerformanceParams();

  //NetworkParams & getNetworkParams();

  //SystemParams & getSystemParams();

  /// Sets internal prototcol buffer fields from the TrainingParams object.
  /// You should call this prior to calling write()
  /*
  void load(const TrainingParams &training);

  void load(const PerformanceParams &performance);

  void load(const NetworkParams &network);

  void load(const SystemParams &System);
  

  /// Writes a prototxt file that saves an lbann model. 
  /// You should call the appropriate load() methods prior to calling write()
  void write(const std::string &filename);
  */

private:
  static lbann_proto *s_instance;

  lbann_proto();
  ~lbann_proto();
  lbann_proto(lbann_proto &) {}
  lbann_proto operator=(lbann_proto&) { return *this; }

  //bool m_init_from_prototxt;

  lbann_data::LbannPB m_pb;
  lbann_data::DataReader *m_reader;

  
  void allocateDataReader();
};

} //namespace lbann

#endif //LBANN_PROTO_HPP_INCLUDED

