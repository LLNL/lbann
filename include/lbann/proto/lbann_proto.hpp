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
#include "lbann/lbann_params.hpp"
#include <string>
#include <vector>
#include <float.h>

#if 0
#define PB_FIX(a) (a == 0 ? -3 : a)
#define PB_FIXD(a) (a == 0.0 ? -3 : a)

#define PB_FIX_I(a) (a == -3 ? 0 : a)
#define PB_FIXD_I(a) (a == -3 ? 0.0 : a)
#endif
#define PB_FIX(a) (a)
#define PB_FIXD(a) (a)

#define PB_FIX_I(a) (a)
#define PB_FIXD_I(a) (a)

namespace lbann
{


class lbann_proto {
public :

struct data_reader_params {
  /// mnist, cifar10, imagenet, nci, nci_regression
  std::string name;

  /// train or test
  std::string role;

  std::string root_dir;
  std::string data_filename;
  std::string label_filename;

  int mini_batch_size;
  double percent_samples;
  bool shuffle;
};

struct optimizer_params {
  optimizer_params() : name("none"), learn_rate(-2), momentum(-2), nesterov(false) {}

  //adagrad, rmsprop, adam, sgd
  std::string name;
  double learn_rate;
  double momentum;
  double decay;
  bool nesterov;
};

struct model_params {
  model_params() : name("none"), objective_function("none"), mini_batch_size(-2) {}
  //dnn, greedy_layerwise_autoencoder, stacked_autoencoder
  std::string name;
  //categorical_cross_entropy, mean_squared_error
  std::string objective_function;
  int mini_batch_size;
  int num_epochs;

  std::vector<string> metrics;
  void add_metric(std::string s) { metrics.push_back(s); }
};


struct regularizer_params {
  //dropout
  std::string name;
  double dropout;
};

struct layer_params {
  //input_distributed_minibatch_parallel_io, fully_connected, 
  //target_distributed_minibatch_parallel_io, softmax
  std::string name;
  int mini_batch_size;
  int num_prev_neurons;
  int num_neurons;
  activation_type activation;
  weight_initialization weight_init;
  std::string optimizer;
  int num_parallel_readers;
  bool shared_data_reader;
  std::vector<regularizer_params> regularizers;
  void add_regularizer(regularizer_params &p) { regularizers.push_back(p); }
  bool for_regression;
};

  /// returns a pointer to the lbann_proto singleton
  static lbann_proto * get() {
    return s_instance;
  }

  lbann_data::LbannPB & getLbannPB() { return m_pb; }

  //only the master will 
  void set_master(bool m) { m_master = m; }

  void add_network_params(const NetworkParams &p);
  void add_performance_params(const PerformanceParams &p);
  void add_system_params(const SystemParams &p);
  void add_training_params(const TrainingParams &p);

  void add_data_reader(const data_reader_params &p);
  void add_optimizer(const optimizer_params &p);
  void add_model(const model_params &p);
  void add_layer(const layer_params &p);

  void readPrototextFile(const char *fn); 
  bool writePrototextFile(const char *fn); 

private:

  lbann_proto();
  ~lbann_proto();
  lbann_proto(lbann_proto &) {}
  lbann_proto operator=(lbann_proto&) { return *this; }

  static lbann_proto *s_instance;

  lbann_data::LbannPB m_pb;

  bool m_master;
};

}

#endif //LBANN_PROTO_HPP_INCLUDED

