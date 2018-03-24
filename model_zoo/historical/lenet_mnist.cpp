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

#include "lbann/lbann.hpp"

using namespace lbann;

/** Main function. */
int main(int argc, char *argv[]) {
  lbann::lbann_comm *comm = lbann::initialize(argc, argv, 42);
  try {

    ///////////////////////////////////////////////////////////////////
    // Parse experiment parameters
    ///////////////////////////////////////////////////////////////////

    // Data set parameters
    const std::string data_dir = El::Input("--data-dir",
                                           "MNIST data set directory",
                                           std::string("/p/lscratchf/brainusr/datasets/MNIST/"));
    const std::string train_label_file = El::Input("--train-label-file",
                                                   "MNIST training set label file",
                                                   std::string("train-labels-idx1-ubyte"));
    const std::string train_image_file = El::Input("--train-image-file",
                                                   "MNIST training set image file",
                                                   std::string("train-images-idx3-ubyte"));
    const std::string test_label_file = El::Input("--test-label-file",
                                                  "MNIST test set label file",
                                                  std::string("t10k-labels-idx1-ubyte"));
    const std::string test_image_file = El::Input("--test-image-file",
                                                  "MNIST test set image file",
                                                  std::string("t10k-images-idx3-ubyte"));

    // Hardware parameters
    const int procs_per_model = El::Input("--procs-per-model",
                                          "MPI processes per LBANN model",
                                          0);
    const bool using_gpus = El::Input("--using-gpus",
                                      "whether to use GPUs if available",
                                      true);

    // Model parameters
    const int mini_batch_size = El::Input("--mini-batch-size",
                                          "mini-batch size",
                                          128);
    const int num_epochs = El::Input("--num-epochs",
                                     "number of training epochs",
                                     20);
    const double learning_rate = 0.01;

    // Parse command-line arguments
    El::ProcessInput();
    El::PrintInputReport();

    ///////////////////////////////////////////////////////////////////
    // Initialize communicator and cuDNN manager
    ///////////////////////////////////////////////////////////////////

    // Set up the model communicators
    comm->split_models(procs_per_model);
    const bool is_world_master = comm->am_world_master();
    if (is_world_master) {
      const auto& grid = comm->get_model_grid();
      std::cout << "Number of models: " << comm->get_num_models()
                << std::endl
                << "MPI process grid per model: "
                << grid.Height() << " x " << grid.Width() << std::endl;
    }

    // Initialize cuDNN manager
    lbann::cudnn::cudnn_manager *cudnn = nullptr;
    if (using_gpus) {
#ifdef LBANN_HAS_CUDNN
      cudnn = new lbann::cudnn::cudnn_manager(comm);
#endif // LBANN_HAS_CUDNN
    }
    if (is_world_master) {
      std::cout << "cuDNN: "
                << ((cudnn != nullptr) ? "enabled" : "disabled")
                << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // Load data sets for training, validation, and testing
    ///////////////////////////////////////////////////////////////////

    // Load training set
    auto* train_set = new lbann::mnist_reader(true);
    train_set->set_file_dir(data_dir);
    train_set->set_data_filename(train_image_file);
    train_set->set_label_filename(train_label_file);
    train_set->set_validation_percent(0.1);
    train_set->load();
    train_set->scale(true);

    // Create validation set from unused training data
    auto* val_set = new lbann::mnist_reader(*train_set);
    val_set->use_unused_index_set();

    // Load testing data
    auto* test_set = new lbann::mnist_reader(true);
    test_set->set_file_dir(data_dir);
    test_set->set_data_filename(test_image_file);
    test_set->set_label_filename(test_label_file);
    test_set->set_use_percent(1.0);
    test_set->load();
    test_set->scale(true);

    // Create map to data readers
    std::map<execution_mode, lbann::generic_data_reader *> data_readers;
    data_readers[execution_mode::training] = train_set;
    data_readers[execution_mode::validation] = val_set;
    data_readers[execution_mode::testing] = test_set;

    // Print data set information
    if (is_world_master) {
      const int train_size = train_set->get_num_data();
      const int val_size = val_set->get_num_data();
      const int test_size = test_set->get_num_data();
      const double train_percent = train_size * 100.0 / (train_size + val_size);
      const double val_percent = val_size * 100.0 / (train_size + val_size);
      std::cout << "Training data set has "
                << train_size + val_size << " samples: "
                << train_size << " for training "
                << "(" << train_percent << "%), "
                << val_size << " for validation "
                << "(" << val_percent << "%)"
                << std::endl
                << "Testing data set has " << test_size << " samples"
                << std::endl;
    }

    ///////////////////////////////////////////////////////////////////
    // Initialize model
    ///////////////////////////////////////////////////////////////////

    // Initialize objective function
    auto* obj = new lbann::objective_function();
    obj->add_term(new lbann::cross_entropy());

    // Initialize default optimizer
    auto* opt = new lbann::sgd(comm, learning_rate, 0.9, false);

    // Construct model
    lbann::model* m = new lbann::sequential_model(comm, mini_batch_size, obj, opt);

    ///////////////////////////////////////////////////////////////////
    // Add model layers
    ///////////////////////////////////////////////////////////////////

    // Add layers
    lbann::generic_input_layer* input;
    {
      auto* l = new lbann::input_layer<lbann::partitioned_io_buffer,
                                       data_layout::DATA_PARALLEL>(
                  comm, procs_per_model, data_readers, true);
      l->set_name("data");
      m->add_layer(l);
      input = l;
    }
    {
      auto* l = new lbann::convolution_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                  comm, 2, 20, 5, 0, 1, true, cudnn);
      l->set_name("conv1");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::pooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                  comm, 2, 2, 0, 2, pool_mode::max, cudnn);
      l->set_name("pool1");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::convolution_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                  comm, 2, 50, 5, 0, 1, true, cudnn);
      l->set_name("conv2");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::pooling_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                  comm, 2, 2, 0, 2, pool_mode::max, cudnn);
      l->set_name("pool2");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                  comm, 500, nullptr, true, cudnn);
      l->set_name("ip1");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::relu_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(comm, cudnn);
      l->set_name("relu1");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::fully_connected_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(
                  comm, 10, nullptr, true, cudnn);
      l->set_name("ip2");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::softmax_layer<data_layout::DATA_PARALLEL, El::Device::CPU>(comm, cudnn);
      l->set_name("prob");
      m->add_layer(l);
    }
    {
      auto* l = new lbann::target_layer<lbann::partitioned_io_buffer,
                                        data_layout::DATA_PARALLEL>(
                  comm, nullptr, procs_per_model, data_readers, false, false);
      l->set_name("target");
      l->set_paired_input_layer(input);
      m->add_layer(l);
    }

    ///////////////////////////////////////////////////////////////////
    // Setup model
    ///////////////////////////////////////////////////////////////////

    // Print layer information
    if (is_world_master) {
      std::cout << "Mini-batch size: " << mini_batch_size << std::endl
                << "Epoch count: " << num_epochs << std::endl
                << "Optimizer: " << opt->get_description() << std::endl
                << std::endl
                << "Layers:" << std::endl;
      for (const auto* l : m->get_layers()) {
        std::cout << "\t" << l->get_description() << std::endl;
      }
      std::cout << std::endl;
    }

    // Add metrics
    m->add_metric(new lbann::categorical_accuracy_metric(comm));

    // Add callbacks
    m->add_callback(new lbann::lbann_callback_print());
    m->add_callback(new lbann::lbann_callback_timer());

    // Setup model
    m->setup();

    ///////////////////////////////////////////////////////////////////
    // Train and evaluate
    ///////////////////////////////////////////////////////////////////

    // Train model on training set
    m->train(num_epochs);

    // Evaluate model on test set
    m->evaluate(execution_mode::testing);

    ///////////////////////////////////////////////////////////////////
    // Clean up
    ///////////////////////////////////////////////////////////////////

    if (m != nullptr) { delete m; }
    if (cudnn != nullptr) { delete cudnn; }

  } catch (lbann_exception& e) {
    // Report LBANN exception
    lbann::lbann_report_exception(e, comm);
  } catch (std::exception& e) {
    // Report Elemental exception
    El::ReportException(e);
  }
  lbann::finalize(comm);
  return 0;
}
