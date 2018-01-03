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

#include "lbann/params.hpp"

lbann::TrainingParams::TrainingParams()
  : EnableProfiling(false), RandomSeed(1), ShuffleTrainingData(1),
    PercentageTrainingSamples(1.00), PercentageValidationSamples(1.00),
    PercentageTestingSamples(1.00), TestWithTrainData(0),
    TrainingSamples(1024), TestingSamples(256),
    EpochStart(0), EpochCount(2), MBSize(192),
    LearnRate(0.3), LearnRateMethod(2),
    LrDecayRate(0.5), LrDecayCycles(5000),
    ActivationType(1), DropOut(-1), Lambda(0),
    DatasetRootDir("."), SaveImageDir("."), ParameterDir("."),
    SaveModel(false), LoadModel(false),
    CkptEpochs(0), CkptSteps(0), CkptSecs(0.0),
    TrainFile(" "), TestFile(" "), SummaryDir("."), DumpWeights(false), DumpActivations(false),
    DumpGradients(false), DumpDir("."), IntermodelCommMethod(0),
    ProcsPerModel(0) {
}

void lbann::TrainingParams::parse_params() {

  EnableProfiling = El::Input("--profiling", "Enable profiling", EnableProfiling);

  RandomSeed = El::Input("--seed", "Random seed", RandomSeed);
  ShuffleTrainingData = El::Input("--random-training-samples", "0 - Pick first N training samples, 1 - Select N random training samples", ShuffleTrainingData);

  PercentageTrainingSamples = El::Input("--percentage-training-samples", "Percentage of training set sampled during training [0.00 to 1.00]", PercentageTrainingSamples);
  PercentageValidationSamples = El::Input("--percentage-validation-samples", "Percentage of the unused training set sampled during validation [0.00 to 1.00]", PercentageValidationSamples);
  PercentageTestingSamples = El::Input("--percentage-testing-samples", "Percentage of testing set sampled during testing", PercentageTestingSamples);
  TestWithTrainData = El::Input("--test-with-train-data", "Use the training data for validation", TestWithTrainData);

  EpochCount = El::Input("--num-epochs", "# of training epochs", EpochCount);
  MBSize = El::Input("--mb-size", "Size of the mini-batch to be trained", MBSize);

  TrainingSamples = El::Input("--training-samples", "# of samples to use in training", TrainingSamples);
  TestingSamples = El::Input("--testing-samples", "# of samples to use in testing", TestingSamples);

  LearnRate = El::Input("--learning-rate", "How much of the gradient update is applied to the weight matrix", LearnRate);
  LearnRateMethod = El::Input("--learning-rate-method", "1 - Adagrad, 2 - RMSprop, 3 - Adam", LearnRateMethod);
  LrDecayRate = El::Input("--lr-decay-rate", "How much does the learning rate decay when it decays", LrDecayRate);
  LrDecayCycles = El::Input("--lr-decay-cycle", "How often does the learning rate decay", LrDecayCycles);
  ActivationType = El::Input("--activation-type", "1 - Sigmoid, 2 - Tanh, 3 - reLU, 4 - id", ActivationType);
  DropOut = El::Input("--drop-out", "% dropout", DropOut);
  Lambda = El::Input("--lambda", "Lambda for L2 Regularization", Lambda);

  DatasetRootDir = El::Input("--dataset", "Location of train and test data", DatasetRootDir);
  TrainFile = El::Input("--train-file", "Train data file",TrainFile);
  TestFile = El::Input("--test-file", "Test data file", TestFile);

  SaveImageDir = El::Input("--output", "Location to save output images", SaveImageDir);
  ParameterDir = El::Input("--params", "Location to save model parameters", ParameterDir);
  SaveModel = El::Input("--save-model", "Save the current model", SaveModel);
  LoadModel = El::Input("--load-model", "Load a saved model", LoadModel);

  CkptEpochs = El::Input("--ckpt-epochs", "Number of training epochs between checkpoints", CkptEpochs);
  CkptSteps  = El::Input("--ckpt-steps", "Number of training steps between checkpoints", CkptSteps);
  CkptSecs   = El::Input("--ckpt-secs", "Number of seconds between checkpoints", CkptSecs);

  SummaryDir = El::Input("--summary-dir", "Directory to write summary files", SummaryDir);
  DumpWeights = El::Input("--dump-weights", "Whether to dump weights", DumpWeights);
  DumpActivations = El::Input("--dump-activations", "Whether to dump weights", DumpActivations);
  DumpGradients = El::Input("--dump-gradients", "Whether to dump gradients", DumpGradients);
  DumpDir = El::Input("--dump-dir", "Directory to dump matrices", DumpDir);

  IntermodelCommMethod = El::Input("--imcomm", "Type of inter-model communication",
                               IntermodelCommMethod);
  ProcsPerModel = El::Input("--procs-per-model",
                        "Number of processes per model (0 = one model)",
                        ProcsPerModel);
}

lbann::PerformanceParams::PerformanceParams() : BlockSize(256), MaxParIOSize(0) {}

void lbann::PerformanceParams::parse_params() {
  BlockSize = El::Input("--block-size", "libElemental Block Size", BlockSize);
  MaxParIOSize = El::Input("--par-IO", "Maximum parallel I/O size (0 - unlimited)", MaxParIOSize);
}

lbann::NetworkParams::NetworkParams() : NetworkStr("1000") {
  parse_network_string();
}

void lbann::NetworkParams::parse_params() {
  NetworkStr = El::Input("--network", "Specify the hidden layers of the topology", NetworkStr);
  parse_network_string();
}

void lbann::NetworkParams::parse_network_string() {
  Network.clear();
  const std::string delim = ",";
  size_t start = 0U;
  size_t end = NetworkStr.find(delim);
  while(end != std::string::npos) {
    Network.push_back(std::stoi(NetworkStr.substr(start, end - start), nullptr, 0));
    start = end + delim.length();
    end = NetworkStr.find(delim, start);
  }
  Network.push_back(std::stoi(NetworkStr.substr(start, end), nullptr, 0));
}

lbann::SystemParams::SystemParams()
  : HostName("Unknwn"), NumNodes(-1), NumCores(-1), TasksPerNode(-1) {}

void lbann::SystemParams::parse_params() {
  HostName = El::Input("--hostname", "HPC hostname", HostName);
  NumNodes = El::Input("--num-nodes", "Total allocation size", NumNodes);
  NumCores = El::Input("--num-cores", "Number of cores per node", NumCores);
  TasksPerNode = El::Input("--tasks-per-node", "MPI Tasks allowed per node", TasksPerNode);
}
