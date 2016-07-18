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

#include "lbann/lbann_params.hpp"

using namespace std;
using namespace El;

lbann::TrainingParams::TrainingParams(void)
  : EnableProfiling(false), RandomSeed(1), ShuffleTrainingData(1),
    MaxTrainingSamples(-1), MaxValidationSamples(-1),
    MaxTestSamples(-1), TestWithTrainData(0),
    EpochCount(2), MBSize(192),
    LearnRate(0.3), LearnRateMethod(2),
    LrDecayRate(0.5), LrDecayCycles(5000),
    ActivationType(activation_type::SIGMOID), DropOut(-1), Lambda(0),
    DatasetRootDir("."), SaveImageDir("."), ParameterDir("."),
    SaveModel(false), LoadModel(false), Checkpoint(10), TrainFile(" "), TestFile(" "){

  MaxMBCount = MaxTrainingSamples > 0 ? ceil((double)MaxTrainingSamples / (double)MBSize) : 0;
}

void lbann::TrainingParams::parse_params(void) {

  EnableProfiling = Input("--profiling", "Enable profiling", EnableProfiling);

  RandomSeed = Input("--seed", "Random seed", RandomSeed);
  ShuffleTrainingData = Input("--random-training-samples", "0 - Pick first N training samples, 1 - Select N random training samples", ShuffleTrainingData);

  MaxTrainingSamples = Input("--max-training-samples", "Max # of samples used for training", MaxTrainingSamples);
  MaxValidationSamples = Input("--max-validation-samples", "Max # of samples used for validation", MaxValidationSamples);
  MaxTestSamples = Input("--max-test-samples", "Max # of samples used for testing", MaxTestSamples);
  TestWithTrainData = Input("--test-with-train-data", "Use the training data for validation", TestWithTrainData);

  EpochCount = Input("--num-epochs", "# of training epochs", EpochCount);
  MBSize = Input("--mb-size", "Size of the mini-batch to be trained", MBSize);
  MaxMBCount = MaxTrainingSamples > 0 ? ceil((double)MaxTrainingSamples / (double)MBSize) : 0;

  LearnRate = Input("--learning-rate", "How much of the gradient update is applied to the weight matrix", LearnRate);
  LearnRateMethod = Input("--learning-rate-method", "1 - Adagrad, 2 - RMSprop", LearnRateMethod);
  LrDecayRate = Input("--lr-decay-rate", "How much does the learning rate decay when it decays", LrDecayRate);
  LrDecayCycles = Input("--lr-decay-cycle", "How often does the learning rate decay", LrDecayCycles);
  ActivationType = static_cast<activation_type>(Input("--activation-type", "1 - Sigmoid, 2 - Tanh, 3 - reLU, 4 - id", static_cast<int>(ActivationType)));
  DropOut = Input("--drop-out", "% dropout", DropOut);
  Lambda = Input("--lambda", "Lambda for L2 Regularization", Lambda);

  DatasetRootDir = Input("--dataset", "Location of train and test data", DatasetRootDir);
  TrainFile = Input("--train-file", "Train data file",TrainFile);
  TestFile = Input("--test-file", "Test data file", TestFile);

  SaveImageDir = Input("--output", "Location to save output images", SaveImageDir);
  ParameterDir = Input("--params", "Location to save model parameters", ParameterDir);
  SaveModel = Input("--save-model", "Save the current model", SaveModel);
  LoadModel = Input("--load-model", "Load a saved model", LoadModel);
  Checkpoint = Input("--checkpoint", "Number of training epochs between checkpoints", Checkpoint);

}

lbann::PerformanceParams::PerformanceParams(void) : BlockSize(256), MaxParIOSize(0) {}

void lbann::PerformanceParams::parse_params(void) {
  BlockSize = Input("--block-size", "libElemental Block Size", BlockSize);
  MaxParIOSize = Input("--par-IO", "Maximum parallel I/O size (0 - unlimited)", MaxParIOSize);
}

lbann::NetworkParams::NetworkParams(void) : NetworkStr("1000") {
  parse_network_string();
}

void lbann::NetworkParams::parse_params(void) {
  NetworkStr = Input("--network", "Specify the hidden layers of the topology", NetworkStr);
  parse_network_string();
}

void lbann::NetworkParams::parse_network_string(void) {
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

lbann::SystemParams::SystemParams(void)
  : HostName("Unknwn"), NumNodes(-1), NumCores(-1), TasksPerNode(-1) {}

void lbann::SystemParams::parse_params(void) {
  HostName = Input("--hostname", "HPC hostname", HostName);
  NumNodes = Input("--num-nodes", "Total allocation size", NumNodes);
  NumCores = Input("--num-cores", "Number of cores per node", NumCores);
  TasksPerNode = Input("--tasks-per-node", "MPI Tasks allowed per node", TasksPerNode);
}
