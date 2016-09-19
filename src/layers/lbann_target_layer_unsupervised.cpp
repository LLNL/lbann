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

#include "lbann/layers/lbann_target_layer_unsupervised.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "lbann/lbann_Elemental_extensions.h"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace El;

lbann::target_layer_unsupervised::target_layer_unsupervised(lbann_comm* comm, int num_parallel_readers, uint mini_batch_size, std::map<execution_mode, DataReader*> data_readers, bool shared_data_reader)
  : target_layer(comm, mini_batch_size, data_readers, shared_data_reader),
    distributed_minibatch_parallel_io(comm, num_parallel_readers, mini_batch_size, data_readers)
{

}

/*lbann::target_layer_unsupervised::target_layer_unsupervised(lbann_comm* comm, input_layer* in_layer)
{
  input_circmat(comm->get_model_grid());
  m_input_layer = in_layer;
}*/

void lbann::target_layer_unsupervised::setup(int num_prev_neurons) {

  NumNeurons = m_input_layer->get_linearized_data_size(); //need inherittance
  Zeros(*Ds_Temp, NumNeurons, Layer::m_mini_batch_size);
}

///@todo update this to use the new fp_linearity framework
DataType lbann::target_layer_unsupervised::forwardProp(DataType prev_WBL2NormSum) {
  input_mat = m_input_layer->get_local_mat();
  //Print(*input_mat);
  int num_errors = 0;
  //Layer::m_mini_batch_size or get input layer num_samples in batch
  target_layer::update_num_samples_processed(input_mat->Width());
  //std::cout << "Input " << input_mat->Width() << input_mat->Height() << std::endl;
  for (int mb_index= 0; mb_index < input_mat->Width(); mb_index++) { /// For each sample in mini-batch
    DataType sum_errors=0.0;
    for (int f_index= 0; f_index < input_mat->Height(); f_index++) {
      //sumerrors += ((X[m][0] - XP[m][0]) * (X[m][0] - XP[m][0]));
      DataType x = input_mat->Get(f_index,mb_index);
      DataType x_bar = fp_input->GetLocal(f_index,mb_index);
      //num_errors += (x-x_bar) * (x-x_bar); //a good metric?
      sum_errors += (x-x_bar) * (x-x_bar);
    }
    num_errors = sum_errors;
  }
  num_errors = Layer::comm->model_allreduce(num_errors);
  return num_errors;
}

void lbann::target_layer_unsupervised::backProp() {
  /// Copy the results to the Ds_Temp variable for access by the next lower layer
  input_circmat = m_input_layer->get_dist_mat();
  Copy(*input_circmat, *Ds_Temp);
}


execution_mode lbann::target_layer_unsupervised::get_execution_mode() {
  return m_execution_mode;
}

void lbann::target_layer_unsupervised::set_input_layer(input_layer_distributed_minibatch_parallel_io* input_layer) {
  m_input_layer = input_layer;
}
