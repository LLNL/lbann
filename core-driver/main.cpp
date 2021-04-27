///////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
//// Produced at the Lawrence Livermore National Laboratory.
//// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
//// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
////
//// LLNL-CODE-697807.
//// All rights reserved.
////
//// This file is part of LBANN: Livermore Big Artificial Neural Network
//// Toolkit. For details, see http://software.llnl.gov/LBANN or
//// https://github.com/LLNL/LBANN.
////
//// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
//// may not use this file except in compliance with the License.  You may
//// obtain a copy of the License at:
////
//// http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//// implied. See the License for the specific language governing
//// permissions and limitations under the license.
///////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include <mpi.h>
#include <stdio.h>

int max_idx(const El::AbstractDistMatrix<float>& data, int row) {
  float max = 0;
  int idx = 0;
  for (int i=0; i<data.Height(); i++) {
    if (data.Get(row, i) > max) {
      max = data.Get(row, i);
      idx = i;
    }
  }
  return idx;
}

std::vector<int>
get_label(lbann::directed_acyclic_graph_model* m, std::string pred_layer) {
  std::vector<int> pred_labels;
  for (const auto* l : m->get_layers()) {
    if (l->get_name() == pred_layer) {
      auto const& dtl = dynamic_cast<lbann::data_type_layer<float> const&>(*l);
      const auto& labels = dtl.get_activations();

      for (int row_idx=0; row_idx<labels.Width(); row_idx++) {
        pred_labels.push_back(max_idx(labels, row_idx));

      }
    }
  }
  return pred_labels;
}

void print_inf(lbann::directed_acyclic_graph_model* m, lbann::lbann_comm* lc) {
  // Get predicted labels
  if (lc->am_world_master()) {
  std::cout << std::endl << "predicted:" << std::endl;
  }
  for (const auto* l : m->get_layers()) {
    if (l->get_type() == "softmax") {
      auto const& dtl = dynamic_cast<lbann::data_type_layer<float> const&>(*l);
      const auto& labels = dtl.get_activations();
      for (int row_idx=0; row_idx<labels.Width(); row_idx++) {
        int label_val = max_idx(labels, row_idx);
        if (lc->am_world_master()) {
          std::cout << label_val << ", ";
        }
      }
      //std::cout << "labels size: " << labels.Height() << ", " << labels.Width() << std::endl;
      //El::Display(labels);
    }
  }

  if (lc->am_world_master()) {
  std::cout << std::endl << "truth:" << std::endl;
  }
  // Get actual labels (just to verify the predictions we're getting)
  for (const auto* l : m->get_layers()) {
    if (l->get_name() == "layer3") {
      auto const& dtli = dynamic_cast<lbann::data_type_layer<float> const&>(*l);
      const auto& t_labels = dtli.get_activations();
      for (int row_idx=0; row_idx<t_labels.Width(); row_idx++) {
        int t_label_val = max_idx(t_labels, row_idx);
        if (lc->am_world_master()) {
          std::cout << t_label_val << ", ";
        }
      }
    }
  }
}

auto mock_dr_metadata() {
  lbann::DataReaderMetaData drmd;
  auto& md_dims = drmd.data_dims;
  md_dims[lbann::data_reader_target_mode::CLASSIFICATION] = {10};
  md_dims[lbann::data_reader_target_mode::INPUT] = {1,28,28};
  return drmd;
}

std::unique_ptr<lbann::model>
load_model(lbann::lbann_comm* lc, std::string cp_dir, int mbs) {
  // Open checkpoint
  lbann::persist p;
  p.open_restart(cp_dir.c_str());

  // Model
  auto m = lbann::make_unique<lbann::directed_acyclic_graph_model>(lc, nullptr, nullptr);

  // Load model from checkpoint
  auto m_flag = m->load_from_checkpoint_shared(p);
  if (lc->am_world_master()) {
    std::cout << "model load: " << m_flag << std::endl;
  }

  // Close checkpoint
  p.close_restart();

  // Setup the model
  auto dr_metadata = mock_dr_metadata();
  m->setup(mbs, dr_metadata);

  /*
  for (const auto* l : m->get_layers()) {
    if (lc->am_world_master()) {
      std::cout << l->get_name() << ": ";
      auto dims = l->get_output_dims(0);
      for (int k=0; k<dims.size(); k++) {
        std::cout << dims[k] << ", ";
      }
      std::cout << std::endl;
    }
  }
  p.close_restart();

  */

  return m;
}

const El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>
load_samples() {
  int h = 128, w = 128, c = 1, N = 64;
  const El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU> samples(c * h * w, N);
  return samples;
}

int main(int argc, char *argv[]) {
  // Input params
  std::string model_dir, sample_dir, input_data_layer, input_label_layer, pred_layer;
  model_dir = "/usr/workspace/wyatt5/cp_models/trainer0/sgd.shared.epoch_begin.epoch.10.step.8440/";
  sample_dir = "/usr/workspace/wyatt5/mnist_data/mnist.csv";
  input_data_layer = "layer1";
  input_label_layer = "layer3";
  pred_layer = "layer15";
  int mbs = 64;

  // Init MPI and verify MPI_THREADED_MULTIPLE
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
  }

  // Setup comms
  auto lbann_comm = lbann::driver_init(MPI_COMM_WORLD);

  int ppt = lbann_comm->get_procs_in_world();
  if (ppt != lbann_comm->get_procs_per_trainer()) {
    lbann_comm->split_trainers(ppt);
  }

  // Load the model
  auto m = load_model(lbann_comm.get(), model_dir, mbs);

  // Load the data
  auto samples = load_samples();
  //const El::Matrix<float, El::Device::CPU> samples;

  // Create inference algorithm
  auto inf_alg = lbann::batch_functional_inference_algorithm();

  // Infer
  auto labels = inf_alg.infer(m.get(), samples, pred_layer, mbs);
  if (lbann_comm->am_world_master()) {
    std::cout << "Predicted Labels: ";
    for (int i=0; i<labels.Height(); i++) {
      std::cout << labels(i) << " ";
    }
    std::cout << std::endl;
  }

  // Clean up
  m.reset();
  lbann::finalize();
  MPI_Finalize();

  return 0;
}
