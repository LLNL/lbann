////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/check_dataset.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_ingestion/coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"

#include <iomanip>
#include <vector>

namespace lbann {
namespace callback {

template <class Archive>
void check_dataset::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_basename),
     CEREAL_NVP(training_set),
     CEREAL_NVP(validation_set),
     CEREAL_NVP(testing_set));
}

void check_dataset::write_specific_proto(lbann_data::Callback& proto) const
{
  proto.mutable_check_dataset();
}

void check_dataset::add_to_set(model* m,
                               Layer* l,
                               int64_t step,
                               std::set<long>& set)
{
  if (!dynamic_cast<input_layer<DataType>*>(l)) {
    return;
  }

  // FIXME (trb 10/03/2023): This is not the "right" fix (which is
  // probably to reconsider this callback entirely, but this is the
  // drop-in replacement for the line that was here
  // (l->get_sample_indices_per_mb(), which just returns 'nullptr'
  // (which is never checked, of course, so the loop below would
  // segfault almost instantly))).
  auto const mode = m->get_execution_context().get_execution_mode();
  El::Matrix<El::Int> const* const indices =
    get_trainer().get_data_coordinator().get_sample_indices_per_mb(mode);

  for (El::Int j = 0; j < indices->Width(); j++) {
    for (El::Int i = 0; i < indices->Height(); i++) {
      El::Int const idx = indices->Get(i, j);
      auto const [_, new_value] = set.insert(idx);
      if (!new_value) {
        LBANN_ERROR("Step ",
                    step,
                    " :: found a duplicate index in being loaded: ",
                    idx);
      }
    }
  }
}

void check_dataset::on_forward_prop_end(model* m, Layer* l)
{
  const auto& c = m->get_execution_context();
  add_to_set(m, l, c.get_step(), training_set);
}

void check_dataset::on_evaluate_forward_prop_end(model* m, Layer* l)
{
  const auto& c = m->get_execution_context();
  switch (c.get_execution_mode()) {
  case execution_mode::validation:
    add_to_set(m, l, c.get_step(), validation_set);
    break;
  case execution_mode::testing:
    add_to_set(m, l, c.get_step(), testing_set);
    break;
  default:
    LBANN_ERROR("check_dataset: invalid execution phase");
  }
}

void check_dataset::on_epoch_end(model* m)
{
  lbann_comm* comm = m->get_comm();
  std::cout << "Training [" << comm->get_rank_in_trainer()
            << "] : I have processed " << training_set.size() << " elements"
            << std::endl;

  // Get first input layer in model
  input_layer<DataType>* input = nullptr;
  for (auto&& l : m->get_layers()) {
    input = dynamic_cast<input_layer<DataType>*>(l);
    if (input != nullptr) {
      break;
    }
  }
  if (input == nullptr) {
    LBANN_ERROR("could not get input layer");
  }

  int num_samples = training_set.size();
  std::vector<int> vec_num_samples(comm->get_procs_per_trainer());
  if (comm->am_trainer_master()) {
    comm->trainer_gather(num_samples, vec_num_samples.data());
  }
  else {
    comm->trainer_gather(num_samples, comm->get_trainer_master());
  }
  std::vector<int> sample_offsets(comm->get_procs_per_trainer());
  std::partial_sum(vec_num_samples.begin(),
                   vec_num_samples.end(),
                   sample_offsets.begin());
  std::cout << "Training [" << comm->get_rank_in_trainer() << "] offsets";
  for (const auto& idx : sample_offsets) {
    std::cout << idx << " ";
  }
  std::cout << std::endl;
  std::cout << "Training [" << comm->get_rank_in_trainer() << "] counts";
  for (const auto& idx : vec_num_samples) {
    std::cout << idx << " ";
  }
  std::cout << std::endl;

  // sample_offset[]
  // for (int i = 0; i < vec_num_samples.size(); i++) {
  //   //  for (const auto& idx : vec_num_samples) {

  // }

  // Build a vector large enough to hold all the data indices for this rank.
  std::vector<int> local_data(training_set.size());
  std::copy(training_set.begin(), training_set.end(), local_data.data());

  std::cout << "Training: my local vector has size " << local_data.size()
            << std::endl;
  if (comm->am_trainer_master()) {
    data_coordinator& dc = get_trainer().get_data_coordinator();
    // Build a vector large enough to hold all indices for the model.
    std::vector<int> model_training_set(
      dc.get_num_iterations_per_epoch(execution_mode::training) *
      get_trainer().get_max_mini_batch_size());

    std::cout << "Training: my model vector has size "
              << model_training_set.size() << std::endl;
    // comm->trainer_gatherv(local_data.data(), local_data.size(),
    //                     model_training_set.data(), vec_num_samples.data(),
    //                     sample_offsets.data());

    std::cout << "Training: The entire model has processed "
              << model_training_set.size() << " elements" << std::endl;
  }
  else {
    // comm->trainer_gatherv(local_data.data(), local_data.size(),
    //                     m->get_comm()->get_trainer_master());
  }

  std::cout << "Training [" << comm->get_rank_in_trainer() << "] ";
  for (const auto& idx : training_set) {
    std::cout << idx << " ";
  }
  std::cout << std::endl;

  training_set.clear();
}

void check_dataset::on_validation_end(model* m)
{
  std::cout << "Validation [" << m->get_comm()->get_rank_in_trainer()
            << "] : I have processed " << validation_set.size() << " elements"
            << std::endl;
#if 0
  std::cout << "Validation [" << m->get_comm()->get_rank_in_trainer() << "] ";
  for(std::set<long>::iterator iter=validation_set.begin(); iter!=validation_set.end();++iter) {
    std::cout << *iter << " ";
  }
  std::cout << std::endl;
#endif
  validation_set.clear();
}

void check_dataset::on_test_end(model* m)
{
  std::cout << "Testing [" << m->get_comm()->get_rank_in_trainer()
            << "] : I have processed " << testing_set.size() << " elements"
            << std::endl;
  testing_set.clear();
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::check_dataset
#define LBANN_CLASS_LIBNAME callback_check_dataset
#include <lbann/macros/register_class_with_cereal.hpp>
