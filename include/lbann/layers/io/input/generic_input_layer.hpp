////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED

#include "lbann/layers/io/io_layer.hpp"
//#include "lbann/utils/dataset.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/data_coordinator/buffered_data_coordinator.hpp"
#include "lbann/models/model.hpp"
#include "lbann/callbacks/imcomm.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include <cereal/types/utility.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

#include <future>

namespace lbann {

/** @todo Move functionality to input_layer. */
template <typename TensorDataType>
class generic_input_layer : public io_layer<TensorDataType> {
 public:
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
  generic_input_layer(lbann_comm *comm,
                      data_reader_target_mode dr_mode = data_reader_target_mode::CLASSIFICATION)
    : io_layer<TensorDataType>(comm, dr_mode) {
    // Input layers have no parents
    this->m_expected_num_parent_layers = 0;
    if(dr_mode == data_reader_target_mode::NA) {
      this->m_expected_num_child_layers = 1;
    }else {
      // Input layers output a sample and target, which could be the
      // original value, categorical label, or regression value
      this->m_expected_num_child_layers = 2;
    }
  }

  ~generic_input_layer() override {

    // Synchronize the I/O thread pool
    // Note: The thread pool may still be running asynchronously if the
    // trainer is destroyed in the middle of an epoch. The thread pool
    // needs to interact with data readers, etc., so it needs to be
    // synchronized before any of them are destroyed.
    if (this->m_model != nullptr) {
      if (this->m_model->has_valid_execution_context()) {
        this->m_model->get_execution_context().get_io_thread_pool().reap_threads();
      }
    }
  }

  // Input layers copy their datareaders.
  generic_input_layer(const generic_input_layer& other)
    : io_layer<TensorDataType>(other) {
  }

  generic_input_layer& operator=(const generic_input_layer& other) {
    io_layer<TensorDataType>::operator=(other);
    return *this;
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize( Archive & ar ) {
    // ar(CEREAL_NVP(m_io_buffer));
  }

  std::string get_type() const override { return "generic_input"; }

  description get_description() const override {
    auto desc = io_layer<TensorDataType>::get_description();
    return desc;
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    io_layer<TensorDataType>::setup_dims(dr_metadata);
    for (int i = 0; i < this->get_num_children(); ++i) {
      this->set_output_dims(get_data_dims(dr_metadata, i), i);
    }
  }

  void setup_data(size_t max_mini_batch_size) override {
    io_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Resize output to maximum mini-batch size
    for (int i = 0; i < this->get_num_children(); ++i) {
      auto& output = this->get_activations(i);
      output.Resize(output.Height(), max_mini_batch_size);
    }
  }

  /** Setup output tensors.
   *  Sets up the effective (global) mini-batch size.
   */
  void fp_setup_outputs(El::Int mini_batch_size) override {
    /// During model setup there is no valid execution context, but
    /// during execution there is a context
    if(this->m_model->has_valid_execution_context()) {
      auto& c = static_cast<sgd_execution_context&>(this->m_model->get_execution_context());
      auto mode = c.get_execution_mode();
      data_coordinator& dc = c.get_trainer().get_data_coordinator();
      // Determine model mini-batch size and effective mini-batch size
      // Note: If inter-model communication is activated, the effective
      // mini-batch is equal to the global mini-batch size.
      /// @todo This functionality should probably be moved elsewhere
      mini_batch_size = dc.get_current_mini_batch_size(mode);

      auto effective_mini_batch_size = mini_batch_size;
      for (auto&& cb : this->m_model->get_callbacks()) {
        if (dynamic_cast<callback::imcomm*>(cb) != nullptr) {
          effective_mini_batch_size = dc.get_current_global_mini_batch_size(mode);
          break;
        }
      }

      // Set mini-batch size in model
      c.set_current_mini_batch_size(mini_batch_size);
      c.set_effective_mini_batch_size(effective_mini_batch_size);
    }

    // Initialize matrices
    io_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);
  }

  void fp_compute() override {
    execution_mode mode = this->m_model->get_execution_context().get_execution_mode();
    buffered_data_coordinator<TensorDataType>& dc = static_cast<buffered_data_coordinator<TensorDataType>&>(this->m_model->get_execution_context().get_trainer().get_data_coordinator());

    partitioned_io_buffer<TensorDataType>* io_buffer = dc.get_active_buffer(mode);
    // generic_io_buffer<TensorDataType>* io_buffer = dc.m_io_buffers[dc.get_active_buffer_idx(mode) % dc.m_io_buffers.size()];

    // if(dynamic_cast<partitioned_io_buffer<TensorDataType>*>(io_buffer) != nullptr) {
      // Use the predetermined size of the mini-batch to set the current
      // batch size for the neural network
      int num_samples_in_batch = dc.get_current_mini_batch_size(mode);

      dc.update_num_samples_processed(mode, num_samples_in_batch);
      if(this->m_expected_num_child_layers == 1) {
        io_buffer->distribute_from_local_matrix(dc.get_data_reader(mode), mode, this->get_activations(0));
      }else {
        io_buffer->distribute_from_local_matrix(dc.get_data_reader(mode), mode, this->get_activations(0), this->get_activations(1));
      }
    // }else {
    //   LBANN_ERROR("could not fp_compute for I/O layers : encoutered generic_io_buffer type");
    // }

  }

  /**
   * Get the dimensions of the underlying data.
   */
  std::vector<int> get_data_dims(DataReaderMetaData& dr_metadata, int child_index = 0) const override {
    if(child_index == 0) {
      return dr_metadata.data_dims[data_reader_target_mode::INPUT];
    }else if(child_index == 1) {
      return dr_metadata.data_dims[this->m_data_reader_mode];
    }else {
      LBANN_ERROR("get_data_dims: Invalid child index");
    }
    return std::vector<int>(1, 0);
  }

  //************************************************************************
  //
  //************************************************************************

  // save state of IO to a checkpoint
  bool save_to_checkpoint_shared(persist& p) const override {
    // save state of data readers from input layer
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      this->m_model->get_execution_context().get_trainer().get_data_coordinator().save_to_checkpoint_shared(p);

      if (this->get_comm()->am_trainer_master()) {
        write_cereal_archive<const generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
      }

    }
    return true;
  }

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) override {
    // save state of the input layer
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      std::string buf;
      if (this->get_comm()->am_trainer_master()) {
        read_cereal_archive<generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
        buf = create_cereal_archive_binary_string<generic_input_layer>(*this);
      }

      // TODO: this assumes homogeneous processors
      // broadcast state from rank 0
      this->get_comm()->trainer_broadcast(0, buf);

      if (!this->get_comm()->am_trainer_master()) {
        unpack_cereal_archive_binary_string<generic_input_layer>(*this, buf);
      }

    }
    return true;
  }

  bool save_to_checkpoint_distributed(persist& p) const override {
    // save state of data readers from input layer
    if(p.get_cb_type() == callback_type::execution_context_only || p.get_cb_type() == callback_type::full_checkpoint) {
      this->m_model->get_execution_context().get_trainer().get_data_coordinator().save_to_checkpoint_distributed(p);

      write_cereal_archive<const generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
    }
    return true;
  }

  bool load_from_checkpoint_distributed(persist& p) override {
    // load state of data readers for input layer

    this->m_model->get_execution_context().get_trainer().get_data_coordinator().load_from_checkpoint_distributed(p);

    read_cereal_archive<generic_input_layer>(*this, p, execution_mode::training, "_io.xml");
    return true;
  }

 protected:
};

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_INPUT_LAYER_HPP_INCLUDED
