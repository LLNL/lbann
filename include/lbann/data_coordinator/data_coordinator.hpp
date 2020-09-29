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

#ifndef LBANN_DATA_COORDINATOR_HPP
#define LBANN_DATA_COORDINATOR_HPP

#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/utils/dataset.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include <cereal/types/utility.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#ifdef LBANN_HAS_DISTCONV
#include "lbann/data_readers/data_reader_hdf5.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

// Forward-declare trainer
class trainer;

class data_coordinator {
 public:
  using data_reader_map_t = std::map<execution_mode, generic_data_reader *>;
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
  data_coordinator(trainer& trainer, lbann_comm *comm) :
    m_trainer(&trainer),
    m_comm(comm),
    m_data_set_processed(false),
    m_execution_context(nullptr) {}

  ~data_coordinator() {
    // Data coordinator always frees data readers.
    for (auto& dr : m_data_readers) {
      delete dr.second;
    }
  }

  // Data Coordinators copy their data readers.
  data_coordinator(const data_coordinator& other)
    : m_comm(other.m_comm),
      m_training_dataset(other.m_training_dataset),
      m_testing_dataset(other.m_testing_dataset),
      m_validation_dataset(other.m_validation_dataset),
      m_data_readers(other.m_data_readers),
      m_execution_context(other.m_execution_context) {
    for (auto& dr : m_data_readers) {
      dr.second = dr.second ? dr.second->copy() : nullptr;
    }
  }

  data_coordinator& operator=(const data_coordinator& other) {
    for (auto& dr : m_data_readers) {
      dr.second = dr.second ? dr.second->copy() : nullptr;
    }
    return *this;
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize( Archive & ar ) {
    ar(/*CEREAL_NVP(m_io_buffer),*/
       CEREAL_NVP(m_training_dataset),
       CEREAL_NVP(m_testing_dataset),
       CEREAL_NVP(m_validation_dataset)/*,
       CEREAL_NVP(m_data_readers),
       CEREAL_NVP(m_data_set_processed)*/);
  }

  void setup(int max_mini_batch_size, std::map<execution_mode, generic_data_reader *> data_readers);

  /** Check to see if there is a valid training context for the data coordinator */
  bool has_valid_execution_context() const {
    return (m_execution_context != nullptr);
  }

  /** Grab the training context of the data coordinator */
  const execution_context& get_execution_context() const {
    if(m_execution_context == nullptr) {
      LBANN_ERROR("execution context is not set");
    }
    return *m_execution_context;
  }

  /** Grab the training context of the data coordinator */
  execution_context& get_execution_context() {
    return const_cast<execution_context&>(static_cast<const data_coordinator&>(*this).get_execution_context());
  }

  //************************************************************************
  // Helper functions to access the data readers
  //************************************************************************

  generic_data_reader *get_data_reader(const execution_mode mode) const {
    generic_data_reader *data_reader = nullptr;

    auto it = m_data_readers.find(mode);
    if (it != m_data_readers.end()) data_reader = it->second;

    switch(mode) {
    case execution_mode::training:
      break;
    case execution_mode::validation:
      break;
    case execution_mode::testing:
      break;
    default:
      LBANN_ERROR("generic data distribution: invalid execution phase");
    }
    return data_reader;
  }

  /**
   * Get the dimensions of the underlying data.
   */
  TargetModeDimMap get_data_dims() {
    TargetModeDimMap map;
    generic_data_reader *dr;
    for(execution_mode mode : execution_mode_iterator()) {
      dr = get_data_reader(mode);
      if (dr != nullptr) {
        map[data_reader_target_mode::INPUT] = dr->get_data_dims();
        map[data_reader_target_mode::CLASSIFICATION] = std::vector<int>(1, dr->get_num_labels());
        map[data_reader_target_mode::REGRESSION] = std::vector<int>(1, dr->get_num_responses());
        map[data_reader_target_mode::RECONSTRUCTION] = dr->get_data_dims();
        map[data_reader_target_mode::LABEL_RECONSTRUCTION] = dr->get_data_dims();
        map[data_reader_target_mode::NA] = std::vector<int>(1, 0);
        return map;
      }
    }
    LBANN_ERROR("get_data_dims: no available data readers");
    return {};
  }

  /**
   * Get the dimensions of the underlying data.
   */
  SPModeSlicePoints get_slice_points() {
    SPModeSlicePoints map;
    generic_data_reader *dr;
    for(execution_mode mode : execution_mode_iterator()) {
      dr = get_data_reader(mode);
      if (dr != nullptr) {
        for(slice_points_mode sp_mode : slice_points_mode_iterator()) {
          bool is_supported;
          std::vector<El::Int> tmp = dr->get_slice_points(sp_mode, is_supported);
          if(is_supported) {
            map[sp_mode] = tmp;
          }
        }
        return map;
      }
    }
    LBANN_ERROR("get_data_dims: no available data readers");
    return {};
  }

  DataReaderMetaData get_dr_metadata() {
    DataReaderMetaData drm;
    drm.data_dims = get_data_dims();
    drm.slice_points = get_slice_points();
#ifdef LBANN_HAS_DISTCONV
    const auto training_dr = m_data_readers[execution_mode::training];
    drm.shuffle_required = training_dr->is_tensor_shuffle_required();
#endif // LBANN_HAS_DISTCONV
    return drm;
  }

  // At the start of the epoch, set the execution mode and make sure
  // that each layer points to this model
  void reset_mode(execution_context& context) {
    m_execution_context = static_cast<observer_ptr<execution_context>>(&context);
  }

  //************************************************************************
  // Helper functions to access the dataset statistics
  //************************************************************************
  dataset& get_dataset(execution_mode m) {
    switch(m) {
    case execution_mode::training:
      return m_training_dataset;
      break;
    case execution_mode::validation:
      return m_validation_dataset;
      break;
    case execution_mode::testing:
      return m_testing_dataset;
      break;
    default:
      LBANN_ERROR("get_dataset: invalid execution mode");
    }
  }

  const dataset& get_dataset(execution_mode m) const {
    switch(m) {
    case execution_mode::training:
      return m_training_dataset;
      break;
    case execution_mode::validation:
      return m_validation_dataset;
      break;
    case execution_mode::testing:
      return m_testing_dataset;
      break;
    default:
       LBANN_ERROR("get_dataset: invalid execution mode");
    }
  }

  /**
   * Return the first dataset with a valid (non-null) datareader.
   * Returns null if none are valid.
   */
  dataset* select_first_valid_dataset() {
    if (m_data_readers[execution_mode::training]) {
      return &m_training_dataset;
    } else if (m_data_readers[execution_mode::validation]) {
      return &m_validation_dataset;
    } else if (m_data_readers[execution_mode::testing]) {
      return &m_testing_dataset;
    } else {
      return nullptr;
    }
  }

  long get_num_samples_trained() const {
    return m_training_dataset.get_num_samples_processed();
  }
  long get_num_samples_tested() const {
    return m_testing_dataset.get_num_samples_processed();
  }
  long get_total_num_training_samples() const {
    return m_training_dataset.get_total_samples();
  }
  long get_total_num_testing_samples() const {
    return m_testing_dataset.get_total_samples();
  }

  //************************************************************************
  //
  //************************************************************************

  void calculate_num_iterations_per_epoch(int max_mini_batch_size, generic_data_reader *data_reader);
  void calculate_num_iterations_per_epoch(int mini_batch_size);

  int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers) const;
  static int compute_max_num_parallel_readers(long data_set_size, int mini_batch_size, int requested_num_parallel_readers, const lbann_comm* comm);

  //************************************************************************
  //
  //************************************************************************

  // save state of IO to a checkpoint
  bool save_to_checkpoint_shared(persist& p) const {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, execution_mode::training);
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, execution_mode::testing);
      }
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_shared(p, execution_mode::validation);
      }

      if (this->m_comm->am_trainer_master()) {
        write_cereal_archive<const data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
      }
    }
    return true;
  }

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint){

      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->load_from_checkpoint_shared(p, execution_mode::training);
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->load_from_checkpoint_shared(p, execution_mode::testing);
      }
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->load_from_checkpoint_shared(p, execution_mode::validation);
      }

      std::string buf;
      if (this->m_comm->am_trainer_master()) {
        read_cereal_archive<data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
        buf = create_cereal_archive_binary_string<data_coordinator>(*this);
      }

      // TODO: this assumes homogeneous processors
      // broadcast state from rank 0
      this->m_comm->trainer_broadcast(0, buf);

      if (!this->m_comm->am_trainer_master()) {
        unpack_cereal_archive_binary_string<data_coordinator>(*this, buf);
      }
    }

    return true;
  }

  bool save_to_checkpoint_distributed(persist& p) const {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    if(p.get_cb_type() == callback_type::execution_context_only
       || p.get_cb_type() == callback_type::full_checkpoint) {

      it = this->m_data_readers.find(execution_mode::training);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, execution_mode::training);
      }
      it = this->m_data_readers.find(execution_mode::testing);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, execution_mode::testing);
      }
      it = this->m_data_readers.find(execution_mode::validation);
      if ((it != this->m_data_readers.end()) && it->second) {
        (it->second)->save_to_checkpoint_distributed(p, execution_mode::validation);
      }

      write_cereal_archive<const data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
    }
    return true;
  }

  bool load_from_checkpoint_distributed(persist& p) {
    // save state of data readers from input layer
    data_reader_map_t::const_iterator it;
    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_distributed(p, execution_mode::validation);
    }

    read_cereal_archive<data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
    return true;
  }

 protected:
  /** Pointer to hosting trainer */
  trainer *m_trainer;
  /** Pointer to LBANN communicator. */
  lbann_comm *m_comm;

  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;

  data_reader_map_t m_data_readers;
 //  std::map<execution_mode, dataset_stats> m_dataset_stats;
public:  // @todo BVE FIXME
  bool m_data_set_processed;
  std::mutex dr_mutex;

  /** Pointer to the execution context object used for training or evaluating this model */
  observer_ptr<execution_context> m_execution_context;
};

} // namespace lbann

#endif // LBANN_DATA_COORDINATOR_HPP
