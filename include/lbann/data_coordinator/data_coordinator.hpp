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

#ifndef LBANN_DATA_COORDINATOR_HPP
#define LBANN_DATA_COORDINATOR_HPP

#include "lbann/utils/dataset.hpp"

namespace lbann {

class data_coordinator {
 public:
  using data_reader_map_t = std::map<execution_mode, generic_data_reader *>;
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
  data_coordinator(std::map<execution_mode, generic_data_reader *> data_readers) :
    m_training_dataset(),
    m_testing_dataset(),
    m_validation_dataset(),
    m_data_readers(data_readers),
    m_data_set_processed(false) {

    if(m_data_readers[execution_mode::training] != nullptr) {
      this->m_training_dataset.total_samples() = m_data_readers[execution_mode::training]->get_num_data();
    }

    if(m_data_readers[execution_mode::validation] != nullptr) {
      this->m_validation_dataset.total_samples() = m_data_readers[execution_mode::validation]->get_num_data();
    }

    if(m_data_readers[execution_mode::testing] != nullptr) {
      this->m_testing_dataset.total_samples() = m_data_readers[execution_mode::testing]->get_num_data();
    }
  }

  ~data_coordinator() {
    // Data coordinator always frees data readers.
    for (auto& dr : m_data_readers) {
      delete dr.second;
    }
  }

  // Data Coordinators copy their data readers.
  data_coordinator(const data_coordinator& other)
    : m_training_dataset(other.m_training_dataset),
      m_testing_dataset(other.m_testing_dataset),
      m_validation_dataset(other.m_validation_dataset),
      m_data_readers(other.m_data_readers) {
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

      if (this->get_comm()->am_trainer_master()) {
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
      if (this->get_comm()->am_trainer_master()) {
        read_cereal_archive<data_coordinator>(*this, p, execution_mode::training, "_dc.xml");
        buf = create_cereal_archive_binary_string<data_coordinator>(*this);
      }

      // TODO: this assumes homogeneous processors
      // broadcast state from rank 0
      this->get_comm()->trainer_broadcast(0, buf);

      if (!this->get_comm()->am_trainer_master()) {
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
  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;
  //  bool m_data_sets_span_models;

  data_reader_map_t m_data_readers;
 //  std::map<execution_mode, dataset_stats> m_dataset_stats;
public:  // @todo BVE FIXME
  bool m_data_set_processed;
  std::mutex dr_mutex;

};

} // namespace lbann

#endif // LBANN_DATA_COORDINATOR_HPP
