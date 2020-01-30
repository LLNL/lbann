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

namespace lbann {

class data_coordinator {
 public:
  using data_reader_map_t = std::map<execution_mode, generic_data_reader *>;
  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

 public:
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

 protected:
  std::vector<generic_io_buffer<TensorDataType>*> m_io_buffers;
  io_buffer_map_t m_active_buffer;

  dataset m_training_dataset;
  dataset m_testing_dataset;
  dataset m_validation_dataset;
  //  bool m_data_sets_span_models;

  data_reader_map_t m_data_readers;
 //  std::map<execution_mode, dataset_stats> m_dataset_stats;
  bool m_data_set_processed;
  std::mutex dr_mutex;

};

} // namespace lbann

#endif // LBANN_DATA_COORDINATOR_HPP
