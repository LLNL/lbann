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

#ifndef LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_IO_LAYER_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/dataset.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/exception.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// snprintf
#include <cstdio>

namespace lbann {

/** @todo Move functionality to input_layer. */
template <typename TensorDataType>
class io_layer : public data_type_layer<TensorDataType> {
 protected:
  data_reader_target_mode m_data_reader_mode;

 public:
  io_layer(lbann_comm *comm,
           data_reader_target_mode data_reader_mode = data_reader_target_mode::CLASSIFICATION)
    : data_type_layer<TensorDataType>(comm),
      m_data_reader_mode(data_reader_mode) {
  }

  /**
   * Get the dimensions of the underlying data.
   */
  virtual std::vector<int> get_data_dims(DataReaderMetaData& dr_metadata, int child_index = 0) const = 0;

  bool is_for_regression() const {
    return (m_data_reader_mode == data_reader_target_mode::REGRESSION);
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_IO_LAYER_HPP_INCLUDED
