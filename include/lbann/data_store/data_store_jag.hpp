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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_CSV_HPP__
#define __DATA_STORE_CSV_HPP__

#include "lbann/data_store/generic_data_store.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include "conduit/conduit_relay_mpi.hpp"
#include <unordered_map>

namespace lbann {


class data_reader_jag_conduit_hdf5;
typedef data_reader_jag_conduit_hdf5 conduit_reader;

//class data_reader_jag_conduit;
//typedef data_reader_jag_conduit conduit_reader;


class data_store_jag : public generic_data_store {
 public:

  //! ctor
  data_store_jag(generic_data_reader *reader, model *m);

  //! copy ctor
  data_store_jag(const data_store_jag&) = default;

  //! operator=
  data_store_jag& operator=(const data_store_jag&) = default;

  data_store_jag * copy() const override { return new data_store_jag(*this); }

  //! dtor
  ~data_store_jag() override;

  void setup() override;

protected :

  conduit_reader *m_jag_reader;

  /// buffers for data that will be passed to the data reader's fetch_datum method
  std::unordered_map<int, std::vector<DataType>> m_my_minibatch_data;

  /// retrive data needed for passing to the data reader for the next epoch
  void exchange_data() override;
  /// returns, in "indices," the set of indices that processor "p"
  /// needs for the next epoch. Called by exchange_data
  void get_indices(std::unordered_set<int> &indices, int p);

  std::map<int, conduit::Node> m_data_index;
  std::vector<conduit::Node> m_data;

  /// fills in m_data (the data store)
  void populate_datastore();

  /// these hold the names of the dependent and independant variables
  /// that we're using
  std::vector<std::string> m_inputs_to_use;
  std::vector<std::string> m_scalars_to_use;
  std::vector<std::string> m_image_views_to_use;
  std::vector<int> m_image_channels_to_use;

  /// these fill in the above four variables;
  /// they are called by load_variable_names()
  void load_inputs_to_use(const std::string &keys);
  void load_scalars_to_use(const std::string &keys);
  void load_image_views_to_use(const std::string &keys);
  void load_image_channels_to_use(const std::string &keys);

  void load_variable_names();
};

}  // namespace lbann

#endif  // __DATA_STORE_CSV_HPP__
