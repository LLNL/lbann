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

#ifndef __DATA_STORE_JAG_HPP__
#define __DATA_STORE_JAG_HPP__

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_store/generic_data_store.hpp"
#include "conduit/conduit_relay_io.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "conduit/conduit_relay_mpi.hpp"
#include <unordered_map>

namespace lbann {

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

  /// returns the conduit node 
  const conduit::Node & get_conduit_node(int data_id, bool any_node = false) const;

  void set_conduit_node(int data_id, conduit::Node &node);

protected :

  /// used to map a shuffled index to the original subscript;
  /// used in set_conduit_node()
  std::unordered_map<int,int> m_unshuffle;

  bool m_super_node;

  /// retrive data needed for passing to the data reader for the next epoch
  void exchange_data() override {
    if (m_super_node) {
      exchange_data_by_super_node();
    } else {
      exchange_data_by_sample();
    }
  }
  void exchange_data_by_super_node();
  void exchange_data_by_sample();

  // when m_super_node = false
  std::unordered_map<int,int> m_index_to_data_id;

  /// contains the Nodes that this processor owns;
  /// maps data_id to conduit::Node
  std::unordered_map<int, conduit::Node> m_data;

  /// This vector contains Nodes that this processor needs for
  /// the current minibatch; this is filled in by exchange_data()
//  std::unordered_map<int, conduit::Node*> m_minibatch_data;
  std::unordered_map<int, conduit::Node> m_minibatch_data;

  /// work space; used in exchange_data
  std::vector<conduit::Node> m_send_buffer;
  std::vector<conduit::Node> m_send_buffer_2;
  std::vector<MPI_Request> m_send_requests;
  std::vector<MPI_Request> m_recv_requests;
  std::vector<MPI_Status> m_status;
  std::vector<conduit::Node> m_recv_buffer;
  std::vector<int> m_outgoing_msg_sizes;
  std::vector<int> m_incoming_msg_sizes;

  /// called by exchange_data
  void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);

  /// fills in m_owner, which maps an index to the owning processor;
  /// fills in m_my_datastore_indices, which is the set of indices that I own
//  void exchange_ds_indices();

  /// fills in m_all_minibatch_indices; m_all_minibatch_indices[j] 
  /// will contain all indices that will be passed to 
  /// data_reader::fetch_datum in one epoch
  //void build_all_minibatch_indices();

  /// fills in m_all_minibatch_indices; m_all_minibatch_indices[j] 
  /// will contain all indices that will be passed to data_reader::fetch_datum
  /// in one epoch. Also fills in m_owner
  void exchange_ds_indices();
};

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT

#endif  // __DATA_STORE_JAG_HPP__
