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
  const conduit::Node & get_conduit_node(int data_id) const;

  void set_conduit_node(int data_id, conduit::Node &node);

protected :

  bool m_super_node;

  /// retrive data needed for passing to the data reader for the next epoch
  /// this is pure virtual in generic_data_reader, so must include it for
  /// now. May go away when we refactore/revise all of data_store
  void exchange_data() override {}

  void exchange_mini_batch_data(size_t current_pos, size_t mb_size) override {
    if (m_super_node) {
      exchange_data_by_super_node(current_pos, mb_size);
    } else {
      exchange_data_by_sample(current_pos, mb_size);
    }
  }
  void exchange_data_by_super_node(size_t current_pos, size_t mb_size);
  void exchange_data_by_sample(size_t current_pos, size_t mb_size);
  void setup_data_store_buffers();

  // when m_super_node = false
  std::unordered_map<int,int> m_index_to_data_id;

  /// contains the Nodes that this processor owns;
  /// maps data_id to conduit::Node
  std::unordered_map<int, conduit::Node> m_data;

  /// This vector contains Nodes that this processor needs for
  /// the current minibatch; this is filled in by exchange_data()
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

  std::vector<conduit::Node> m_reconstituted;

  /// called by exchange_data
  void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);

  /// m_ds_indices[j] contains the sample indices (data store (ds) indices)
  // for the samples that P_j owns
  std::vector<std::unordered_set<int>> m_ds_indices;

  /// fills in m_ds_indices and m_owner
  void build_ds_indices();
};

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT

#endif  // __DATA_STORE_JAG_HPP__
