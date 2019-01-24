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

#include "lbann/data_store/data_store_jag.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/file_utils.hpp" // for add_delimiter()

#include <unordered_set>

#undef DEBUG
#define DEBUG

namespace lbann {

data_store_jag::data_store_jag(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m) {
  set_name("data_store_jag");
}

data_store_jag::~data_store_jag() {
}

void data_store_jag::setup() {
  double tm1 = get_time();
  std::stringstream err;

  if (m_master) {
    std::cout << "starting data_store_jag::setup() for role: " 
              << m_reader->get_role() << "\n"
              << "calling generic_data_store::setup()\n";
  }
  generic_data_store::setup();
  m_jag_reader = dynamic_cast<data_reader_jag_conduit*>(m_reader);
  if (m_jag_reader == nullptr) {
    LBANN_ERROR(" dynamic_cast<data_reader_jag_conduit*>(m_reader) failed");
  }

/*
  m_minibatch_data.resize(m_np);
  m_send_buffer.resize(m_np);
  m_send_buffer_2.resize(m_np);
  m_send_requests.resize(m_np);
  m_recv_requests.resize(m_np);
  m_status.resize(m_np);
  m_outgoing_msg_sizes.resize(m_np);
  m_incoming_msg_sizes.resize(m_np);
  m_recv_buffer.resize(m_np);
*/

  // builds map: shuffled_index subscript -> owning proc
//  build_index_owner();

#if 0
  if (! m_in_memory) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "not yet implemented";
  td::unordered_map<int, int> m_owner;
  throw lbann_exception(err.str());
  } 
  
  else {
    //sanity check
    conduit_reader *reader = dynamic_cast<conduit_reader*>(m_reader);
    if (reader == nullptr) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "dynamic_cast<conduit_reader*>(m_reader) failed";
      throw lbann_exception(err.str());
    }
    m_data_reader = reader;

    //load_variable_names();

    if (m_master) std::cout << "calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();
    
    if (m_master) std::cout << "calling exchange_mb_indices()\n";
    exchange_mb_indices();

    if (m_master) std::cout << "calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cout << "calling populate_datastore()\n";
    populate_datastore(); 

    if (m_master) std::cout << "calling exchange_data()\n";
    exchange_data();
    if (m_master) std::cout << "DONE! calling exchange_data()\n";
  }
#endif
  if (m_master) {
    std::cout << "TIME for data_store_jag setup: " << get_time() - tm1 << "\n";
  }
}

#if 0
void data_store_jag::get_indices(std::unordered_set<int> &indices, int p) {
  indices.clear();
  std::vector<int> &v = m_all_minibatch_indices[p];
  for (auto t : v) {
    indices.insert((*m_shuffled_indices)[t]);
  }
}
#endif


void data_store_jag::exchange_data() {
#if 0
  double tm1 = get_time();

  //========================================================================
  //build map: proc -> global indices that P_x needs for this epoch, and
  //                   which I own
  std::vector<std::unordered_set<int>> proc_to_indices(m_np);
  for (size_t p=0; p<m_all_minibatch_indices.size(); p++) {
    for (auto idx : m_all_minibatch_indices[p]) {
      int index = (*m_shuffled_indices)[idx];
      if (m_my_datastore_indices.find(index) != m_my_datastore_indices.end()) {
        proc_to_indices[p].insert(index);
      }
    }
  }

  if (m_master) std::cout << "exchange_data; built map\n";

  //========================================================================
  //part 1: exchange the sizes of the data
  
  for (size_t j=0; j<m_send_buffer.size(); j++) {
    m_send_buffer[j].reset();
  }
  int my_first = m_sample_ownership[m_rank];
  for (int p=0; p<m_np; p++) {
    for (auto idx : proc_to_indices[p]) {
      int local_idx = idx - my_first;
      if (local_idx >= (int)m_data.size()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: local index: " + std::to_string(local_idx) + " is >= m_data.size(): " + std::to_string(m_data.size()));
      }
      if (local_idx < 0) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: local index: " + std::to_string(local_idx) + " is < 0");
      }

      m_send_buffer[p][std::to_string(idx)] = m_data[local_idx];
    }

    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);

    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    MPI_Isend((void*)&m_outgoing_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  //start receives for sizes of the data
  for (int p=0; p<m_np; p++) {
    MPI_Irecv((void*)&m_incoming_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_recv_requests[p]);
  }

  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());

  //========================================================================
  //part 2: exchange the actual data
  
  // start sends for outgoing data
  for (int p=0; p<m_np; p++) {
  if (m_master && p==0) {
  }
    const void *s = m_send_buffer_2[p].data_ptr();
    MPI_Isend(s, m_outgoing_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  m_comm->global_barrier();
  if (m_master) std::cout << "started sends for outgoing data\n";
  m_comm->global_barrier();

  // start recvs for incoming data
  for (int p=0; p<m_np; p++) {
    m_recv_buffer[p].set(conduit::DataType::uint8(m_incoming_msg_sizes[p]));
    MPI_Irecv(m_recv_buffer[p].data_ptr(), m_incoming_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_recv_requests[p]);
  }

  m_comm->global_barrier();
  if (m_master) std::cout << "started recvs for incoming data\n";
  m_comm->global_barrier();


  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());
if (m_master) std::cout << "finished waiting!\n\n";

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  for (int p=0; p<m_np; p++) {
    conduit::uint8 *n_buff_ptr = (conduit::uint8*)m_recv_buffer[p].data_ptr();
    conduit::Node n_msg;
    n_msg["schema_len"].set_external((conduit::int64*)n_buff_ptr);
    n_buff_ptr +=8;
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    conduit::Schema rcv_schema;
    conduit::Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);
    m_minibatch_data[p].update(n_msg["data"]);
  }

  if (m_master) std::cout << "data_store_jag::exchange_data time: " << get_time() - tm1 << "\n";
#endif
}

void data_store_jag::populate_datastore() {
#if 0
  // master reads the index file and bcasts to all others
  int n;
  std::string st;
  if (m_master) {
    const std::string filelist = m_reader->get_data_filename();
    std::ifstream in(filelist.c_str());
    if (!in) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + filelist + " for reading");
    }
    in.seekg(0, std::ios_base::end);
    n = in.tellg();
    in.seekg(0, std::ios_base::beg);
    st.resize(n);
    in.read((char*)st.data(), n);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  st.resize(n);
  MPI_Bcast(&st[0], n, MPI_BYTE, 0, MPI_COMM_WORLD);

  // get num samples, num conduit files, and base data directory
  int num_files;
  std::string base_dir;
  std::stringstream ss(st);
  ss >> m_num_samples >> num_files >> base_dir;
  std::string line;
  getline(ss, line); //discard '\n' at end of 2nd line of file
  
  // build vector to be used to map a sample index to the owning processor;
  // also, cache the lines from the index file that contain data for the
  // files and samples that this processor owns.
  int num_files_per_proc = num_files / m_np;
  int remainder = num_files % m_np;
  m_sample_ownership.reserve(m_np+1); 
  m_sample_ownership.push_back(0);
  std::string fn;
  int tot = 0;
  std::vector<std::string> my_files;
  for (int proc = 0; proc<m_np; proc++) {
    int k = proc < remainder ? num_files_per_proc + 1 : num_files_per_proc;
    for (int hh = 0; hh<k; ++hh) {
      getline(ss, line);
      if (proc == m_rank) {
        my_files.push_back(line);
      }
      std::stringstream s3(line);
      s3 >> fn >> n;
      tot += n;
    }  
    m_sample_ownership.push_back(tot);
  }  
  int my_num_samples = m_sample_ownership[m_rank+1] - m_sample_ownership[m_rank];
  if (m_master) std::cout << "num samples: " << my_num_samples << "\n";

  // load this processor's data
  if (m_master) std::cout << "starting to load data ...\n";
  int my_first = m_sample_ownership[m_rank];
  m_my_datastore_indices.clear();
  m_data.resize(my_num_samples);
  size_t sample_id_count = 0;
  for (size_t j=0; j<my_files.size(); j++) {
    std::stringstream s(my_files[j]);
    int good;
    size_t failed;
    s >> fn >> good >> failed;
    std::unordered_set<int> bad_indices;
    int bad;
    while (s >> bad) {
      bad_indices.insert(bad);
    }
    if (bad_indices.size() != failed) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: num_bad: " + std::to_string(failed) + " num bad_indices: " + std::to_string(bad_indices.size()) + " should be equal!");
    }
    const std::string filename = base_dir + "/" + fn;
if (m_master) std::cout << "opening file #" << j << " of "<< my_files.size() << "\n";
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    conduit::Node tmp;
    for (size_t jj=0; jj<cnames.size(); jj++) {
      if (bad_indices.find(jj) == bad_indices.end()) {
        const std::string key =  cnames[jj] + "/inputs";
        if (sample_id_count > m_data.size()) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: sample_id_count: " + std::to_string(sample_id_count) + " m_data.size(): " + std::to_string(m_data.size()));
        }
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
        m_data[sample_id_count]["/inputs"] = tmp;

        /*
        const std::vector<std::string> &scalar_keys = m_data_reader->get_scalar_choices();
        for (auto scalar_key : scalar_keys) {
          const std::string sc_key = cnames[jj] +  "/outputs/scalars/"  + scalar_key;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, sc_key, tmp);
          m_data[sample_id_count]["outputs/scalars/" + scalar_key] = tmp;
        }
        */

        // read in all scalars; this wastes a bit of memory, but is about twice
        // as fast as looping over each scalar that we're actually interested
        // in; and the amount of wasted memory is small compared to memory
        // required by the images
        const std::string sc_key = cnames[jj] +  "/outputs/scalars/";
        conduit::relay::io::hdf5_read(hdf5_file_hnd, sc_key, tmp);
        m_data[sample_id_count]["outputs/scalars/"] = tmp;

        const std::vector<std::string> &image_keys = m_data_reader->get_image_choices();
        for (auto image_key : image_keys) {
          const std::string img_key  = cnames[j] + "/outputs/images/" + image_key + "/0.0/emi";
          conduit::relay::io::hdf5_read(hdf5_file_hnd, img_key, tmp);
          m_data[sample_id_count]["outputs/images/" + image_key + "/0.0/emi"] = tmp;
        }
        m_my_datastore_indices.insert(sample_id_count + my_first);
        ++sample_id_count;
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
  if (sample_id_count != m_data.size()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: sample_id_count: " + std::to_string(sample_id_count) + " m_data.size(): " + std::to_string(m_data.size()) + " should be equal");
  }  
#endif
}

void data_store_jag::set_conduit_node(int data_id) {
}
  
const conduit::Node & data_store_jag::get_conduit_node(int data_id) const {
#if 0
  //TODO: add error checking
  std::vector<int>::const_iterator it = std::upper_bound(m_sample_ownership.begin(), m_sample_ownership.end(), data_id);
  int idx = (it - m_sample_ownership.begin()) - 1;
  #ifdef DEBUG
  if (m_master) {
    std::cout << "data_id: " << data_id << "  idx: " << idx << "\n";
  }
  #endif
  return m_minibatch_data[idx];
#endif

//this will go away; temp code to get buid working
conduit::Node *n = new conduit::Node;
return *n;
}

void data_store_jag::testme() {
#if 0
  if (m_master) {
    std::cout << "starting testme\n";
    std::vector<int> &mine = m_all_minibatch_indices[m_rank];
    for (auto idx : mine) {
      int s = (*m_shuffled_indices)[idx];
      std::cout << "idx: " << idx << " shuffled: " << s << "\n";
      const conduit::Node &n = get_node(s, 0);
      n.print();
      std::cout << "  is_external? " << n.is_data_external() << "\n";
      std::stringstream key;
      key << "/" << s << "/inputs";
      conduit::Node nd = n[key.str()];
      nd.print();
    }
  }
#endif
}

#if 0
void data_store_jag::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {
  node_out.reset();
  conduit::Schema s_data_compact;
  if( node_in.is_compact() && node_in.is_contiguous()) {
    s_data_compact = node_in.schema();
  } else {
    node_in.schema().compact_to(s_data_compact);
  }

  std::string snd_schema_json = s_data_compact.to_json();
if (m_master)  {
  std::cout << "XXXX:\n";
  std::cout << snd_schema_json << "\n";
  }

  conduit::Schema s_msg;
  s_msg["schema_len"].set(conduit::DataType::int64());
  s_msg["schema"].set(conduit::DataType::char8_str(snd_schema_json.size()+1));
  s_msg["data"].set(s_data_compact);

  conduit::Schema s_msg_compact;
  s_msg.compact_to(s_msg_compact);
  conduit::Node n_msg(s_msg_compact);
  conduit::Node tmp;
  tmp["schema"].set(snd_schema_json);
  tmp["data"].update(node_in);
static bool doit = true;
if (m_master && doit) {
  std::cout << "1. RRRRRRRRRRRR\n";
  conduit::Node n3 = tmp["schema"];
  n3.print();
  std::cout << "WWWWWWWWWWWWWWWWWWWWWWWWWWWWW\n\n";
}
  tmp.compact_to(node_out);
if (m_master && doit) {
  std::cout << "2. RRRRRRRRRRRR\n";
  conduit::Node n3 = node_out["schema"];
  n3.print();
  std::cout << "WWWWWWWWWWWWWWWWWWWWWWWWWWWWW\n\n";
  doit = false;
}
}
#endif

void data_store_jag::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {
#if 0
  conduit::Schema s_data_compact;
  if( node_in.is_compact() && node_in.is_contiguous()) {
    s_data_compact = node_in.schema();
  } else {
    node_in.schema().compact_to(s_data_compact);
  }

  std::string snd_schema_json = s_data_compact.to_json();
/*
if (m_master)  {
  std::cout << "XXXX:\n";
  std::cout << snd_schema_json << "\n";
  }
*/

  conduit::Schema s_msg;
  s_msg["schema_len"].set(conduit::DataType::int64());
  s_msg["schema"].set(conduit::DataType::char8_str(snd_schema_json.size()+1));
  s_msg["data"].set(s_data_compact);

  conduit::Schema s_msg_compact;
  s_msg.compact_to(s_msg_compact);
  //conduit::Node n_msg(s_msg_compact);
  node_out.reset();
  node_out.set(s_msg_compact);
  node_out["schema"].set(snd_schema_json);
  node_out["data"].update(node_in);
/*
static bool doit = true;
if (m_master && doit) {
  std::cout << "1. RRRRRRRRRRRR\n";
  conduit::Node n3 = node_out["schema"];
  n3.print();
  std::cout << "WWWWWWWWWWWWWWWWWWWWWWWWWWWWW\n\n";
}
*/
/*
  node_in.compact_to(node_out);
if (m_master && doit) {
  std::cout << "2. RRRRRRRRRRRR\n";
  conduit::Node n3 = node_out["schema"];
  n3.print();
  std::cout << "WWWWWWWWWWWWWWWWWWWWWWWWWWWWW\n\n";
  doit = false;
}
*/
#endif
}

// fills in m_owner: std::unordered_map<int, int> m_owner
// which maps an index to the processor that owns the associated data
void data_store_jag::build_index_owner() {
#if 0
  //todo: should be performed by P_0 then bcast; for now we'll
  //      have all procs do it
  
  std::stringstream err;

  // get filename for sample list
  m_base_dir = add_delimiter(m_data_reader->get_file_dir());
  m_sample_list_fn = m_base_dir + m_data_reader->get_data_index_list();

  // get filename for mapping file
  m_mapping_fn = m_base_dir + m_data_reader->get_local_file_dir();

  //file_owners[i] contains the counduit filenames that P_i owns
  std::vector<std::vector<std::string>> file_owners(m_np);

  // sample_counts[i][j] contains the number of valid samples
  // in the conduit file: file_owners[[i][j]
  std::vector<std::vector<int>> sample_counts(m_np);

  std::ifstream in(sample_list_file);
  if (!in) {
    err << "failed to open " << sample_list_file << " for reading";
    LBANN_ERROR(err);
  }


  in.close()
#endif
}


}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
