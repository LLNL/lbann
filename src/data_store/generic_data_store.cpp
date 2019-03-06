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

#include "lbann/data_store/generic_data_store.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/models/model.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <numeric>
#include <string.h>

namespace lbann {

generic_data_store::generic_data_store(generic_data_reader *reader, model *m) :
m_n(0),
    m_reader(reader),
    m_my_minibatch_indices(nullptr),
    m_in_memory(true),
    m_model(m),
    m_extended_testing(false),
    m_is_subsidiary_store(false),
    m_cur_minibatch(1000000),
    m_is_setup(false),
    m_verbose(false)
{
  if (m_reader == nullptr) {
    LBANN_ERROR(" m_reader is nullptr");
  }

  if (m_model == nullptr) {
    LBANN_ERROR(" m_model is nullptr");
  }

  m_comm = m_model->get_comm();
  if (m_comm == nullptr) {
    LBANN_ERROR(" m_comm is nullptr");
  }

  m_master = m_comm->am_world_master();
  m_rank = m_comm->get_rank_in_trainer();
  m_np = m_comm->get_procs_per_trainer();
  m_mpi_comm = m_comm->get_trainer_comm().GetMPIComm();

  m_dir = m_reader->get_file_dir();

  set_name("generic_data_store");

  if (m_master) std::cerr << "generic_data_store::generic_data_store; np: " << m_np << "\n";
  options *opts = options::get();
  if (opts->has_bool("extended_testing") && opts->get_bool("extended_testing")) {
    m_extended_testing = true;
  }

  if (opts->has_bool("local_disk") && opts->get_bool("local_disk")) {
    if (m_master) std::cerr << "running in out-of-memory mode\n";
    m_in_memory = false;
  }

  if (opts->has_bool("verbose") && opts->get_bool("verbose")) {
    m_verbose = true;
  }

  if (opts->has_string("use_tarball")) {
    m_dir = m_reader->get_local_file_dir();
  }

  if (m_comm->get_num_trainers() != 1) {
    if (m_master) {
      std::cerr << "\nFATAL ERROR: data store classes currently assume there is\n"
                << "a single model; please ask Dave Hysom to fix!\n\n";
    }
    exit(9);
  }
}

void generic_data_store::get_minibatch_index_vector() {
  size_t s2 = 0;
  for (auto t1 : (*m_my_minibatch_indices)) {
    s2 += t1.size();
  }
  m_my_minibatch_indices_v.reserve(s2);
  for (auto t1 : (*m_my_minibatch_indices)) {
    for (auto t2 : t1) {
      m_my_minibatch_indices_v.push_back(t2);
    }
  }
}

void generic_data_store::get_my_tarball_indices() {
 size_t idx = m_rank;
 do {
   m_my_datastore_indices.insert(idx);
   idx += m_np;
 } while (idx < m_num_global_indices);
}

void generic_data_store::get_my_datastore_indices() {
  for (size_t j=0; j<m_shuffled_indices->size(); ++j) {
    int idx = (*m_shuffled_indices)[j];
    int owner = idx % m_np;
    if (owner == m_rank) {
      m_my_datastore_indices.insert(idx);
    }
  }
}

void generic_data_store::setup(int mini_batch_size) {
  set_shuffled_indices( &(m_reader->get_shuffled_indices()) );
  set_num_global_indices();
  m_num_readers = m_reader->get_num_parallel_readers();
  if (m_master) {
    std::cerr << "data_reader type is: " << m_reader->get_type()
              << " num_readers: " << m_num_readers << " role: "
              << m_reader->get_role() << "\n";
  }

  if (is_subsidiary_store()) {
    return;
  }

  #if 0
  // get the set of global indices used by this processor in
  // generic_data_reader::fetch_data(). Note that these are
  // "original' indices, not shuffled indices, i.e, these indices
  // remain constant through all epochs
  if (m_master) { std::cerr << "calling m_model->collect_indices\n"; }
  m_reader->set_save_minibatch_entries(true);
  if (m_reader->get_role() == "train") {
    m_model->collect_indices(execution_mode::training);
  } else if (m_reader->get_role() == "validate") {
    m_model->collect_indices(execution_mode::validation);
  } else if (m_reader->get_role() == "test") {
    m_model->collect_indices(execution_mode::testing);
  } else {
    std::stringstream s2;
    s2 << __FILE__ << " " << __LINE__ << " :: "
       << " bad role; should be train, test, or validate;"
       << " we got: " << m_reader->get_role();
      throw lbann_exception(s2.str());
  }
  m_reader->set_save_minibatch_entries(false);
  m_my_minibatch_indices = &(m_reader->get_minibatch_indices());
  if (m_master) {
    std::cerr << "my num minibatch indices: " << m_my_minibatch_indices->size() << "\n";
  }
#endif
}

void generic_data_store::print_partitioned_indices() {
  if (! m_master) {
    return;
  }
  std::cerr << "\n\n=============================================\n"
            << "minibatch indices:\n";
  for (size_t j=0; j<m_all_partitioned_indices.size(); j++) {
    std::cerr << "===== P_"<<j<<"\n";
    for (size_t i=0; i<m_all_partitioned_indices[j].size(); i++) {
      std::cerr << "  mb #" << i << " ";
      for (size_t k=0; k<m_all_partitioned_indices[j][i].size(); k++) {
        std::cerr << m_all_partitioned_indices[j][i][k] <<  " ";
      }
      std::cerr << "\n";
    }
  }
  std::cerr << "=============================================\n\n";
}

size_t generic_data_store::get_file_size(std::string dir, std::string fn) {
  std::string imagepath;
  if (m_dir == "") {
    imagepath = fn;
  } else {
    imagepath = dir + fn;
  }
  struct stat st;
  if (stat(imagepath.c_str(), &st) != 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "stat failed for dir: " << dir
        << " and fn: " << fn
        << " on node: " << getenv("SLURMD_NODENAME");
    throw lbann_exception(err.str());
  }
  return st.st_size;
}

void generic_data_store::set_shuffled_indices(const std::vector<int> *indices, bool exchange_indices) {
if (m_master)std::cerr<<"starting set_shuffled_indices; epoch: "<<m_model->get_epoch()<<" role: " << m_reader->get_role()<<";  n: " << m_n << "\n";
  m_shuffled_indices = indices;
}

void generic_data_store::exchange_mb_counts() {
  int my_num_indices = m_my_minibatch_indices_v.size();
  m_mb_counts.resize(m_np);
  m_comm->trainer_all_gather<int>(my_num_indices, m_mb_counts);
}

void generic_data_store::exchange_mb_indices() {
  exchange_mb_counts();
  //setup data structures to exchange minibatch indices with all processors
  //displacement vector
  std::vector<int> displ(m_np);
  displ[0] = 0;
  for (size_t j=1; j<m_mb_counts.size(); j++) {
    displ[j] = displ[j-1] + m_mb_counts[j-1];
  }

  //recv vector
  int n = std::accumulate(m_mb_counts.begin(), m_mb_counts.end(), 0);
  std::vector<int> all_indices(n);

  //receive the indices
  m_comm->all_gather<int>(m_my_minibatch_indices_v, all_indices, m_mb_counts, displ, m_comm->get_world_comm());

  //fill in the final data structure
  m_all_minibatch_indices.resize(m_np);
  for (int j=0; j<m_np; j++) {
    m_all_minibatch_indices[j].reserve(m_mb_counts[j]);
    for (int i=displ[j]; i<displ[j]+m_mb_counts[j]; i++) {
      m_all_minibatch_indices[j].push_back(all_indices[i]);
    }
  }
}

void generic_data_store::exchange_partitioned_indices() {
  //determine the largest number of minibatches over all processors
  std::vector<int> counts(m_np);
  int my_num_mb = m_my_minibatch_indices->size();
  m_comm->trainer_all_gather<int>(my_num_mb, counts);
  m_num_minibatches = 0;
  for (auto t : counts) {
    m_num_minibatches = (size_t)t > m_num_minibatches ? t : m_num_minibatches;
  }
  if (m_master) std::cerr << "num minibatches: " << m_num_minibatches << "\n";

  //pack m_my_minibatch_indices into a single vector;
  //first, compute vector size, and exchange size with all procs
  std::vector<int> v;
  int count = m_my_minibatch_indices->size() + 1;
  for (auto t : (*m_my_minibatch_indices)) {
    count += t.size();
  }
  m_comm->trainer_all_gather<int>(count, counts);


  //now, fill in the vector
  std::vector<int> w;
  w.reserve(count);
  w.push_back(m_my_minibatch_indices->size());
  for (auto t : (*m_my_minibatch_indices)) {
    w.push_back(t.size());
    for (size_t h=0; h<t.size(); h++) {
      w.push_back(t[h]);
    }
  }
  if (w.size() != (size_t)count) {
    std::stringstream err;
    err << "count: " << count << " w.size: " << w.size();
    throw lbann_exception(err.str());
  }

  // exchange the vectors
  std::vector<int> displ(m_np);
  displ[0] = 0;
  for (size_t k=1; k<counts.size(); k++) {
    displ[k] = displ[k-1] + counts[k-1];
  }

  //construct recv vector
  int n = std::accumulate(counts.begin(), counts.end(), 0);
  std::vector<int> all_w(n);

  //exchange the indices
  m_comm->all_gather<int>(w, all_w, counts, displ, m_comm->get_world_comm());

  //fill in the final data structure
  m_all_partitioned_indices.resize(m_np);
  for (size_t p=0; p<(size_t)m_np; p++) {
    int *ww = all_w.data() + displ[p];
    //note: it's possible that m_num_minibatches > num_minibatches;
    //      that's OK; for simplicity elsewhere in the code we want
    //      all procs to have the same number of minibatches
    m_all_partitioned_indices[p].resize(m_num_minibatches);
    size_t num_minibatches = *ww++;
    for (size_t i=0; i<num_minibatches; i++) {
      int mb_size = *ww++;
      m_all_partitioned_indices[p][i].reserve(mb_size);
      for (int j=0; j<mb_size; j++) {
        m_all_partitioned_indices[p][i].push_back(*ww++);
      }
    }
  }
}

std::pair<std::string, std::string> generic_data_store::get_pathname_and_prefix(std::string s) {
  int num_slash = std::count(s.begin(), s.end(), '/');
  if (num_slash < 1 || s.back() == '/') {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "<string> must be of the form: <pathname>/prefix";
    throw lbann_exception(err.str());
  }

  size_t j = s.rfind('/');
  std::string prefix = s.substr(j+1);
  std::string pathname = s.substr(0, j);
  return std::make_pair(prefix, pathname);
}

void generic_data_store::create_dirs(std::string s) {
  if (m_comm->get_rank_in_node() == 0) {
    if (s.back() != '/') {
      s += '/';
    }
    size_t i = s.find('/', 1);
    while (i != std::string::npos) {
      std::string s2 = s.substr(0, i);
      const int dir_err = mkdir(s2.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if (dir_err == -1 && errno != 17) { // 17: File Exists
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "failed to create directory: " << s2 << "\n"
            << "error code is: " << errno << " -> " << std::strerror(errno)
            << "\n" << getenv("SLURMD_NODENAME");
        throw lbann_exception(err.str());
      }
      i = s.find('/', i+1);
    }
  }
  m_comm->barrier(m_comm->get_node_comm());
}

std::string generic_data_store::run_cmd(std::string cmd, bool exit_on_error) {
  std::array<char, 128> buffer;
  std::string result;
  size_t len = cmd.size();
  //copy to c-style string; Jay-Seung says this may be needed on ray
  char *b = new char[len+1];
  strcpy(b, cmd.data());
  b[len] = '\0';
  std::shared_ptr<FILE> pipe(popen(b, "r"), pclose);
  if (!pipe) throw std::runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
      if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
        result += buffer.data();
  }
  if (exit_on_error && result != "") {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "system call returned:\n" << result;
    throw lbann_exception(err.str());
  }
  return result;
}

int generic_data_store::get_index_owner(int idx) {
  if (m_owner.find(idx) == m_owner.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << " idx: " << idx << " was not found in the m_owner map;"
        << " map size: " << m_owner.size();
    throw lbann_exception(err.str());
  }
  return m_owner[idx];
}

void generic_data_store::build_index_owner() {
  m_owner.clear();
  int num_indices = m_my_datastore_indices.size();
  if (num_indices == 0) {
    num_indices = 1;
  }
  std::vector<int>counts(m_np);
  m_comm->trainer_all_gather<int>(num_indices, counts);

  std::vector<int> disp(m_np);
  disp[0] = 0;
  for (int h=1; h<m_np; h++) {
    disp[h] = disp[h-1] + counts[h-1];
  }
  int num_global_indices = std::accumulate(counts.begin(), counts.end(), 0);
  std::vector<int> my_indices;
  my_indices.reserve(num_indices);
  if (m_my_datastore_indices.empty()) {
    my_indices.push_back(-1);
  }
  for (auto t : m_my_datastore_indices) {
    my_indices.push_back(t);
  }

  std::vector<int> all_indices(num_global_indices);
  m_comm->all_gather<int>(my_indices, all_indices, counts, disp, m_comm->get_world_comm());
  for (size_t rank=0; rank<counts.size(); rank++) {
    for (int j = disp[rank]; j<disp[rank] + counts[rank]; j++) {
      if (all_indices[j] != -1) {
        m_owner[all_indices[j]] = rank;
      }
    }
  }
}

}  // namespace lbann
