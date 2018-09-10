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

#include "lbann/data_store/data_store_image.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/data_readers/data_reader.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/io/file_io.hpp"

#ifdef LBANN_SYS_SENDFILE_OK
#include <sys/sendfile.h>
#endif // LBANN_SYS_SENDFILE_OK

#include <sys/stat.h>

namespace lbann {

data_store_image::~data_store_image() {
}

void data_store_image::setup() {
  set_name("data_store_image");
  if (m_master) std::cerr << "starting data_store_image::setup()\n";

  options *opts = options::get();
  bool using_tarball = opts->has_string("use_tarball") ? true : false;
  bool creating_tarball = are_we_creating_tarballs();
  if (using_tarball && creating_tarball) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
             " :: you cannot use both --using_tarball and --creating_tarball options");
  }

  if (using_tarball) {
    stage_tarball();
    m_comm->global_barrier();
  }

  if (m_master) std::cerr << "data_store_image - calling generic_data_store::setup()\n";
  generic_data_store::setup();

  double tm;

  //==========================================================================
  // block for running in out-of-memory mode
  //==========================================================================
  if (! m_in_memory) {
    if (!creating_tarball) {
      if (m_master) std::cerr << "data_store_image - calling exchange_partitioned_indices\n";
      exchange_partitioned_indices();
    }

    if (m_master) std::cerr << "data_store_image - calling get_my_datastore_indices\n";
    if (creating_tarball) {
      get_my_tarball_indices();
    } else {
      get_my_datastore_indices();
    }

    if (m_master) std::cerr << "data_store_image - calling build_data_filepaths\n";
    if (using_tarball) {
      read_data_filepaths();
    } else {
      build_data_filepaths();
    }

    if (m_master) std::cerr << "data_store_image - calling get_file_sizes\n";
    tm = get_time();
    if (using_tarball) {
      read_file_sizes();
      read_datastore_indices();
    } else {
      get_file_sizes();
    }
    if (m_master) std::cerr << "TIME for get_file_sizes: " << get_time() - tm << "\n";

    if (m_master) std::cerr << "data_store_image - calling build_index_owner\n";
    tm = get_time();
    build_index_owner();
    if (m_master) std::cerr << "TIME for build_index_owner: " << get_time() - tm << "\n";

    if (! using_tarball) {
      if (m_master) std::cerr << "data_store_image - calling stage_files\n";
      tm = get_time();
      stage_files();
      if (m_master) std::cerr << "TIME for stage_files: " << get_time() - tm << "\n";
    }

    // create tarball and copy to lscratch (or where ever)
    if (creating_tarball) {
      if (m_master) std::cerr << "data_store_image - creating tarball\n";
      tm = get_time();
      create_tarball();
      if (m_master) std::cerr << "TIME for creating tarball: " << get_time() - tm << "\n";
    }

    m_is_setup = true;
  }

  //==========================================================================
  // block for running in in-memory mode
  //==========================================================================
  else {
    if (m_master) std::cerr << "data_store_image - calling get_minibatch_index_vector\n";
    get_minibatch_index_vector();

    if (m_master) std::cerr << "data_store_image - calling exchange_mb_indices\n";
    exchange_mb_indices();

    if (m_master) std::cerr << "data_store_image - calling get_my_datastore_indices\n";
    get_my_datastore_indices();

    if (m_master) std::cerr << "data_store_image - calling get_file_sizes\n";
    double tma = get_time();
    get_file_sizes();
    size_t num_bytes = get_global_num_file_bytes();
    if (m_master) std::cerr << "TIME for get_file_sizes: " << get_time() - tma << " global num files: " << m_file_sizes.size() << " data set size: " << ((double)num_bytes/1000000) << " MB\n";

    if (m_master) std::cerr << "data_store_image - calling build_index_owner\n";
    tm = get_time();
    build_index_owner();
    if (m_master) std::cerr << "TIME for build_index_owner: " << get_time() - tm << "\n";

    if (m_master) std::cerr << "data_store_image - calling report_memory_constrains\n";
    report_memory_constraints();

    if (m_master) std::cerr << "data_store_image - calling read_files\n";
    tma = get_time();
    read_files();
    if (m_master) std::cerr << "TIME for read_files: " << get_time() - tma << "\n";

    if (m_master) std::cerr << "data_store_image - calling exchange_data\n";
    exchange_data();

    if (m_extended_testing) {
      if (m_master) std::cerr << "data_store_image - calling extended_testing\n";
      extended_testing();
    }
  }
}


void data_store_image::get_data_buf(int data_id, std::vector<unsigned char> *&buf, int multi_idx) {
  std::stringstream err;
  int index = data_id * m_num_img_srcs + multi_idx;
  if (m_my_minibatch_data.find(index) == m_my_minibatch_data.end()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find index: " << index << " in m_my_minibatch_data; size: "
        << m_my_minibatch_data.size() << " role: " << m_reader->get_role();
    throw lbann_exception(err.str());
  }

  buf = &m_my_minibatch_data[index];
}

void data_store_image::load_file(const std::string &dir, const std::string &fn, unsigned char *p, size_t sz) {
  std::string imagepath;
  if (dir != "") {
    imagepath = dir + fn;
  } else {
    imagepath = fn;
  }
  std::ifstream in(imagepath.c_str(), std::ios::in | std::ios::binary);
  if (!in) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << imagepath << " for reading"
        << "; dir: " << dir << "  fn: " << fn << "\n"
        << "hostname: " << getenv("SLURMD_NODENAME") << " role: " << m_reader->get_role();
    throw lbann_exception(err.str());
  }
  in.read((char*)p, sz);
  if ((int)sz != in.gcount()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to read " << sz << " bytes from " << imagepath
        << " num bytes read: " << in.gcount()
        << "\nhostname: " << getenv("SLURMD_NODENAME") << " role: " << m_reader->get_role();
    throw lbann_exception(err.str());
  }
  in.close();
}

void data_store_image::exchange_data() {
  double tm1 = get_time();
  std::stringstream err;

  //build map: proc -> global indices that proc needs for this epoch, and
  //                   which I own
  std::unordered_map<int, std::unordered_set<int>> proc_to_indices;
  for (size_t p=0; p<m_all_minibatch_indices.size(); p++) {
    for (auto idx : m_all_minibatch_indices[p]) {
      int index = (*m_shuffled_indices)[idx];
      if (m_my_datastore_indices.find(index) != m_my_datastore_indices.end()) {
        proc_to_indices[p].insert(index);
      }
    }
  }

  //start sends
  std::vector<std::vector<El::mpi::Request<unsigned char>>> send_req(m_np);
  for (int p=0; p<m_np; p++) {
    send_req[p].resize(proc_to_indices[p].size()*m_num_img_srcs);
    size_t jj = 0;
    for (auto idx : proc_to_indices[p]) {
      for (size_t k=0; k<m_num_img_srcs; k++) {
        int index = idx*m_num_img_srcs+k;
        if (m_file_sizes.find(index) == m_file_sizes.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_file_sizes.find(" << index << ") failed";
          throw lbann_exception(err.str());
        }
        int len = m_file_sizes[index];
        m_comm->nb_tagged_send<unsigned char>(
            m_data[index].data(), len, p, index,
            send_req[p][jj++], m_comm->get_model_comm());

      }
    }
    if (jj != send_req[p].size()) throw lbann_exception("ERROR 1");
  } //start sends

  //build map: proc -> global indices that proc owns that I need
  proc_to_indices.clear();
  for (auto idx : m_my_minibatch_indices_v) {
    int index = (*m_shuffled_indices)[idx];
    int owner = get_index_owner(index);
    proc_to_indices[owner].insert(index);
  }

  //start recvs
  m_my_minibatch_data.clear();
  std::vector<std::vector<El::mpi::Request<unsigned char>>> recv_req(m_np);
  for (auto t : proc_to_indices) {
    int owner = t.first;
    size_t jj = 0;
    const std::unordered_set<int> &s = t.second;
    recv_req[owner].resize(s.size()*m_num_img_srcs);
    for (auto idx : s) {
      for (size_t k=0; k<m_num_img_srcs; k++) {
        size_t index = idx*m_num_img_srcs+k;
        if (m_file_sizes.find(index) == m_file_sizes.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_file_sizes.find(" << index << ") failed"
              << " m_file_sizes.size(): " << m_file_sizes.size()
              << " m_my_minibatch_indices_v.size(): " << m_my_minibatch_indices_v.size();
        }
        size_t len = m_file_sizes[index];
        m_my_minibatch_data[index].resize(len);
        m_comm->nb_tagged_recv<unsigned char>(
            m_my_minibatch_data[index].data(), len, owner,
            index, recv_req[owner][jj++], m_comm->get_model_comm());
      }
    }
  }

  //wait for sends to finish
  for (size_t i=0; i<send_req.size(); i++) {
    m_comm->wait_all<unsigned char>(send_req[i]);
  }

  //wait for recvs to finish
  for (size_t i=0; i<recv_req.size(); i++) {
    m_comm->wait_all<unsigned char>(recv_req[i]);
  }

  if (m_master) {
    std::cerr << "TIME for exchange_data: " << get_time() - tm1
              << "; role: " << m_reader->get_role() << "\n";
  }
}


void data_store_image::exchange_file_sizes(
  std::vector<int> &my_global_indices,
  std::vector<int> &my_num_bytes) {

  if (my_global_indices.size() == 0) {
    my_global_indices.push_back(-1);
    my_num_bytes.push_back(-1);
  }

  std::vector<int> rcv_counts(m_np);
  int nbytes = my_global_indices.size();
  m_comm->model_all_gather<int>(nbytes, rcv_counts);
  int num_global_indices = std::accumulate(rcv_counts.begin(), rcv_counts.end(), 0);

  std::vector<int> disp(m_np);   //@todo: fix for model
  disp[0] = 0;
  for (int h=1; h<m_np; h++) {
    disp[h] = disp[h-1] + rcv_counts[h-1];
  }
  std::vector<int> all_global_indices(num_global_indices);
  std::vector<int> all_num_bytes(num_global_indices);

  m_comm->all_gather<int>(my_global_indices, all_global_indices, rcv_counts, disp, m_comm->get_world_comm());

  m_comm->all_gather<int>(my_num_bytes, all_num_bytes, rcv_counts, disp, m_comm->get_world_comm());

  for (size_t j=0; j<all_global_indices.size(); j++) {
    if (all_global_indices[j] != -1) {
      m_file_sizes[all_global_indices[j]] = all_num_bytes[j];
    }
  }
}

size_t data_store_image::get_global_num_file_bytes() {
  size_t n = get_my_num_file_bytes();
  size_t g = 0;
  if (m_master) {
    g = m_comm->reduce(n, m_comm->get_world_comm());
  } else {
    m_comm->reduce(n, 0, m_comm->get_world_comm());
  }
  return g;
}

size_t data_store_image::get_my_num_file_bytes() {
  size_t count = 0;
  for (auto idx : m_my_datastore_indices) {
    for (size_t i=0; i<m_num_img_srcs; i++) {
      int index = idx*m_num_img_srcs + i;
      if (m_file_sizes.find(index) == m_file_sizes.end()) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << " failed to find " << idx << " in m_file_sizes; count: " << count
            << " m_file_sizes.size(): " << m_file_sizes.size();
        throw lbann_exception(err.str());
      }
      count += m_file_sizes[index];
    }
  }
  return count;
}

size_t data_store_image::get_available_memory() {
  std::ifstream in("/proc/meminfo");
  std::string line;
  size_t size;
  bool found = false;
  std::string name;
  std::string units;
  while (! in.eof()) {
    getline(in, line);
    std::stringstream s(line);
    s >> name >> size >> units;
    if (name.find("MemFree") != std::string::npos) {
      found = true;
      break;
    }
  }
  in.close();

  if (!found) {
    if (m_master) {
      std::cerr <<
        "\nWARNING: data_store_image::get_available_memory failed\n"
        "failed to find 'MemFree in /proc/meminfo\n"
        "therefore we cannot advise whether you have enough resources\n"
        "to contain all data files in memory\n";
    }
    return 0;
  }
  return size;
}


//note: this could be done on P_0 with no communication,
//      but it's a cheap operation, so I'm coding it the
//      easy way
void data_store_image::report_memory_constraints() {
  size_t count = get_my_num_file_bytes();

  std::vector<long long> counts(m_np);
  if (m_master) {
    m_comm->gather<long long>(count, counts.data(), m_comm->get_world_comm());
  } else {
    m_comm->gather<long long>(count, 0, m_comm->get_world_comm());
  }

  double global = get_global_num_file_bytes()/1000000;

  if (!m_master) {
    return;
  }

  /// determine the amount of memory required for files for all
  /// processors on this node
  double required = 0;
  for (int p=0; p<m_np; p++) {
    if (m_comm->is_rank_node_local(p, m_comm->get_world_comm())) {
      required += counts[p];
    }
  }
  required /= 1000000;

  double available = get_available_memory();
  if (available == 0) {
    std::cerr << required << " kB of memory are required for files on this node\n";
    return;
  }
  available /= 1000;

  double percent = required / available * 100.0;
  std::cerr << "\n"
            << "===============================================\n"
            << "Memory Constraints for: " << m_reader->get_role() << "\n"
            << "Global data set size:               " << global << " MB\n"
            << "Required for data set on this node: " << required << " MB\n"
            << "Available memory on this node: "      << available << " MB\n"
            << "Required is " << percent << " % of Available\n"
            << "===============================================\n\n";

  double limit = 0.8;
  if (options::get()->has_float("mem_limit")) {
    limit = options::get()->get_float("mem_limit");
  }
  if (required > limit*available) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "You have insufficient memory to hold all required files;\n"
        << "required is > 80% of available\n"
        << "quitting now, so you don't waste your time\n";
  }
}


// the input string "s" should be one of the forms:
//   dir1/[dir2/...]/filename
//   /dir1/[dir2/...]/filename
//   /dir1/[dir2/...]/

void data_store_image::stage_files() {
  std::stringstream err;

  //create directory structure on local file store
  std::string local_dir = m_reader->get_local_file_dir();
  create_dirs(local_dir);
  m_comm->global_barrier();
  std::unordered_set<std::string> make_dirs;
  for (auto t : m_data_filepaths) {
    size_t j = t.second.rfind('/');
    if (j != std::string::npos) {
      make_dirs.insert(t.second.substr(0, j+1));
    }
  }

  std::string dir = m_reader->get_file_dir();
  std::stringstream ss;
  for (auto t : make_dirs) {
    ss.clear();
    ss.str("");
    ss << local_dir << "/" << t;
    create_dirs(ss.str());
  }
  m_comm->global_barrier();

  // copy files to local store
  size_t j = 0;
  struct stat stat_buf;
  double tm = get_time();
  std::stringstream s;
  int write_fd;

  for (auto t : m_data_filepaths) {
    s.clear();
    s.str("");
    s << local_dir << '/' << t.second;
    ++j;
    if (j % 100 == 0 and m_master) {
      double e = get_time() - tm;
      double time_per_file = e / j;
      int remaining_files = m_data_filepaths.size()-j;
      double estimated_remaining_time = time_per_file * remaining_files;
      std::cerr << "P_0: staged " << j << " of " << m_data_filepaths.size()
                << " files; elapsed time: " << get_time() - tm
                << "s est. remaining time: " << estimated_remaining_time << "s\n";
    }
    if (access(s.str().c_str(), F_OK | R_OK) == -1 ) {
      write_fd = open(s.str().c_str(),  O_RDWR | O_CREAT, S_IRWXU);
      if (write_fd == -1) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "failed to open " << s.str() << " for writing;\n"
            << "error code is: " << std::strerror(errno) << "\n"
            << "local_dir: " << local_dir << " m_cur_minibatch: " << 1+m_cur_minibatch;
        throw lbann_exception(err.str());
      }
      off_t offset = 0;
      s.clear();
      s.str("");
      s << dir << '/' << t.second;
      int read_fd = open(s.str().c_str(), O_RDONLY);
      if (read_fd == -1) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "failed to open " << s.str() << " for reading;\n"
            << "error code is: " << std::strerror(errno);
        throw lbann_exception(err.str());
      }
      int e2 = fstat(read_fd, &stat_buf);
      if (e2 == -1) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "fstat failed for file: " << s.str();
        throw lbann_exception(err.str());
      }
#ifdef LBANN_SYS_SENDFILE_OK
      ssize_t e = sendfile(write_fd, read_fd, &offset, stat_buf.st_size);
      if (e == -1) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "failed to copy file to location: " << s.str()
            << ";\nerror code is: " << std::strerror(errno);
        throw lbann_exception(err.str());
      }
#else
      // FIXME: This is the fastest way to deal with this issue for
      // OSX pending a "real" fix.
      (void) offset;// Silence a warning
      err << __FILE__ << " " << __LINE__ << " :: "
          << "Header <sys/sendfile.h> is not found on this system. "
          << "sendfile() won't work. This is not a permanent fix.";
      throw lbann_exception(err.str());
#endif // LBANN_SYS_SENDFILE_OK

      close(read_fd);
      close(write_fd);
    }
  }
}

void data_store_image::fetch_data() {
  if (!m_is_setup) {
    return;
  }
  std::stringstream err;
  double tm1 = get_time();

  ++m_cur_minibatch;
  if (m_cur_minibatch >= m_all_partitioned_indices[0].size()) {
    m_cur_minibatch = 0;
  }

  //build map: proc -> global indices that proc needs for this epoch, and
  //                   which I own
  std::unordered_map<int, std::unordered_set<int>> proc_to_indices;

  for (int p = 0; p<m_np; p++) {
      if (m_cur_minibatch > m_all_partitioned_indices[p].size() -1) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "send to: P_" << p << " m_cur_minibatch: " << m_cur_minibatch
            << " m_all_partitioned_indices[p].size(): " << m_all_partitioned_indices[p].size();
        throw lbann_exception(err.str());
      }
      const std::vector<int> &v = m_all_partitioned_indices[p][m_cur_minibatch];
      for (auto idx : v) {
        int index = (*m_shuffled_indices)[idx];
        if (m_my_datastore_indices.find(index) != m_my_datastore_indices.end()) {
          proc_to_indices[p].insert(index);
        }
      }
  }

  //read required files and start sends
  m_data.clear();

  //compute number of sends, and allocate Request vector
  size_t num_sends = 0;
  for (auto t : proc_to_indices) {
    num_sends += t.second.size();
  }
  num_sends *= m_num_img_srcs;
  std::vector<El::mpi::Request<unsigned char>> send_req(num_sends);

  size_t req_idx = 0;
  for (int p=0; p<m_np; p++) {
    if (m_all_partitioned_indices[p].size() >= m_cur_minibatch
        && proc_to_indices.find(p) != proc_to_indices.end()) {
      const std::unordered_set<int> &s = proc_to_indices[p];
      read_files(s);
      for (auto idx : s) {
        for (size_t k=0; k<m_num_img_srcs; k++) {
          int index = idx*m_num_img_srcs+k;
          int len = m_file_sizes[index];
          m_comm->nb_tagged_send<unsigned char>(
                         m_data[index].data(), len, p, index,
                         send_req[req_idx++], m_comm->get_model_comm());
        }
      }
    }
  }


  //build map: proc -> global indices that proc owns that I need
  proc_to_indices.clear();
  if (m_cur_minibatch < m_my_minibatch_indices->size()) {
    for (auto idx  : (*m_my_minibatch_indices)[m_cur_minibatch]) {
      int index = (*m_shuffled_indices)[idx];
      int owner = get_index_owner(index);
      proc_to_indices[owner].insert(index);
    }
  }

  //compute number recvs, and allocate Request vector
  size_t num_recvs = 0;
  for (auto t : proc_to_indices) {
    num_recvs += t.second.size();
  }
  num_recvs *= m_num_img_srcs;


  //start recvs
  m_my_minibatch_data.clear();
  req_idx = 0;
  std::vector<El::mpi::Request<unsigned char>> recv_req(num_recvs);
  for (auto t : proc_to_indices) {
    int owner = t.first;
    const std::unordered_set<int> &s = t.second;
    for (auto idx : s) {
      //note: for imagenet_reader, m_num_img_srcs = 1;
      //      for other readers (multi, triplet) it is larger, probably three
      for (size_t k=0; k<m_num_img_srcs; k++) {
        size_t index = idx*m_num_img_srcs+k;
        if (m_file_sizes.find(index) == m_file_sizes.end()) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << " m_file_sizes.find(" << index << ") failed"
              << " m_file_sizes.size(): " << m_file_sizes.size()
              << " m_my_minibatch_indices_v.size(): " << m_my_minibatch_indices_v.size();
         throw lbann_exception(err.str());
        }
        size_t len = m_file_sizes[index];
        m_my_minibatch_data[index].resize(len);
        m_comm->nb_tagged_recv<unsigned char>(
            m_my_minibatch_data[index].data(), len, owner,
            index, recv_req[req_idx++], m_comm->get_model_comm());
      }
    }
  }

  //wait for sends to finish
  m_comm->wait_all<unsigned char>(send_req);

  //wait for recvs to finish
  m_comm->wait_all<unsigned char>(recv_req);

  if (m_master && m_verbose) {
    std::cerr << "TIME (P_0) for reading from local disk: "
              << get_time() - tm1 << "; role: " << m_reader->get_role()
              << "  minibatch " << 1+m_cur_minibatch << " of "
              << m_num_minibatches << "; " << m_reader->get_role() << "\n";
  }
}

void data_store_image::write_data_filepaths() {
  std::string local_dir = m_reader->get_local_file_dir();
  std::stringstream s;
  s << local_dir << "/data_filepaths.txt";
  std::ofstream out(s.str().c_str());
  if (! out.good()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << s.str() << " for reading";
    throw lbann_exception(err.str());
  }
  for (auto t : m_data_filepaths) {
    out << t.first << " " << t.second << "\n";
  }
  out.close();
}

void data_store_image::read_data_filepaths() {
  std::string local_dir = m_reader->get_local_file_dir();
  std::stringstream s;
  s << local_dir << "/data_filepaths.txt";
  std::ifstream in(s.str().c_str());
  //note: file_sizes.txt may not exist on all processors;
  //      this is the case where we're using tarballed data, and
  //      running with more processors than were used to create
  //      the tarball
  if (in.good()) {
    std::string path;
    int idx;
    while (in >> idx >> path) {
      m_data_filepaths[idx] = path;
    }
    in.close();
  }
}

void data_store_image::write_file_sizes() {
  std::string local_dir = m_reader->get_local_file_dir();
  std::stringstream s;
  s << local_dir << "/file_sizes.txt";
  std::ofstream out(s.str().c_str());
  if (! out.good()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << s.str() << " for reading";
    throw lbann_exception(err.str());
  }
  for (auto t : m_file_sizes) {
    out << t.first << " " << t.second << "\n";
  }
  out.close();
}

std::pair<std::string, std::string>  data_store_image::get_tarball_exe() {
  std::pair<std::string, std::string> names = get_pathname_and_prefix(
                           options::get()->get_string("create_tarball"));
  if (m_master) {
    std::cerr << "tarball dir: " << names.second << "\n"
              << "tarball prefix: " << names.first << "\n\n";
  }

  //This is somewhat fragile. For now I assume the local_dir is
  //of the form: /l/ssd/train
  std::string local_dir = m_reader->get_local_file_dir();
  size_t j = local_dir.rfind('/', local_dir.size()-2);
  std::string work_dir = local_dir.substr(0, j);

  std::stringstream tarball_name;
  tarball_name << names.first << "_" << m_reader->get_role()
      << "_rank=" << m_rank << "_np=" << m_np << ".tar";
  std::stringstream exe_1;
  std::stringstream exe_2;
  exe_1 << "tar cf " << work_dir << '/' << tarball_name.str()
      << " " << local_dir;
  exe_2 << "cp -f " << work_dir << '/' << tarball_name.str()
      << " " << names.second;
  return std::make_pair(exe_1.str(), exe_2.str());
}

void data_store_image::create_tarball() {
  write_file_sizes();
  write_datastore_indices();
  write_data_filepaths();
  std::pair<std::string, std::string> cmds = get_tarball_exe();
  if (m_master)  std::cerr << "\nabout to execute: " << cmds.first << "\n";
  run_cmd(cmds.first);
  if (m_master)  std::cerr << "\nabout to execute: " << cmds.second << "\n";
  run_cmd(cmds.second);
}

bool data_store_image::are_we_creating_tarballs() {
  bool retval = false;
  options *opts = options::get();
  if (opts->has_string("create_tarball")) {
    if (m_comm->get_procs_per_node() != 1) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "--create_tarball=<string> was specified; you have "
          << m_comm->get_procs_per_node() << "; you must use a"
          << "single core per node when creating tarballs";
      throw lbann_exception(err.str());
    }
    retval = true;
    if (m_reader->get_role() == "validate") {
      return retval;
    }
  }
  return retval;
}

void data_store_image::read_file_sizes() {
  std::stringstream s;
  s << m_reader->get_local_file_dir() << "/file_sizes.txt";
  //note: file_sizes.txt may not exist on all processors;
  //      this is the case where we're using tarballed data, and
  //      running with more processors than were used to create
  //      the tarball
  std::ifstream in(s.str().c_str());
  if (in.good()) {
    size_t idx;
    size_t len;
    while (in >> idx >> len) {
      m_file_sizes[idx] = len;
    }
  }
  m_comm->global_barrier();
  int n = m_file_sizes.size();
  m_comm->broadcast<int>(0, n, m_comm->get_world_comm());
  std::vector<int> s2(n*2);
  if (m_rank == 0) {
    size_t j = 0;
    for (auto t : m_file_sizes) {
      s2[j++] = t.first;
      s2[j++] = t.second;
    }
  }

  m_comm->broadcast(0, s2.data(), s2.size(), m_comm->get_model_comm(),
                    El::SyncInfo<El::Device::CPU>{});
  for (size_t j=0; j<s2.size(); j+=2) {
    m_file_sizes[s2[j]] = s2[j+1];
  }
}

void data_store_image::write_datastore_indices() {
  std::string local_dir = m_reader->get_local_file_dir();
  std::stringstream s;
  s << local_dir << "/datastore_indices.txt";
  std::ofstream out(s.str().c_str());
  if (! out.good()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << s.str() << " for reading";
    throw lbann_exception(err.str());
  }
  for (auto t: m_my_datastore_indices) {
    out << t << "\n";
  }
}

void data_store_image::read_datastore_indices() {
  m_my_datastore_indices.clear();
  if (m_comm->get_rank_in_node() == 0) {
    std::stringstream s;
    s << m_reader->get_local_file_dir() << "/datastore_indices.txt";
    std::ifstream in(s.str().c_str());

    //note: datastore_indices.txt may not exist on all processors;
    //      this is the case where we're using tarballed data, and
    //      running with more processors than were used to create
    //      the tarball
    if (in.good()) {
      int idx;
      while (in >> idx) {
        m_my_datastore_indices.insert(idx);
      }
      in.close();
    }
  }
}

void data_store_image::stage_tarball() {
  if (m_reader->get_role() == "validate") {
    return;
  }
  std::stringstream err;
  int procs_per_node = m_comm->get_procs_per_node();
  int max = m_comm->allreduce(procs_per_node, m_comm->get_world_comm(),El::mpi::MAX);
  int min = m_comm->allreduce(procs_per_node, m_comm->get_world_comm(),El::mpi::MIN);
  if (max != min) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "all nodes must contain the same number of active processors";
    throw lbann_exception(err.str());
  }

/*
  for (int j=0; j<m_np; j++) {
    m_comm->global_barrier();
    if (m_rank == j) {
      std::cerr << "rank: " << m_rank << " rank in node: " << m_comm->get_rank_in_node() << " procs_per_node: " << procs_per_node << "
    }
  }
*/

  options *opts = options::get();
  if (!opts->has_int("num_tarballs")) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "you must specify --num_tarballs=<int>";
    throw lbann_exception(err.str());
  }
  int num_tarballs = opts->get_int("num_tarballs");

  int fake_rank = m_rank / procs_per_node;
  if (m_comm->get_rank_in_node() == 0 && fake_rank < num_tarballs) {
    std::string raw_name = options::get()->get_string("use_tarball");
    std::pair<std::string, std::string> names = get_pathname_and_prefix(raw_name);

  if (m_master) std::cerr << "num tarballs: " << num_tarballs << "\n";
    std::stringstream tarball_filename(names.first);
    tarball_filename << names.first << "_" << m_reader->get_role()
              << "_rank=" << fake_rank << "_np=" << num_tarballs << ".tar";

    //This is somewhat fragile. For now I assume the local_dir is
    //of the form: /l/ssd/train
    std::string local_dir = m_reader->get_local_file_dir();
    size_t j = local_dir.rfind('/', local_dir.size()-2);
    std::string ssd = local_dir.substr(0, j);

    std::stringstream s;
    s << "cp -f " << names.second << "/" << tarball_filename.str() << " " << ssd;
    if (m_master)  std::cerr << "\nabout to execute: " << s.str()<< "\n";
    run_cmd(s.str());

    s.clear();
    s.str("");
    s << "cd " << ssd << "; tar xf " << tarball_filename.str();
    if (m_master)  std::cerr << "\nabout to execute: " << s.str()<< "\n";
    run_cmd(s.str());

    s.clear();
    s.str("");
    s << "mv " << ssd  << "/" << local_dir << " " << ssd;
    if (m_master)  std::cerr << "\nabout to execute: " << s.str()<< "\n";
    run_cmd(s.str());
  }
  m_comm->global_barrier();
}

}  // namespace lbann
