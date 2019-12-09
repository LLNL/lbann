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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/pilot2_conduit_data_reader.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_node.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include <unordered_set>
#include "lbann/utils/file_utils.hpp" // pad()
#include "lbann/utils/jag_utils.hpp"  // read_filelist(..) TODO should be move to file_utils
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/lbann_library.hpp"

namespace lbann {

pilot2_conduit_data_reader::pilot2_conduit_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

pilot2_conduit_data_reader::pilot2_conduit_data_reader(const pilot2_conduit_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

pilot2_conduit_data_reader& pilot2_conduit_data_reader::operator=(const pilot2_conduit_data_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void pilot2_conduit_data_reader::copy_members(const pilot2_conduit_data_reader &rhs) {
  if (is_master()) {
    std::cout << "Starting pilot2_conduit_data_reader::copy_members\n";
  }
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);
  m_filenames = rhs.m_filenames;
  m_samples_per_file = rhs.m_samples_per_file;
  m_data_id_map = rhs.m_data_id_map;
  m_datum_shapes = rhs.m_datum_shapes;
  m_datum_word_sizes = rhs.m_datum_word_sizes;
  m_datum_num_bytes = rhs.m_datum_num_bytes;
  m_datum_num_words = rhs.m_datum_num_words;
  m_num_features = rhs.m_num_features;
  m_num_labels = rhs.m_num_labels;
  m_num_response_features = rhs.m_num_response_features;
  m_data_dims = rhs.m_data_dims;
}

void pilot2_conduit_data_reader::load() {
  options *opts = options::get();

  if (! opts->get_bool("preload_data_store")) {
    LBANN_ERROR("pilot2_conduit_data_reader requires data_store; please pass either --preload_data_store on the cmd line");
  }

  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    // The following member variables of the leadering reader should have been
    // copied when this was copy-constructed: m_sample_list, and m_open_hdf5_files
    return;
  }

  m_shuffled_indices.clear();

  //TODO: "Load the sample list" and "Merge all of the sample lists"
  //      is cut-n-paste from data_reader_jag_conduit.cpp; this should
  //      be refactored to avoid duplication

  // Load the sample list
  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string sample_list_file = data_dir + get_data_index_list();
  size_t stride = m_comm->get_procs_per_trainer();
  size_t offset = m_comm->get_rank_in_trainer();
  double tm1 = get_time();
  m_sample_list.load(sample_list_file, stride, offset);
  double tm2 = get_time();
  if (is_master()) {
    std::cout << "Time to load sample list: " << tm2 - tm1 << std::endl;
  }

  /// Merge all of the sample lists
  tm2 = get_time();
  m_sample_list.all_gather_packed_lists(*m_comm);
  if (opts->has_string("write_sample_list") && m_comm->am_trainer_master()) {
    {
      const std::string msg = " writing sample list " + sample_list_file;
      LBANN_WARNING(msg);
    }
    std::stringstream s;
    std::string basename = get_basename_without_ext(sample_list_file);
    std::string ext = get_ext_name(sample_list_file);
    s << basename << "." << ext;
    m_sample_list.write(s.str());
  }
  if (is_master()) {
    std::cout << "time for all_gather_packed_lists: " << get_time() - tm2 << std::endl;
  }

  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  fill_in_metadata();

  instantiate_data_store();
  select_subset_of_data();
}

void pilot2_conduit_data_reader::do_preload_data_store() {
  if (is_master()) std::cout << "Starting pilot2_conduit_data_reader::do_preload_data_store; num indices: " << utils::commify(m_shuffled_indices.size()) << std::endl;

#if 0
==========================================================================
data types, from python+numpy:

  rots          numpy.float64  RAS rotation angle    
  states        numpy.float64  the assigned state
  tilts         numpy.float64  RAS tilt angle
  density_sig1  numpy.float64  13x13x14 lipid density data
  frames        numpy.int64    list of frame ids
  bbs           numpy.float32  184x3 xyz coordinates for 
                               each of the 184 RAS backbone beads
  probs         numpy.float64  probability to be in each of the three states

  Notes: 
    1. Adam talks about 'frames,' which is equivalent, in lbannese, to 'sample'
    2. the "frames" field is simply a set of sequential integers, 0 .. 539

==========================================================================
#endif

  // two hacks follow. I call them hack, because they wouldn't be needed
  // if we were using sample lists. 

  // hack: re-build the data store's owner map
  rebuild_data_store_owner_map();

  // hack: get the data_ids that this rank owns
  std::unordered_map<int, std::vector<std::pair<int,int>>> my_samples;
  get_my_indices(my_samples);

  //XX bool verbose = options::get()->get_bool("verbose");

  std::ofstream out;
  if (is_master() && options::get()->has_string("pilot2_profile")) {
    out.open(options::get()->get_string("pilot2_profile").c_str());
    if (!out) {
      LBANN_ERROR("failed to open ", options::get()->get_string("pilot2_profile"), " for writing");
    }
  }

#if 0
  std::vector<double> min;
  std::vector<double> max_min;
  std::vector<double> mean;
  std::vector<double> std_dev;
  bool min_max = false;
  bool z_score = false;
  if (options::get()->has_string("normalization")) {
    min_max = true;
    z_score = options::get()->get_bool("z_score");
    if (is_master()) {
      if (z_score) {
        std::cout << "Normalizing data using z-score" << std::endl;
      } else {
        std::cout << "Normalizing data using min-max" << std::endl;
      }
    }

    std::string fn = options::get()->get_string("normalization");
    std::ifstream in(fn.c_str());
    if (!in) {
      LBANN_ERROR("failed to open ", fn, " for reading");
    }
    std::string line;
    getline(in, line);
    max_min.reserve(14);
    min.reserve(14);
    mean.reserve(14);
    std_dev.reserve(14);
    double v_max, v_min, v_mean, v_std_dev;
    while (in >> v_max >> v_min >> v_mean >> v_std_dev) { 
      min.push_back(v_min);
      max_min.push_back(v_max - v_min);
      mean.push_back(v_mean);
      std_dev.push_back(v_std_dev);
    }
    in.close();
    if (min.size() != 14) {
      LBANN_ERROR("normalization.size() = ", min.size(), "; should be 14");
    }
  } else {
    if (is_master()) {
      std::cout << "NOT Normalizing data!" << std::endl;
    }
  }
#endif

  // construct a conduit::Node for each sample that this rank owns,
  // and set it in the data_store
  //XX size_t nn = 0;
  std::vector<size_t> dist(3, 0);
#if 0
XX
  for (const auto &t : my_samples) {
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(m_filenames[t.first]);
    for (const auto &t4 : t.second) {
      int data_id = t4.first;
      int sample_index = t4.second;
      conduit::Node &node = m_data_store->get_empty_node(data_id);

      size_t offset;
      for (const auto &t5 : m_datum_shapes) {
        const std::string &name = t5.first;
        // this could be done better ... read the choices of fields
        // to use from file, as is done in data_reader_jag_conduit?

        if (name == "frames") {
          conduit::int64 *data = reinterpret_cast<conduit::int64*>(a[name].data_holder->data());
          offset = sample_index*m_datum_num_words["frames"];
          node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
        } 

        else if (name == "bbs") {
          conduit::float32 *data = reinterpret_cast<conduit::float32*>(a[name].data_holder->data());
          offset = sample_index*m_datum_num_words["bbs"];
          node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
        } 

        else { // rots, states, tilts, density_sig1, probs
          offset = sample_index*m_datum_num_words[name];
          conduit::float64 *data = reinterpret_cast<conduit::float64*>(a[name].data_holder->data());

          if (name == "states") {
            int label = (data + offset)[0];
            if (label < 0 || label > 2) {
              LBANN_ERROR("bad label; should be 0, 1, or 2 but it's: ", label);
            }
            dist[label] += 1;
            node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(label);

          } else if (name == "density_sig1") {
            int s = 0;
            if (z_score) {
              for (size_t j=offset; j<offset+m_datum_num_words[name]; j++) {
                data[j]= (data[j] - mean[s]) / std_dev[s];
                ++s;
                if (s == 14) {
                  s = 0;
                }
              }
            } else if (min_max) {
              for (size_t j=offset; j<offset+m_datum_num_words[name]; j++) {
                data[j] = (data[j] - min[s]) / max_min[s];
                ++s;
                if (s == 14) {
                  s = 0;
                }
              }
            }
            node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);

            if (out) {
              for (size_t j=offset; j<offset+m_datum_num_words[name]; j++) {
                out << data[j] << std::endl;
              }
            }
          
          // rots, tilts, probs
          } else {
            node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
          }  
        }
      }

      m_data_store->set_preloaded_conduit_node(data_id, node);

      //user feedback
      ++nn;
      if (verbose && is_master() && nn % 1000 == 0) {
        int np = m_comm->get_procs_per_trainer();
        std::cout << "estimated number of samples loaded: " << utils::commify(nn/1000*np) << "K" << std::endl;
      }  
    }
  }
#endif

  if (out) {
    out.close();
  }

  //user feedback
  if (is_master()) {
    std::vector<size_t> r(3);
    m_comm->trainer_reduce(dist.data(), 3, r.data());
    std::cout << "\nLabel distribution:\n";
    for (size_t h=0; h<3; h++) {
      std::cout << "  " << h << " " << r[h] << std::endl;
    }
    std::cout << "\nData Shapes:\n";
    for (auto t : m_datum_shapes) {
      std::cout << "  " << t.first << " ";
      for (auto t2 : t.second) {
        std::cout << t2 << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  } else {
    m_comm->trainer_reduce(dist.data(), 3, 0);
  }
}

bool pilot2_conduit_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  double scaling_factor = 1.0;
  const double *data = node[LBANN_DATA_ID_STR(data_id) + "/density_sig1"].value();
  size_t n = m_datum_num_words["density_sig1"];
  for (size_t j = 0; j < n; ++j) {
    X(j, mb_idx) = data[j] * scaling_factor;
  }  

#if 0
Notes from Adam:
The keras model that I gave you only looks at the density_sig1 data as input data and it uses the states data as labels.  We¿ll want to also extract bbs to merge that with density_sig1 in various ways as input data in future models that we¿re putting together.

 The probs field can be useful as an alternate label if building a regression model instead of a classification model.  I¿ve also been using the probs field as a filter on the training data to only consider those input data whose state probability exceeds some threshold.

  So that works out to:

   bb, density_sig1 - datum
   states           - label
   probs            - used as a filter to include/exclude certain samples

#endif
  return true;
}

bool pilot2_conduit_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  int label = node[LBANN_DATA_ID_STR(data_id) + "/states"].value();
  Y.Set(label, mb_idx, 1);
  return true;
}

bool pilot2_conduit_data_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("pilot2_conduit_data_reader: do not have responses");
  return true;
}

void pilot2_conduit_data_reader::fill_in_metadata() {
  int index = 0;
  const sample_t& s = m_sample_list[index];
  const std::string& sample_name = s.second;
//XX
std::cout << " sample name: " << sample_name << std::endl;
  sample_file_id_t id = s.first;
std::cout << " id: " << id << std::endl;
  m_sample_list.open_samples_file_handle(index, true);
  auto h = m_sample_list.get_samples_file_handle(id);
  conduit::Node node;
  const std::string path = "/";
  read_node(h, path, node);
node.print();
exit(0);

#if 0 
XX
  std::map<std::string, cnpy::NpyArray> aa = cnpy::npz_load(m_filenames[0]);
  for (const auto &t : aa) {
    const std::string &name = t.first;
    size_t word_size = t.second.word_size;
    const std::vector<size_t> &shape = t.second.shape;
    size_t num_words = 1;
    if (shape.size() == 1) {
      m_datum_shapes[name].push_back(1);
    } else {
      for (size_t x=1; x<shape.size(); x++) {
        num_words *= shape[x];
        m_datum_shapes[name].push_back(shape[x]);
      }
    }
    m_datum_num_words[name] = num_words;
    m_datum_word_sizes[name] = word_size;
    m_datum_num_bytes[name] = num_words*word_size;
  }

  //TODO: this should be more generic, will need to change depending on what we fetch
  if (m_datum_shapes.find("density_sig1") == m_datum_shapes.end()) {
    LBANN_ERROR("m_datum_shapes.find(\"density_sig1\") = m_datum_shapes.end()");
  }
  m_num_features = 1;
  for (auto t : m_datum_shapes["density_sig1"]) {
    m_data_dims.push_back(t);
    m_num_features *= t;
  }
#endif
}

void pilot2_conduit_data_reader::get_my_indices(std::unordered_map<int, std::vector<std::pair<int,int>>> &my_samples) {
  std::unordered_set<size_t> indices;
  for (const auto &t : m_shuffled_indices) {
    indices.insert(t);
  }
  int my_rank = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();
  size_t data_id = 0;
  for (size_t j=0; j<m_filenames.size(); ++j) {
int x = 0;
    int file_owner = j % np;
    for (int h=0; h<m_samples_per_file[j]; h++) {
      if (indices.find(data_id) != indices.end()) {
        if (file_owner == my_rank) {
          my_samples[j].push_back(std::make_pair(data_id, h));
++x;
        }
      }
      ++data_id;
    }
  }
}

void pilot2_conduit_data_reader::rebuild_data_store_owner_map() {
  m_data_store->clear_owner_map();
  int np = m_comm->get_procs_per_trainer();
  size_t data_id = 0;
  for (size_t j=0; j<m_filenames.size(); ++j) {
    int file_owner = j % np;
    for (int h=0; h<m_samples_per_file[j]; h++) {
      m_data_store->add_owner(data_id, file_owner);
      ++data_id;
    }
  }
}

void pilot2_conduit_data_reader::get_samples_per_file() {
#if 0
  int me = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();
  std::vector<int> work;
  int x = 0;
  for (size_t j=me; j<m_filenames.size(); j+=np) {
    ++x;
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(m_filenames[j]);
    size_t n = 0;
    for (const auto &t2 : a) {
      size_t n2 = t2.second.shape[0];
      if (n == 0) {
        n = n2;
      } else {
        if (n2 != n) {
          LBANN_ERROR("n2 != n; ", n2, n);
        }
      }
    }
    work.push_back(j);
    work.push_back(n);
  }

  std::vector<int> num_files(np, 0);
  for (size_t j=0; j<m_filenames.size(); ++j) {
    int owner = j % np;
    num_files[owner] += 1;
  }

  m_samples_per_file.resize(m_filenames.size());
  std::vector<int> work_2;
  std::vector<int> *work_ptr;
  for (int j=0; j<np; j++) {
    if (me == j) {
      work_ptr = &work;
    } else {
      work_2.resize(num_files[j]*2);
      work_ptr = &work_2;
    }
    m_comm->trainer_broadcast<int>(j, work_ptr->data(), work_ptr->size());
    for (size_t h=0; h<work_ptr->size(); h+= 2) {
      m_samples_per_file[(*work_ptr)[h]] = (*work_ptr)[h+1];
    }
  }
#endif
}

void pilot2_conduit_data_reader::write_file_sizes() {
  if (! is_master()) {
    return;
  }
  std::string fn = options::get()->get_string("pilot2_save_file_sizes");
  std::ofstream out(fn.c_str());
  if (!out) {
    LBANN_ERROR("failed to open ", fn, " for writing");
  }
  for (size_t j=0; j<m_samples_per_file.size(); j++) {
    out << m_filenames[j] << " " << m_samples_per_file[j] << std::endl;
  }
  out.close();
}

void pilot2_conduit_data_reader::read_file_sizes() {
  std::string fn = options::get()->get_string("pilot2_read_file_sizes");
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", fn, " for reading");
  }
  std::unordered_map<std::string, int> mp;
  std::string filename;
  int num_samples;
  while (in >> filename >> num_samples) {
    mp[filename] = num_samples;
  }
  in.close();

  m_samples_per_file.resize(m_filenames.size());
  for (size_t h=0; h<m_filenames.size(); h++) {
    if (mp.find(m_filenames[h]) == mp.end()) {
      LBANN_ERROR("failed to find filename '", m_filenames[h], "' in the map");
    }
    m_samples_per_file[h] = mp[m_filenames[h]];
  }
}

#ifdef _USE_IO_HANDLE_
bool pilot2_conduit_data_reader::has_path(const data_reader_jag_conduit::file_handle_t& h,
                                       const std::string& path) const {
  return m_sample_list.is_file_handle_valid(h) && h->has_path(path);
}

void pilot2_conduit_data_reader::read_node(const data_reader_jag_conduit::file_handle_t& h,
                                        const std::string& path,
                                        conduit::Node& n) const {
  if (!h) {
    return;
  }
  h->read(path, n);
}
#else
bool pilot2_conduit_data_reader::has_path(const hid_t& h, const std::string& path) const {
  return (m_sample_list.is_file_handle_valid(h) &&
          conduit::relay::io::hdf5_has_path(h, path));
}

void pilot2_conduit_data_reader::read_node(const hid_t& h, const std::string& path, conduit::Node& n) const {
  conduit::relay::io::hdf5_read(h, path, n);
}
#endif //#ifdef _USE_IO_HANDLE_

}  // namespace lbann
