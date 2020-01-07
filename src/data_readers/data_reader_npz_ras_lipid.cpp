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

#include "lbann/data_readers/data_reader_npz_ras_lipid.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include <unordered_set>
#include "lbann/utils/file_utils.hpp" // pad()
#include "lbann/utils/jag_utils.hpp"  // read_filelist(..) TODO should be move to file_utils
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/lbann_library.hpp"

#undef DEBUG
#define DEBUG

namespace lbann {

ras_lipid_conduit_data_reader::ras_lipid_conduit_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

ras_lipid_conduit_data_reader::ras_lipid_conduit_data_reader(const ras_lipid_conduit_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

ras_lipid_conduit_data_reader& ras_lipid_conduit_data_reader::operator=(const ras_lipid_conduit_data_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void ras_lipid_conduit_data_reader::copy_members(const ras_lipid_conduit_data_reader &rhs) {
  if (is_master()) {
    std::cout << "Starting ras_lipid_conduit_data_reader::copy_members\n";
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

void ras_lipid_conduit_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();
  opts->set_option("preload_data_store", 1);

  // Error check settings for validation percent, etc
  size_t count = get_absolute_sample_count();
  //TODO ???
  if (count) {
    LBANN_ERROR("You cannot use absolute sample count with this data reader");
  }
  double use_percent = get_use_percent();
  if (use_percent != 1) {
    LBANN_ERROR("use_percent for < 1.0 is not yet implemented; please contact Dave Hysom");
  }
  if (m_validation_percent) {
    LBANN_ERROR("validation percent is not yet implemented; please contact Dave Hysom");
  }

  // The input file should contain, on each line, the complete
  // pathname of an npz file. 
  std::string infile = get_data_filename();
  read_filelist(m_comm, infile, m_filenames);

  // Read or compute the number of samples per file (this is the number
  // of samples before we sequentially-concatenate them)
  if (opts->has_string("pilot2_read_file_sizes")) {
    read_file_sizes();
  } else {
    double tm3 = get_time();
    get_samples_per_file();
    if (is_master()) std::cout << "time to compute samples_per_file: " << get_time() - tm3 << std::endl;
  }
  // Optionally save the samples-per-file info to file
  if (opts->has_string("pilot2_save_file_sizes")) {
    write_file_sizes();
  }

  // Get the number of samples that will be combined into a multi-sample
  m_seq_len = 1;
  if (opts->has_int("seq_len")) {
    m_seq_len = opts->get_int("seq_len");
  }

  // Get the number of global multi-samples, and the number of
  // multi-samples in each file
  m_multi_samples_per_file.reserve(m_filenames.size());
  m_num_global_indices = 0;
  for (const auto &t : m_samples_per_file) {
    int n = t / m_seq_len; // this is the number of multi-samples
    m_multi_samples_per_file.push_back(n);
    m_num_global_indices += n;
  }
  m_train_indices = m_num_global_indices;
  m_validate_indices = 0; //TODO

  fill_in_metadata();

  // Compute the data_id of the first sample in each file
  m_first_multi_id_per_file[0] = 0;
  for (size_t j=1; j<m_samples_per_file.size()+1; j++) {
    m_first_multi_id_per_file[j] = (m_first_multi_id_per_file[j-1] + m_multi_samples_per_file[j-1]);
  }

  // user feedback
  print_shapes_etc();

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_global_indices);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  instantiate_data_store();
  select_subset_of_data();
}

void ras_lipid_conduit_data_reader::do_preload_data_store() {
  if (is_master()) std::cout << "Starting ras_lipid_conduit_data_reader::do_preload_data_store; num indices: " << utils::commify(m_shuffled_indices.size()) << std::endl;

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

  // get normalization data
  read_normalization_data();

  // get the data_ids that this rank owns;
  // my_samples maps m_filenames index -> (data_id, index in the file).
  // Note: these are data_ids before we combine them into multi-samles
  std::map<int, std::vector<std::pair<int,int>>> my_samples;
  get_my_indices(my_samples);

  // Variables relating to user feedback; otherwise NA
  bool verbose = options::get()->get_bool("verbose");
  int np = m_comm->get_procs_per_trainer();
  size_t nn = 0; 

  std::vector<conduit::Node> work(m_seq_len);

  // Loop over the files owned by this processer
  for (const auto &t : my_samples) {

    // Load the next data file
    int file_id = t.first;
    std::map<std::string, cnpy::NpyArray> data = cnpy::npz_load(m_filenames[file_id]);
    int multi_data_id = m_first_multi_id_per_file[t.first];

    // loop over the samples in the data file
    int count = 0; // counter for number of samples in a multi-node
    int count_cumulative = 0; // counter for total number of samples in this file
    for (const auto &t4 : t.second) {
      int data_id = t4.first;
      int sample_index = t4.second;

      load_the_next_sample(work[count], data_id, sample_index, data);
      ++count;
      ++count_cumulative;

      //user feedback
      ++nn;
      if (verbose && is_master() && nn % 1000 == 0) {
        std::cout << "estimated number of samples processed: " 
                  << utils::commify(nn/1000*np) << "K" << std::endl;
      }  
      // Construct the multi-node (if seq_len > 1) and put the node 
      // in the data store
      if (count == m_seq_len) {
        if (m_seq_len == 1 && options::get()->get_bool("direct")) {
          m_data_store->set_conduit_node(multi_data_id, work[0]);
        }

        else {

        // get pointers to the children, i.e, bypass the encoded data_id
        std::vector<const conduit::Node*> work_ptr(m_seq_len);
        for (int h=0; h<m_seq_len; h++) {
          work_ptr[h]  = work[h].child_ptr(0);
        }

        conduit::Node n3;
        std::vector<double> work_d;
        std::vector<float> work_f;
        for (const auto &t42 : m_datum_num_words) {
          const std::string &name = t42.first;
          if (name == "frames") {
            continue;
          }
          int n_words = t42.second;
          if (name == "bbs") {
            work_f.resize(m_seq_len*n_words);
            int offset = 0;
            for (const auto &t5 : work_ptr) {
              const float *d = (*t5)[name].value();
              for (size_t u=0; u<m_datum_num_words[name]; u++) {
                work_f[offset++] = d[u];
              }
            }
            n3[LBANN_DATA_ID_STR(multi_data_id) + "/" + name].set(work_f.data(), m_seq_len * m_datum_num_words[name]);
          } else {
            work_d.resize(m_seq_len*n_words);
            int offset = 0;
            for (const auto &t5 : work_ptr) {
              const double *d = (*t5)[name].value();
              for (size_t u=0; u<m_datum_num_words[name]; u++) {
                work_d[offset++] = d[u];
              }
            }
            n3[LBANN_DATA_ID_STR(multi_data_id) + "/" + name].set(work_d.data(), m_seq_len * m_datum_num_words[name]);
          }
        }
        m_data_store->set_conduit_node(multi_data_id, n3);

        }

        count = 0;
        for (auto &t5 : work) {
          t5.reset();
        }
        ++multi_data_id;
      }

      if (count / m_seq_len == m_multi_samples_per_file[t.first]) {
        break;
      }

    }
  }

  //TODO: relook
  //m_data_store->exchange_owner_maps();
  std::unordered_map<int, int> owners;
  for (size_t j=0; j<m_multi_samples_per_file.size(); j++) {
    int owning_rank = j % np;
    for (int h=m_first_multi_id_per_file[j]; h<m_first_multi_id_per_file[j]+m_multi_samples_per_file[j]; h++) {
      owners[h] = owning_rank;
    }
  }
  m_data_store->set_preloaded_owner_map(owners);
}

bool ras_lipid_conduit_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  double scaling_factor = 1.0;
  const double *data = node[LBANN_DATA_ID_STR(data_id) + "/density_sig1"].value();

  size_t n = m_seq_len*m_datum_num_words["density_sig1"];
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

std::map<double,int> m2;

bool ras_lipid_conduit_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  const double *label = node[LBANN_DATA_ID_STR(data_id) + "/states"].value();
  size_t n = m_seq_len*m_datum_num_words["states"];

  int label2 = (int) label[0];

  for (size_t j = 0; j < n; ++j) {
    Y.Set(label2, mb_idx, 1);
  }
  //int label = node[LBANN_DATA_ID_STR(data_id) + "/states"].value();
  //Y.Set(label, mb_idx, 1);
  return true;
}

bool ras_lipid_conduit_data_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("ras_lipid_conduit_data_reader: do not have responses");
  return true;
}

void ras_lipid_conduit_data_reader::fill_in_metadata() {
  std::map<std::string, cnpy::NpyArray> aa = cnpy::npz_load(m_filenames[0]);
  for (const auto &t : aa) {
    const std::string &name = t.first;
    size_t word_size = t.second.word_size;
    const std::vector<size_t> &shape = t.second.shape;
    size_t num_words = 1;
    //size_t num_words = m_seq_len;
    if (shape.size() == 1) {
      m_datum_shapes[name].push_back(1*m_seq_len);
    } else {
      m_datum_shapes[name].push_back(m_seq_len);
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
}

void ras_lipid_conduit_data_reader::get_my_indices(std::map<int, std::vector<std::pair<int,int>>> &my_samples) {
  int my_rank = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();
  size_t data_id = 0;
  for (size_t j=0; j<m_filenames.size(); ++j) {
    int file_owner = j % np;
    for (int h=0; h<m_samples_per_file[j]; h++) {
      if (file_owner == my_rank) {
        my_samples[j].push_back(std::make_pair(data_id, h));
      }
      ++data_id;
    }
  }
}

void ras_lipid_conduit_data_reader::rebuild_data_store_owner_map() {
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
  m_data_store->set_finished_building_map();
}

void ras_lipid_conduit_data_reader::get_samples_per_file() {
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
}

void ras_lipid_conduit_data_reader::write_file_sizes() {
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

void ras_lipid_conduit_data_reader::read_file_sizes() {
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

void ras_lipid_conduit_data_reader::read_normalization_data() {
  m_use_min_max = false;
  m_use_z_score = false;
  if (options::get()->has_string("normalization")) {
   m_use_min_max = true;
    m_use_z_score = options::get()->get_bool("z_score");
    if (is_master()) {
      if (m_use_z_score) {
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
    m_max_min.reserve(14);
    m_min.reserve(14);
    m_mean.reserve(14);
    m_std_dev.reserve(14);
    double v_max, v_min, v_mean, v_std_dev;
    while (in >> v_max >> v_min >> v_mean >> v_std_dev) { 
      m_min.push_back(v_min);
      m_max_min.push_back(v_max - v_min);
      m_mean.push_back(v_mean);
      m_std_dev.push_back(v_std_dev);
    }
    in.close();
    if (m_min.size() != 14) {
      LBANN_ERROR("normalization.size() = ", m_min.size(), "; should be 14");
    }
  } else {
    if (is_master()) {
      std::cout << "NOT Normalizing data!" << std::endl;
    }
  }
}

void ras_lipid_conduit_data_reader::print_shapes_etc() {
  //user feedback
  if (!is_master()) {
    return;
  }
  
  std::cout << "\n======================================================\n";
  std::cout << "Role: " << get_role() << std::endl; 
  std::cout << "Sequence Length: " << m_seq_len << std::endl;
  std::cout << "Num samples: " << m_train_indices << std::endl;
  std::cout << "\nData Shapes:\n";
  for (auto t : m_datum_shapes) {
    std::cout << "  " << t.first << " ";
    for (auto t2 : t.second) {
      std::cout << t2 << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "\nlinearized data size: " << get_linearized_data_size() << "\n"
            << "linearized label size: " << get_linearized_label_size() << "\n"
            << "num labels: " << get_num_labels() << "\n"
            << "data dims: ";
  for (auto t : m_data_dims) std::cout << t << " ";
  std::cout << std::endl;
  std::cout << "======================================================\n\n";
  
  /*
    TODO: label distribution
    std::vector<size_t> r(3);
    m_comm->trainer_reduce(dist.data(), 3, r.data());
    std::cout << "\nLabel distribution:\n";
    for (size_t h=0; h<3; h++) {
      std::cout << "  " << h << " " << r[h] << std::endl;
    }
  else {
    m_comm->trainer_reduce(dist.data(), 3, 0);
  }
    */
} 

void ras_lipid_conduit_data_reader::load_the_next_sample(conduit::Node &node, int data_id, int sample_index, std::map<std::string, cnpy::NpyArray> &a) {
  size_t offset;
  for (const auto &t5 : m_datum_shapes) {
    const std::string &name = t5.first;
    if (name == "bbs") {
      conduit::float32 *data = reinterpret_cast<conduit::float32*>(a[name].data_holder->data());
      offset = sample_index*m_datum_num_words["bbs"];
      node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
    } 

    else { // rots, states, tilts, density_sig1, probs
      offset = sample_index*m_datum_num_words[name];
      conduit::float64 *data = reinterpret_cast<conduit::float64*>(a[name].data_holder->data());

      if (name == "states") {
        node[LBANN_DATA_ID_STR(data_id) + "/states"].set(data + offset, m_datum_num_words[name]);
        /*
        int label = (data + offset)[0];
        if (label < 0 || label > 2) {
          LBANN_ERROR("bad label; should be 0, 1, or 2 but it's: ", label);
        }
        node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(label);
        */

      } else if (name == "density_sig1") {
        int s = 0;
        if (m_use_z_score) {
          for (size_t j=offset; j<offset+m_datum_num_words[name]; j++) {
            data[j]= (data[j] - m_mean[s]) / m_std_dev[s];
            ++s;
            if (s == 14) {
              s = 0;
            }
          }
        } else if (m_use_min_max) {
          for (size_t j=offset; j<offset+m_datum_num_words[name]; j++) {
            data[j] = (data[j] - m_min[s]) / m_max_min[s];
            ++s;
            if (s == 14) {
              s = 0;
            }
          }
        }
        node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);

      // rots, tilts, probs
      } else {
        node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
      }  
    }
  }
}


}  // namespace lbann
