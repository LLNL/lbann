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

#include "lbann/data_readers/data_reader_numpy_npz_conduit.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/data_readers/numpy_conduit_converter.hpp"
#include "lbann/utils/file_utils.hpp"
//#include <cstdio>
#include <unordered_set>
#include <cnpy.h>

// ugh; duplicate from data_reader_jag_conduit;
// also duplicated in numpy_conduit_cache class
#ifdef SAMPLE_ID_PAD
#undef SAMPLE_ID_PAD
#endif
#define SAMPLE_ID_PAD 9

namespace lbann {

numpy_npz_conduit_reader::numpy_npz_conduit_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

numpy_npz_conduit_reader::numpy_npz_conduit_reader(const numpy_npz_conduit_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

numpy_npz_conduit_reader& numpy_npz_conduit_reader::operator=(const numpy_npz_conduit_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void numpy_npz_conduit_reader::copy_members(const numpy_npz_conduit_reader &rhs) {
  LBANN_ERROR("not implemented - TODO");
}

void numpy_npz_conduit_reader::load() {
  if(is_master()) {
    std::cout << "starting load" << std::endl;
  }

  options *opts = options::get();

  // for a first draft, this reader only works with a pre-loaded data store
  opts->set_option("preload_data_store", 1);
  opts->set_option("use_data_store", 1);
  // for a first draft, this reader only works with a pre-loaded data store

  //dah - for now, I assume the input file contains, on each line, the name
  //      of an npz file. This will no doubt change in the future.
  //      I'd like to call load_list_of_samples(), but the sample_list class
  //      is too specialized -- it checks data in a manner particular to 
  //      conduit, and that doesn't apply here.

  std::string infile = get_data_filename();
  std::ifstream ifs(infile);
  if (!ifs) {
    LBANN_ERROR("numpy_npz_conduit_reader::load() - can't open file : " + infile);
  }

  int rank = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();

  // get my_files, data_ids, and local_list_sizes
  std::string npz_filename;
  int data_id = 0;
  std::vector<int> local_list_sizes(np, 0);
  while (getline(ifs, npz_filename)) {
    if (npz_filename.size() > 2) {
      int owner = m_num_samples % np;
      local_list_sizes[owner] += 1;
      if (owner == rank) {
        m_my_files.push_back(npz_filename);
        m_my_data_ids.push_back(data_id);
      }
      ++data_id;
    }
  }
  ifs.close();
  m_num_samples = data_id;

  instantiate_data_store(local_list_sizes);

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

void numpy_npz_conduit_reader::preload() {
  bool first = true;
  std::unordered_set<int> label_classes;
  for (size_t j=0; j<m_my_files.size(); j++) {
    const std::string &npz_filename = m_my_files[j];
    const int &data_id = m_my_data_ids[j];
if (m_master) std::cerr << "attempting to load: " << npz_filename << "\n";
        conduit::Node node;
        numpy_conduit_converter::load_conduit_node(npz_filename, data_id, node);

  // note: in the following block "m_num_samples" plays the role of data_id
  m_num_samples = 0;

  std::string npz_filename;
  bool first = true;
  std::unordered_set<int> label_classes;
  while (getline(ifs, npz_filename)) {
    if (npz_filename.size() > 2) {
      if (m_num_samples % np == rank) {
        conduit::Node &node = m_data_store->get_empty_node(m_num_samples);
        numpy_conduit_cache::load_conduit_node(npz_filename, m_num_samples, node);

        // things that only need to be node for a single sample
        if (first) {
          //fill in m_data_dims
          auto shape = node[std::to_string(m_num_samples) + "/data/shape"].as_uint64_array();
          int shape_num_elts = shape.number_of_elements();
          for (int k=0; k<shape_num_elts; k++) {
            m_data_dims.push_back(shape[k]);
          }
          // Ensure we understand the word sizes
          size_t word_size = node[std::to_string(m_num_samples) + "/data/word_size"].value();
          if (!(word_size == 2 || word_size == 4 || word_size == 8)) {
            LBANN_ERROR("numpy_npz_conduit_reader: word size " + 
                        std::to_string(word_size) + " not supported");
          }
          m_data_word_size = word_size;
          if (m_has_labels) {
            word_size = node[std::to_string(m_num_samples) + "/frm/word_size"].value();
            /*
            if (word_size != 4) {
              LBANN_ERROR("numpy_npz_conduit_reader: label numpy array should be in int32, but word_size= " + std::to_string(word_size));
            }
            */
          }
          first = false;
        } // end, things that only need to be node for a single sample
            if (word_size != 4) {
              LBANN_ERROR("numpy_npz_conduit_reader: label numpy array should be in int32");
            }
          }
          first = false;
        }

        if (m_has_labels) {
          char *char_data = node[std::to_string(m_num_samples) + "/frm/data"].value();
          int *label = reinterpret_cast<int*>(char_data);
          label_classes.insert(*label);
        }

std::cerr << "calling set_conduit_node for data id: " << data_id << " role: " << get_role() << "\n";
        m_data_store->set_conduit_node(data_id, node);
      }
    }
  }
    }
  }
  ifs.close();

  //TODO: need to all-reduce label_classes
  if (m_has_labels) {
    m_num_labels = label_classes.size();

    if (is_master()) {
      std::cout << "num labels: " << m_num_labels << "\n";
    }

    // Sanity checks.
    auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
    if (*minmax.first != 0) {
      LBANN_ERROR("numpy_reader: classes are not indexed from 0");
    }
    if (*minmax.second != (int) label_classes.size() - 1) {
      LBANN_ERROR("numpy_reader: label classes are not contiguous");
    }
    m_num_labels = label_classes.size();
  }

  m_num_features = std::accumulate(m_data_dims.begin() + 1,
                                   m_data_dims.end(),
                                   (unsigned) 1,
                                   std::multiplies<unsigned>());

  /* TODO: revisit; for now, we don't work with responses
  if(m_has_responses) {
    m_num_response_features = std::accumulate(m_responses.shape.begin() + 1,
                                              m_responses.shape.end(),
                                              (unsigned) 1,
                                              std::multiplies<unsigned>());
  }
  */

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool numpy_npz_conduit_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  Mat X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));

  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  const std::string data_id_str = pad(std::to_string(data_id), SAMPLE_ID_PAD, '0');
  const char *char_data = node[data_id_str + "/data/data"].value();
  char *char_data_2 = const_cast<char*>(char_data);

  if (m_data_word_size == 2) {
    // Convert int16 to DataType.
    short *data = reinterpret_cast<short*>(char_data_2);
    DataType *dest = X_v.Buffer();

    // OPTIMIZE
    LBANN_OMP_PARALLEL_FOR
      for(int j = 0; j < m_num_features; j++)
        dest[j] = data[j] * m_scaling_factor_int16;

  } else {
    void *data = NULL;
    if (m_data_word_size == 4) {
      float *f = reinterpret_cast<float*>(char_data_2);
      data = (void*)(f + data_id * m_num_features);
    } else if (m_data_word_size == 8) {
      double *d = reinterpret_cast<double*>(char_data_2);
      data = (void*)(d + data_id * m_num_features);
    }
    std::memcpy(X_v.Buffer(), data, m_num_features * m_data_word_size);
  }
  return true;
}

bool numpy_npz_conduit_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  if (!m_has_labels) {
    LBANN_ERROR("numpy_npz_conduit_reader: do not have labels");
  }
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  const std::string data_id_str = pad(std::to_string(data_id), SAMPLE_ID_PAD, '0');
  const char *char_data = node[data_id_str + "/data/data"].value();
  char *char_data_2 = const_cast<char*>(char_data);
  int *label = reinterpret_cast<int*>(char_data_2);
  Y(*label, mb_idx) = 1;
  return true;
}

bool numpy_npz_conduit_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  if (!m_has_responses) {
    LBANN_ERROR("numpy_npz_conduit_reader: do not have responses");
  }
  if (!m_has_responses) {
    LBANN_ERROR("not implemented");
  }
  #if 0
  void *responses = NULL;
  if (m_responses.word_size == 4) {
    responses = (void *) (m_responses.data<float>()
                          + data_id * m_num_response_features);
  } else if (m_responses.word_size == 8) {
    responses = (void *) (m_responses.data<double>()
                          + data_id * m_num_response_features);
  }
  Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx + 1));
  std::memcpy(Y_v.Buffer(), responses,
              m_num_response_features * m_responses.word_size);
  #endif
  return true;
}

}  // namespace lbann
