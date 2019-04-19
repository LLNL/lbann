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
#include "lbann/data_readers/numpy_conduit_cache.hpp"
#include "lbann/utils/file_utils.hpp"
#include <cstdio>
#include <string>
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
  //TODO
}

void numpy_npz_conduit_reader::load() {
  if(is_master()) {
    std::cout << "starting load" << std::endl;
  }

  // for a first draft, this reader only works with a pre-loaded data store
  m_data_store = new data_store_conduit(this);

  //dah - for now, I assume the input file contains, on each line, the name
  //      of an npz file. This will no doubt change in the future.
  //      I'd like to call load_list_of_samples(), but the sample_list class
  //      is too specialized -- it checks data in a manner particular to 
  //      JAG data, and that doesn't apply here.

  std::string infile = get_data_filename();
  std::ifstream ifs(infile);
  if (!ifs) {
    LBANN_ERROR("numpy_npz_conduit_reader::load() - can't open file : " + infile);
  }

  int rank = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();

  std::string npz_filename;
  m_num_samples = 0;
  bool first = true;
  std::unordered_set<int> label_classes;
  while (getline(ifs, npz_filename)) {
    if (npz_filename.size() > 2) {
      if (m_num_samples % np == rank) {
        conduit::Node &node = m_data_store->get_empty_node(m_num_samples);
        numpy_conduit_cache::load_conduit_node(npz_filename, node, m_num_samples);

        // things that only need to be node for a single sample
        if (first) {
          //fill in m_data_dims
          auto shape = node[std::to_string(m_num_samples) + "/data/shape"].as_uint64_array();
          int shape_num_elts = shape.number_of_elements();
          for (int k=0; k<shape_num_elts; k++) {
            m_data_dims.push_back(shape[k]);
          }
          // Ensure we understand the word sizes
          int word_size = node[std::to_string(m_num_samples) + "/data/word_size"].value();
          if (!(word_size == 2 || word_size == 4 || word_size == 8)) {
            LBANN_ERROR("numpy_npz_conduit_reader: word size " + 
                        std::to_string(word_size) + " not supported");
          }
          m_data_word_size = word_size;
          if (m_has_labels) {
            word_size = node[std::to_string(m_num_samples) + "/frm/word_size"].value();
            if (word_size != 4) {
              LBANN_ERROR("numpy_npz_conduit_reader: label numpy array should be in int32");
            }
          }
          first = false;
        }

        if (m_has_labels) {
          int *label = const_cast<int*>(node[std::to_string(m_num_samples) + "/frm/data"].value());
          label_classes.insert(*label);
        }

        m_data_store->set_conduit_node(m_num_samples, node);
      }
      ++m_num_samples;
    }
  }
  ifs.close();

  //TODO: need to all-reduce label_classes
  if (m_has_labels) {
    m_num_labels = label_classes.size();

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

  if (m_data_word_size == 2) {
    // Convert int16 to DataType.
    const short *data = const_cast<short*>(char_data);
    DataType *dest = X_v.Buffer();

    // OPTIMIZE
    LBANN_OMP_PARALLEL_FOR
      for(int j = 0; j < m_num_features; j++)
        dest[j] = data[j] * m_scaling_factor_int16;

  } else {
    void *data = NULL;
    if (m_data_word_size == 4) {
      const float *data = reinterpret_cast<float*>(char_data);
    } else if (m_data_word_size == 8) {
      const double *data = const_cast<double*>(char_data);
    }
    std::memcpy(X_v.Buffer(), data, m_num_features * m_data.word_size);
  }
  return true;
}

bool numpy_npz_conduit_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  if (!m_has_labels) {
    LBANN_ERROR("numpy_npz_conduit_reader: do not have labels");
  }
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  const std::string data_id_str = pad(std::to_string(data_id), SAMPLE_ID_PAD, '0');
  char *char_data = node[data_id_str + "/data/data"].value();
  int *label = const_cast<int*>(char_data);
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
