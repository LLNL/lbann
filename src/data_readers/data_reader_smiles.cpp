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

#include "lbann/data_readers/data_reader_smiles.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/lbann_library.hpp"

namespace lbann {

smiles_data_reader::smiles_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

smiles_data_reader::smiles_data_reader(const smiles_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

smiles_data_reader& smiles_data_reader::operator=(const smiles_data_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void smiles_data_reader::copy_members(const smiles_data_reader &rhs) {
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);
  m_linearized_data_size = rhs.m_linearized_data_size;
  m_linearized_label_size = rhs.m_linearized_label_size;
  m_linearized_response_size = rhs.m_linearized_response_size;
  m_num_labels = rhs.m_num_labels;
}

void smiles_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();
  opts->set_option("preload_data_store", 1);

  std::string infile = get_data_filename();

#if 0
  // Build owner map
  int np = m_comm->get_procs_per_trainer();
  for (size_t j=0; j<m_filenames.size(); j++) {
    int owner = j % np;
    int first = first_multi_id_per_file[j];
    for (int k=0; k<multi_samples_per_file[j]; ++k) {
      m_multi_sample_to_owner[k+first] = owner;
    }
  }

  int my_rank = m_comm->get_rank_in_trainer();

  //m_filename_to_multi_sample maps filename -> multi-sample data_ids
  //m_multi_sample_id_to_first_sample maps multi-sample data_id 
  //    -> first single-sample that is part of the multi-sample. 
  //Note: multi-sample data_id is global; single-sample data_id is 
  //      local (WRT the current file)
  
  //Note: m_filename_to_multi_sample contains all multi-samples in the file;
  //      some of these may be marked for transfer to the validation set
  //      (during select_subset_of_data)

  for (size_t j=my_rank; j<m_filenames.size(); j += np) { 
    int first_multi_sample_id = first_multi_id_per_file[j];
    int num_multi_samples = multi_samples_per_file[j];
    for (int k=0; k<num_multi_samples; k++) {
      m_filename_to_multi_sample[m_filenames[j]].insert(first_multi_sample_id+k);
      m_multi_sample_id_to_first_sample[first_multi_sample_id+k] = k*m_seq_len;
    }
  }

  fill_in_metadata();

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_global_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  instantiate_data_store();
  select_subset_of_data();

  m_num_train_samples = m_shuffled_indices.size();
  m_num_validate_samples = m_num_global_samples - m_num_train_samples;
#endif
}

void smiles_data_reader::do_preload_data_store() {
#if 0
  if (is_master()) std::cout << "starting smiles_data_reader::do_preload_data_store; num indices: " << utils::commify(m_shuffled_indices.size()) << " for role: " << get_role() << std::endl;

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

  m_data_store->set_owner_map(m_multi_sample_to_owner);

  // get normalization data
  read_normalization_data();

  // get the set of shuffled indices
  std::unordered_set<int> this_readers_indices;
  for (const auto &data_id : m_shuffled_indices) {
    this_readers_indices.insert(data_id);
  }

  // Variables only used for user feedback
  bool verbose = options::get()->get_bool("verbose");
  int np = m_comm->get_procs_per_trainer();
  size_t nn = 0; 

  std::vector<conduit::Node> work(m_seq_len);

  // option and variables only used for testing during development
  bool debug_concatenate = options::get()->get_bool("debug_concatenate");
  if (m_seq_len > 1) {
    debug_concatenate = false;
  }
  bool testme = true;

  // Determine which branch to use when forming multi-sample and inserting
  // in the data store
  int which = 2;
  if (m_seq_len == 1 && !debug_concatenate) {
    which = 1;
  }
  //TODO: fix this
  which = 2;

  // Loop over the files owned by this processer
  for (const auto &t : m_filename_to_multi_sample) {

    // Load the next data file
    std::map<std::string, cnpy::NpyArray> data = cnpy::npz_load(t.first);

    for (const auto &multi_sample_id : t.second) {
      if (this_readers_indices.find(multi_sample_id) != this_readers_indices.end()) {
        int starting_id = m_multi_sample_id_to_first_sample[multi_sample_id];

        // Load the single-samples that will be concatenated to form
        // the next multi-sample
        for (int k=0; k<m_seq_len; ++k) {
          load_the_next_sample(work[k], starting_id+k, data);

          ++nn;
          if (verbose && is_master() && nn % 1000 == 0) {
            std::cout << "estimated number of single-samples processed: "
                      << utils::commify(nn/1000*np) << "K" << std::endl;
          }
        }

        // First branch: seq_len = 1
        if (which == 1) {
          // debug block; will go away
          if (testme && is_master()) {
            std::cout << "Taking first branch (seq_len == 1)" << std::endl;
            testme = false;
          }

          work[0]["states"].value();
          m_data_store->set_conduit_node(multi_sample_id, work[0]);
        }  

        // Second branch: seq_len > 1, or seq_len = 1 and we're using this
        //        branch for debugging
        else {
          // debug block; will go away
          if (is_master() && m_seq_len == 1 && testme) {
            std::cout << "Taking second branch (seq_len == 1)" << std::endl;
            testme = false;
          }

          // Construct the multi-sample and set it in the data store
          conduit::Node n3;
          construct_multi_sample(work, multi_sample_id, n3);
          m_data_store->set_conduit_node(multi_sample_id, n3);
        }
      }
    }
  }

  // user feedback
  if (get_role() == "train") {
    print_shapes_etc();
  }
#endif
}

bool smiles_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
#if 0
  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  const double *data = node[LBANN_DATA_ID_STR(data_id) + "/density_sig1"].value();

  size_t n = m_seq_len*m_datum_num_words["density_sig1"];
  for (size_t j = 0; j < n; ++j) {
    X(j, mb_idx) = data[j];
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
#endif
  return true;
}

bool smiles_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("smiles_data_reader::fetch_label is not implemented");
  return true;
}

bool smiles_data_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("smiles_data_reader::fetch_response is not implemented");
  return true;
}


#if 0
TODO: print statistics
//user feedback
void smiles_data_reader::print_shapes_etc() {
  if (!is_master()) {
    return;
  }

  // master prints statistics
  std::cout << "\n======================================================\n";
  std::cout << "num train samples=" << m_num_train_samples << std::endl;
  std::cout << "num validate samples=" << m_num_validate_samples << std::endl;
  std::cout << "sequence length=" << m_seq_len << std::endl;
  std::cout << "num features=" << get_linearized_data_size() << std::endl;
  std::cout << "num labels=" << get_num_labels() << std::endl;
  std::cout << "data dims=";
  for (size_t h=0; h<m_datum_shapes["density_sig1"].size(); h++) {
    std::cout << m_datum_shapes["density_sig1"][h];
    if (h < m_datum_shapes["density_sig1"].size() - 1) {
      std::cout << "x";
    }
  }
  std::cout << std::endl;

  if (options::get()->get_bool("verbose_print")) {
    std::cout << "\nAll data shapes:\n";
    for (const auto &t : m_datum_shapes) {
      std::cout << "  " << t.first << " ";
      for (const auto &t2 : t.second) {
        std::cout << t2 << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }  

  std::cout << "======================================================\n\n";
}
#endif


}  // namespace lbann
