////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/data_ingestion/readers/data_reader_npz_ras_lipid.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_ingestion/data_store_conduit.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/timer.hpp"
// #include <valarray>

namespace lbann {

ras_lipid_conduit_data_reader::ras_lipid_conduit_data_reader(const bool shuffle)
  : generic_data_reader(shuffle)
{}

ras_lipid_conduit_data_reader::ras_lipid_conduit_data_reader(
  const ras_lipid_conduit_data_reader& rhs)
  : generic_data_reader(rhs)
{
  copy_members(rhs);
}

ras_lipid_conduit_data_reader& ras_lipid_conduit_data_reader::operator=(
  const ras_lipid_conduit_data_reader& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void ras_lipid_conduit_data_reader::copy_members(
  const ras_lipid_conduit_data_reader& rhs)
{
  if (rhs.m_data_store != nullptr) {
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
  m_seq_len = rhs.m_seq_len;
  m_multi_sample_to_owner = rhs.m_multi_sample_to_owner;
  m_filename_to_multi_sample = rhs.m_filename_to_multi_sample;
  m_multi_sample_id_to_first_sample = rhs.m_multi_sample_id_to_first_sample;
}

void ras_lipid_conduit_data_reader::load()
{
  if (get_comm()->am_world_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  auto& arg_parser = global_argument_parser();
  // TODO MRW
  // opts->set_option(PRELOAD_DATA_STORE, 1);

  // Error check some settings
  size_t count = get_absolute_sample_count();
  if (count) {
    LBANN_ERROR("You cannot use absolute sample count with this data reader");
  }
  double use_fraction = get_use_fraction();
  if (use_fraction != 1) {
    LBANN_ERROR("use_fraction for < 1.0 is not yet implemented; please contact "
                "Dave Hysom");
  }

  // The input file should contain, on each line, the complete
  // pathname of an npz file.
  std::string infile = get_data_filename();
  read_filelist(m_comm, infile, m_filenames);

  // Read or compute the number of samples per file (this is the number
  // of samples before we sequentially-concatenate them)
  if (arg_parser.get<std::string>("pilot2_read_file_sizes") != "") {
    read_file_sizes();
  }
  else {
    double tm3 = get_time();
    get_samples_per_file();
    if (get_comm()->am_world_master())
      std::cout << "time to compute samples_per_file: " << get_time() - tm3
                << std::endl;
  }
  // Optionally save the samples-per-file info to file
  if (arg_parser.get<std::string>("pilot2_save_file_sizes") != "") {
    write_file_sizes();
  }

  // Get the number of samples that will be combined into a multi-sample
  m_seq_len = 1;
  if (arg_parser.get<int>(LBANN_OPTION_SEQUENCE_LENGTH) != -1) {
    m_seq_len = arg_parser.get<int>(LBANN_OPTION_SEQUENCE_LENGTH);
  }

  // set the number of labels
  set_num_labels(3);

  // Get the number of global multi-samples, and the number of
  // multi-samples in each file
  std::vector<int> multi_samples_per_file;
  multi_samples_per_file.reserve(m_filenames.size());
  m_num_global_samples = 0;
  for (const auto& t : m_samples_per_file) {
    int n = t / m_seq_len; // this is the number of multi-samples
    multi_samples_per_file.push_back(n);
    m_num_global_samples += n;
  }

  // Compute the data_id of the first multi-sample in each file
  std::unordered_map<int, int> first_multi_id_per_file;
  first_multi_id_per_file[0] = 0;
  for (size_t j = 1; j < m_samples_per_file.size() + 1; j++) {
    first_multi_id_per_file[j] =
      (first_multi_id_per_file[j - 1] + multi_samples_per_file[j - 1]);
  }

  // Build owner map
  int np = m_comm->get_procs_per_trainer();
  for (size_t j = 0; j < m_filenames.size(); j++) {
    int owner = j % np;
    int first = first_multi_id_per_file[j];
    for (int k = 0; k < multi_samples_per_file[j]; ++k) {
      m_multi_sample_to_owner[k + first] = owner;
    }
  }

  int my_rank = m_comm->get_rank_in_trainer();

  // m_filename_to_multi_sample maps filename -> multi-sample data_ids
  // m_multi_sample_id_to_first_sample maps multi-sample data_id
  //     -> first single-sample that is part of the multi-sample.
  // Note: multi-sample data_id is global; single-sample data_id is
  //       local (WRT the current file)

  // Note: m_filename_to_multi_sample contains all multi-samples in the file;
  //       some of these may be marked for transfer to the validation set
  //       (during select_subset_of_data)

  for (size_t j = my_rank; j < m_filenames.size(); j += np) {
    int first_multi_sample_id = first_multi_id_per_file[j];
    int num_multi_samples = multi_samples_per_file[j];
    for (int k = 0; k < num_multi_samples; k++) {
      m_filename_to_multi_sample[m_filenames[j]].insert(first_multi_sample_id +
                                                        k);
      m_multi_sample_id_to_first_sample[first_multi_sample_id + k] =
        k * m_seq_len;
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
}

void ras_lipid_conduit_data_reader::do_preload_data_store()
{
  if (get_comm()->am_world_master())
    std::cout
      << "starting ras_lipid_conduit_data_reader::do_preload_data_store; num "
         "indices: "
      << utils::commify(m_shuffled_indices.size())
      << " for role: " << get_role() << std::endl;

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
  for (const auto& data_id : m_shuffled_indices) {
    this_readers_indices.insert(data_id);
  }

  // Variables only used for user feedback
  auto& arg_parser = global_argument_parser();
  bool verbose = arg_parser.get<bool>(LBANN_OPTION_VERBOSE);
  int np = m_comm->get_procs_per_trainer();
  size_t nn = 0;

  std::vector<conduit::Node> work(m_seq_len);

  // option and variables only used for testing during development
  bool testme = true;

  // Determine which branch to use when forming multi-sample and inserting
  // in the data store
  int which = 2;

  // Loop over the files owned by this processer
  for (const auto& t : m_filename_to_multi_sample) {

    // Load the next data file
    std::map<std::string, cnpy::NpyArray> data = cnpy::npz_load(t.first);

    for (const auto& multi_sample_id : t.second) {
      if (this_readers_indices.find(multi_sample_id) !=
          this_readers_indices.end()) {
        int starting_id = m_multi_sample_id_to_first_sample[multi_sample_id];

        // Load the single-samples that will be concatenated to form
        // the next multi-sample
        for (int k = 0; k < m_seq_len; ++k) {
          load_the_next_sample(work[k], starting_id + k, data);

          ++nn;
          if (verbose && get_comm()->am_world_master() && nn % 1000 == 0) {
            std::cout << "estimated number of single-samples processed: "
                      << utils::commify(nn / 1000 * np) << "K" << std::endl;
          }
        }

        // First branch: seq_len = 1
        if (which == 1) {
          // debug block; will go away
          if (testme && get_comm()->am_world_master()) {
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
          if (get_comm()->am_world_master() && m_seq_len == 1 && testme) {
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
}

bool ras_lipid_conduit_data_reader::fetch_datum(Mat& X,
                                                uint64_t data_id,
                                                uint64_t mb_idx)
{
  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  const double* data =
    node[LBANN_DATA_ID_STR(data_id) + "/density_sig1"].value();

  size_t n = m_seq_len * m_datum_num_words["density_sig1"];
  for (size_t j = 0; j < n; ++j) {
    X(j, mb_idx) = data[j];
  }

  /*
    Notes from Adam:

    The keras model that I gave you only looks at the density_sig1 data
    as input data and it uses the states data as labels.  We'll want to
    also extract bbs to merge that with density_sig1 in various ways as
    input data in future models that we're putting together.

    The probs field can be useful as an alternate label if building a
    regression model instead of a classification model.  I've also been
    using the probs field as a filter on the training data to only
    consider those input data whose state probability exceeds some
    threshold.

    So that works out to:

     bb, density_sig1 - datum
     states           - label
     probs            - used as a filter to include/exclude certain samples
  */

  return true;
}

std::map<double, int> m2;

bool ras_lipid_conduit_data_reader::fetch_label(Mat& Y,
                                                uint64_t data_id,
                                                uint64_t mb_idx)
{
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  const int* labels = node[LBANN_DATA_ID_STR(data_id) + "/states"].value();
  for (int j = 0; j < m_seq_len; j++) {
    Y.Set(3 * j + labels[j], mb_idx, 1);
  }
  return true;
}

bool ras_lipid_conduit_data_reader::fetch_response(Mat& Y,
                                                   uint64_t data_id,
                                                   uint64_t mb_idx)
{
  LBANN_ERROR("ras_lipid_conduit_data_reader: do not have responses");
  return true;
}

void ras_lipid_conduit_data_reader::fill_in_metadata()
{
  std::map<std::string, cnpy::NpyArray> aa = cnpy::npz_load(m_filenames[0]);
  for (const auto& t : aa) {
    const std::string& name = t.first;
    size_t word_size = t.second.word_size;
    const std::vector<size_t>& shape = t.second.shape;
    size_t num_words = 1;
    // size_t num_words = m_seq_len;
    if (shape.size() == 1) {
      m_datum_shapes[name].push_back(1 * m_seq_len);
    }
    else {
      //      m_datum_shapes[name].push_back(m_seq_len);
      for (size_t x = 1; x < shape.size(); x++) {
        num_words *= shape[x];
        m_datum_shapes[name].push_back(shape[x]);
      }
    }
    m_datum_num_words[name] = num_words;
    m_datum_word_sizes[name] = word_size;
    m_datum_num_bytes[name] = num_words * word_size;
  }

  // TODO: this should be more generic, will need to change depending on what we
  // fetch
  if (m_datum_shapes.find("density_sig1") == m_datum_shapes.end()) {
    LBANN_ERROR("m_datum_shapes.find(\"density_sig1\") = m_datum_shapes.end()");
  }
  m_num_features = 1;

  for (auto t : m_datum_shapes["density_sig1"]) {
    m_data_dims.push_back(t);
    m_num_features *= t;
  }
}

void ras_lipid_conduit_data_reader::rebuild_data_store_owner_map()
{
  m_data_store->clear_owner_map();
  int np = m_comm->get_procs_per_trainer();
  size_t data_id = 0;
  for (size_t j = 0; j < m_filenames.size(); ++j) {
    int file_owner = j % np;
    for (int h = 0; h < m_samples_per_file[j]; h++) {
      m_data_store->add_owner(data_id, file_owner);
      ++data_id;
    }
  }
  m_data_store->set_finished_building_map();
}

void ras_lipid_conduit_data_reader::get_samples_per_file()
{
  int me = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();
  std::vector<int> work;
  for (size_t j = me; j < m_filenames.size(); j += np) {
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(m_filenames[j]);
    size_t n = 0;
    for (const auto& t2 : a) {
      size_t n2 = t2.second.shape[0];
      if (n == 0) {
        n = n2;
      }
      else {
        if (n2 != n) {
          LBANN_ERROR("n2 != n; ", n2, n);
        }
      }
    }
    work.push_back(j);
    work.push_back(n);
  }

  std::vector<int> num_files(np, 0);
  for (size_t j = 0; j < m_filenames.size(); ++j) {
    int owner = j % np;
    num_files[owner] += 1;
  }

  m_samples_per_file.resize(m_filenames.size());
  std::vector<int> work_2;
  std::vector<int>* work_ptr;
  for (int j = 0; j < np; j++) {
    if (me == j) {
      work_ptr = &work;
    }
    else {
      work_2.resize(num_files[j] * 2);
      work_ptr = &work_2;
    }
    m_comm->trainer_broadcast<int>(j, work_ptr->data(), work_ptr->size());
    for (size_t h = 0; h < work_ptr->size(); h += 2) {
      m_samples_per_file[(*work_ptr)[h]] = (*work_ptr)[h + 1];
    }
  }
}

void ras_lipid_conduit_data_reader::write_file_sizes()
{
  if (!get_comm()->am_world_master()) {
    return;
  }
  std::string fn = global_argument_parser().get<std::string>(
    LBANN_OPTION_PILOT2_SAVE_FILE_SIZES);
  std::ofstream out(fn.c_str());
  if (!out) {
    LBANN_ERROR("failed to open ", fn, " for writing");
  }
  for (size_t j = 0; j < m_samples_per_file.size(); j++) {
    out << m_filenames[j] << " " << m_samples_per_file[j] << std::endl;
  }
  out.close();
}

void ras_lipid_conduit_data_reader::read_file_sizes()
{
  std::string fn = global_argument_parser().get<std::string>(
    LBANN_OPTION_PILOT2_READ_FILE_SIZES);
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
  for (size_t h = 0; h < m_filenames.size(); h++) {
    if (mp.find(m_filenames[h]) == mp.end()) {
      LBANN_ERROR("failed to find filename '", m_filenames[h], "' in the map");
    }
    m_samples_per_file[h] = mp[m_filenames[h]];
  }
}

void ras_lipid_conduit_data_reader::read_normalization_data()
{
  m_use_min_max = false;
  m_use_z_score = false;
  auto& arg_parser = global_argument_parser();
  if (arg_parser.get<std::string>(LBANN_OPTION_NORMALIZATION) != "") {
    m_use_min_max = true;
    m_use_z_score = arg_parser.get<bool>(LBANN_OPTION_Z_SCORE);
    if (get_comm()->am_world_master()) {
      if (m_use_z_score) {
        std::cout << "Normalizing data using z-score" << std::endl;
      }
      else {
        std::cout << "Normalizing data using min-max" << std::endl;
      }
    }

    std::string fn = arg_parser.get<std::string>(LBANN_OPTION_NORMALIZATION);
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
  }
  else {
    if (get_comm()->am_world_master()) {
      std::cout << "NOT Normalizing data!" << std::endl;
    }
  }
}

// user feedback
void ras_lipid_conduit_data_reader::print_shapes_etc()
{
  if (!get_comm()->am_world_master()) {
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
  for (size_t h = 0; h < m_datum_shapes["density_sig1"].size(); h++) {
    std::cout << m_datum_shapes["density_sig1"][h];
    if (h < m_datum_shapes["density_sig1"].size() - 1) {
      std::cout << "x";
    }
  }
  std::cout << std::endl;

  if (global_argument_parser().get<bool>(LBANN_OPTION_VERBOSE)) {
    std::cout << "\nAll data shapes:\n";
    for (const auto& t : m_datum_shapes) {
      std::cout << "  " << t.first << " ";
      for (const auto& t2 : t.second) {
        std::cout << t2 << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "======================================================\n\n";
}

void ras_lipid_conduit_data_reader::load_the_next_sample(
  conduit::Node& node,
  int sample_index,
  std::map<std::string, cnpy::NpyArray>& a)
{
  node.reset();
  size_t offset;
  for (const auto& t5 : m_datum_shapes) {
    const std::string& name = t5.first;
    if (name == "bbs") {
      conduit::float32* data =
        reinterpret_cast<conduit::float32*>(a[name].data_holder->data());
      offset = sample_index * m_datum_num_words["bbs"];
      node[name].set(data + offset, m_datum_num_words[name]);
    }

    else { // rots, states, tilts, density_sig1, probs
      offset = sample_index * m_datum_num_words[name];
      conduit::float64* data =
        reinterpret_cast<conduit::float64*>(a[name].data_holder->data());

      if (name == "states") {
        int label = static_cast<int>((data + offset)[0]);
        node["states"].set(label);
      }
      else if (name == "density_sig1") {
        int s = 0;
        if (m_use_z_score) {
          for (size_t j = offset; j < offset + m_datum_num_words[name]; j++) {
            data[j] = (data[j] - m_mean[s]) / m_std_dev[s];
            ++s;
            if (s == 14) {
              s = 0;
            }
          }
        }
        else if (m_use_min_max) {
          for (size_t j = offset; j < offset + m_datum_num_words[name]; j++) {
            data[j] = (data[j] - m_min[s]) / m_max_min[s];
            ++s;
            if (s == 14) {
              s = 0;
            }
          }
        }
        node[name].set(data + offset, m_datum_num_words[name]);

        // rots, tilts, probs
      }
      else {
        node[name].set(data + offset, m_datum_num_words[name]);
      }
    }
  }
}

void ras_lipid_conduit_data_reader::construct_multi_sample(
  std::vector<conduit::Node>& work,
  uint64_t data_id,
  conduit::Node& node)
{
  node.reset();
  std::vector<double> work_d;
  std::vector<float> work_f;
  std::vector<int> work_i;
  if (m_datum_num_words["states"] != 1) {
    LBANN_ERROR("m_data_num_words[states] = ",
                m_datum_num_words["states"],
                "; should be 1");
  }
  for (const auto& t42 : m_datum_num_words) {
    const std::string& name = t42.first;
    int n_words = t42.second;

    if (name == "frames") {
      continue;
    }

    if (name == "states") {
      work_i.clear();
      for (const auto& t5 : work) {
        const int label = t5[name].value();
        work_i.push_back(label);
      }
      node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(
        work_i.data(),
        m_seq_len * m_datum_num_words[name]);
    }

    // 'bbs' is float32
    else if (name == "bbs") {
      work_f.resize(m_seq_len * n_words);
      int offset = 0;
      for (const auto& t5 : work) {
        const float* d = t5[name].value();
        for (size_t u = 0; u < m_datum_num_words[name]; u++) {
          work_f[offset++] = d[u];
        }
      }
      node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(
        work_f.data(),
        m_seq_len * m_datum_num_words[name]);
    }

    // rots, tilts, density_sig1, probs are float64
    else {
      work_d.resize(m_seq_len * n_words);
      int offset = 0;
      for (const auto& t5 : work) {
        const double* d = t5[name].value();
        for (size_t u = 0; u < m_datum_num_words[name]; u++) {
          work_d[offset++] = d[u];
        }
      }
      node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(
        work_d.data(),
        m_seq_len * m_datum_num_words[name]);
    }
  }
}

} // namespace lbann
