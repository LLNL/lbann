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
// data_reader_image .hpp .cpp - generic data reader class for image dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_image.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_readers/sample_list_impl.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/vectorwrapbuf.hpp"

#include <fstream>

namespace lbann {

image_data_reader::image_data_reader(bool shuffle)
  : generic_data_reader(shuffle)
{
  set_defaults();
}

image_data_reader::image_data_reader(const image_data_reader& rhs)
  : generic_data_reader(rhs)
{
  copy_members(rhs);
}

image_data_reader& image_data_reader::operator=(const image_data_reader& rhs)
{
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  m_image_dir = rhs.m_image_dir;
  m_labels = rhs.m_labels;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_image_linearized_size = rhs.m_image_linearized_size;
  m_num_labels = rhs.m_num_labels;
  m_sample_list.copy(rhs.m_sample_list);

  return (*this);
}

void image_data_reader::copy_members(const image_data_reader& rhs)
{
  if (this == &rhs) {
    return;
  }

  if (rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
    m_data_store->set_data_reader_ptr(this);
  }

  m_image_dir = rhs.m_image_dir;
  m_labels = rhs.m_labels;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_image_linearized_size = rhs.m_image_linearized_size;
  m_num_labels = rhs.m_num_labels;
  m_sample_list.copy(rhs.m_sample_list);
  // m_thread_cv_buffer = rhs.m_thread_cv_buffer
}

void image_data_reader::set_linearized_image_size()
{
  m_image_linearized_size =
    m_image_width * m_image_height * m_image_num_channels;
}

void image_data_reader::set_defaults()
{
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
}

void image_data_reader::set_input_params(const int width,
                                         const int height,
                                         const int num_ch,
                                         const int num_labels)
{
  if ((width > 0) && (height > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
  }
  else if (!((width == 0) && (height == 0))) { // set but not valid
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: Imagenet data reader setup error: invalid input image sizes";
    throw lbann_exception(err.str());
  }
  if (num_ch > 0) {
    m_image_num_channels = num_ch;
  }
  else if (num_ch < 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: Imagenet data reader setup error: invalid number of channels "
           "of input images";
    throw lbann_exception(err.str());
  }
  set_linearized_image_size();
  if (num_labels > 0) {
    m_num_labels = num_labels;
  }
  else if (num_labels < 0) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: Imagenet data reader setup error: invalid number of labels";
    throw lbann_exception(err.str());
  }
}

bool image_data_reader::fetch_label(CPUMat& Y,
                                    uint64_t data_id,
                                    uint64_t mb_idx)
{
  if (data_id >= m_labels.size()) {
    LBANN_ERROR("Cannot find label for sample " + std::to_string(data_id) +
                ".");
  }
  const label_t label = m_labels[data_id];
  if (label < label_t{0} || label >= static_cast<label_t>(m_num_labels)) {
    LBANN_ERROR("\"",
                this->get_type(),
                "\" data reader ",
                "expects data with ",
                m_num_labels,
                " labels, ",
                "but data sample ",
                data_id,
                " has a label of ",
                label);
  }
  Y.Set(label, mb_idx, 1);
  return true;
}

void image_data_reader::dump_sample_label_list(
  const std::string& dump_file_name)
{
  std::ofstream os(dump_file_name);
  const auto num_samples = m_sample_list.size();
  for (size_t i = 0ul; i < num_samples; ++i) {
    const auto file_id = m_sample_list[i].first;
    const std::string filename = m_sample_list.get_samples_filename(file_id);
    os << filename << ' ' << std::to_string(m_labels[i]) << std::endl;
  }
}

void image_data_reader::load()
{
  auto& arg_parser = global_argument_parser();

  // Load sample list
  const std::string sample_list_file = get_data_sample_list();

  if (sample_list_file.empty()) {
    gen_list_of_samples();
  }
  else {
    load_list_of_samples(sample_list_file);
  }

  if (arg_parser.get<bool>(LBANN_OPTION_WRITE_SAMPLE_LIST) &&
      m_comm->am_trainer_master()) {
    const std::string slist_name =
      (m_sample_list.get_header()).get_sample_list_name();
    std::stringstream s;
    std::string basename = get_basename_without_ext(slist_name);
    std::string ext = get_ext_name(slist_name);
    s << basename << "." << ext;
    {
      const std::string msg =
        " writing sample list '" + slist_name + "' as '" + s.str() + "'";
      LBANN_WARNING(msg);
    }
    m_sample_list.write(s.str());
  }
  if (arg_parser.get<bool>(LBANN_OPTION_WRITE_SAMPLE_LABEL_LIST) &&
      m_comm->am_trainer_master()) {
    if (!(m_keep_sample_order ||
          arg_parser.get<bool>(LBANN_OPTION_KEEP_SAMPLE_ORDER))) {
      std::cout << "Writting sample label list without the option "
                << "`keep_sample_order' set." << std::endl;
    }
    std::string dump_file = "image_list.trainer" +
                            std::to_string(m_comm->get_trainer_rank()) + "." +
                            this->get_role() + ".txt";
    dump_sample_label_list(dump_file);
  }

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  // TODO MRW
  // opts->set_option(NODE_SIZES_VARY, 1);
  instantiate_data_store();

  select_subset_of_data();
}

image_data_reader::sample_t
image_data_reader::get_sample(const size_t idx) const
{
  if (idx >= m_labels.size()) {
    LBANN_ERROR("Cannot find label for sample " + std::to_string(idx) + ".");
  }
  const auto sample_name = m_sample_list[idx].second;
  const auto label = m_labels[idx];
  return sample_t(sample_name, label);
}

void read_raw_data(const std::string& filename, std::vector<char>& data)
{
  data.clear();
  std::ifstream in(filename.c_str());
  if (!in) {
    LBANN_ERROR("failed to open " + filename + " for reading");
  }
  in.seekg(0, in.end);
  int num_bytes = in.tellg();
  in.seekg(0, in.beg);
  data.resize(num_bytes);
  in.read((char*)data.data(), num_bytes);
  in.close();
}

void image_data_reader::do_preload_data_store()
{
  auto& arg_parser = global_argument_parser();

  int rank = m_comm->get_rank_in_trainer();

  bool threaded = !arg_parser.get<bool>(LBANN_OPTION_DATA_STORE_NO_THREAD);
  if (threaded) {
    if (get_comm()->am_world_master()) {
      std::cout << "mode: data_store_thread\n";
    }
    std::shared_ptr<thread_pool> io_thread_pool =
      construct_io_thread_pool(m_comm, false);
    int num_threads = static_cast<int>(io_thread_pool->get_num_threads());

    std::vector<std::unordered_set<int>> data_ids(num_threads);
    int j = 0;
    for (size_t data_id = 0; data_id < m_shuffled_indices.size(); data_id++) {
      int index = m_shuffled_indices[data_id];
      if (m_data_store->get_index_owner(index) != rank) {
        continue;
      }
      data_ids[j++].insert(index);
      if (j == num_threads) {
        j = 0;
      }
    }

    for (int t = 0; t < num_threads; t++) {
      if (t == io_thread_pool->get_local_thread_id()) {
        continue;
      }
      else {
        io_thread_pool->submit_job_to_work_group(
          std::bind(&image_data_reader::load_conduit_nodes_from_file,
                    this,
                    data_ids[t]));
      }
    }
    load_conduit_nodes_from_file(
      data_ids[io_thread_pool->get_local_thread_id()]);
    io_thread_pool->finish_work_group();
  }
  else {
    if (get_comm()->am_world_master()) {
      std::cout << "mode: NOT data_store_thread\n";
    }
    for (size_t data_id = 0; data_id < m_shuffled_indices.size(); data_id++) {
      int index = m_shuffled_indices[data_id];
      if (m_data_store->get_index_owner(index) != rank) {
        continue;
      }
      conduit::Node& node = m_data_store->get_empty_node(index);
      load_conduit_node_from_file(index, node);
      m_data_store->set_preloaded_conduit_node(index, node);
    }
  }
}

void image_data_reader::setup(int num_io_threads,
                              observer_ptr<thread_pool> io_thread_pool)
{
  generic_data_reader::setup(num_io_threads, io_thread_pool);
  m_transform_pipeline.set_expected_out_dims(
    {static_cast<size_t>(m_image_num_channels),
     static_cast<size_t>(m_image_height),
     static_cast<size_t>(m_image_width)});
}

bool image_data_reader::load_conduit_nodes_from_file(
  const std::unordered_set<int>& data_ids)
{
  for (auto data_id : data_ids) {
    conduit::Node& node = m_data_store->get_empty_node(data_id);
    load_conduit_node_from_file(data_id, node);
    m_data_store->set_preloaded_conduit_node(data_id, node);
  }
  return true;
}

void image_data_reader::load_conduit_node_from_file(uint64_t data_id,
                                                    conduit::Node& node)
{
  node.reset();

  const auto file_id = m_sample_list[data_id].first;
  const std::string filename =
    get_file_dir() + m_sample_list.get_samples_filename(file_id);

  if (static_cast<size_t>(data_id) >= m_labels.size()) {
    LBANN_ERROR("Cannot find label for sample " + std::to_string(data_id) +
                ".");
  }
  const label_t label = m_labels[data_id];

  std::vector<char> data;
  read_raw_data(filename, data);
  node[LBANN_DATA_ID_STR(data_id) + "/label"].set(label);
  node[LBANN_DATA_ID_STR(data_id) + "/buffer"].set(data);
  node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"] = data.size();
}

/**
 * Load a sample list and then load labels from a separate file using
 * `load_labels()` With the command line option `--load_full_sample_list_once`,
 * the trainer master first loads the entire sample list file into a memory
 * buffer, and broadcasts it to the other workers within the trainer. Then, the
 * sample list is populated using the buffer content. Otherwise, the sample list
 * is directly read from the file. The prototext variable `data_filedir` when
 * specified overrides the base location of data files, written in the header of
 * the sample list file. The option `keep_sample_order` from the command line or
 * data reader prototexts, makes sure the order of samples in the list remains
 * the same even with loading in an interleaving order by multiple trainer
 * workers.
 */
void image_data_reader::load_list_of_samples(const std::string sample_list_file)
{
  // load the sample list
  double tm1 = get_time();

  auto& arg_parser = global_argument_parser();

  if (m_keep_sample_order ||
      arg_parser.get<bool>(LBANN_OPTION_KEEP_SAMPLE_ORDER)) {
    m_sample_list.keep_sample_order(true);
  }
  else {
    m_sample_list.keep_sample_order(false);
  }

  if (arg_parser.get<bool>(LBANN_OPTION_CHECK_DATA)) {
    m_sample_list.set_data_file_check();
  }

  std::vector<char> buffer;

  if (arg_parser.get<bool>(LBANN_OPTION_LOAD_FULL_SAMPLE_LIST_ONCE)) {
    if (m_comm->am_trainer_master()) {
      load_file(sample_list_file, buffer);
    }
    m_comm->trainer_broadcast(m_comm->get_trainer_master(), buffer);

    vectorwrapbuf<char> strmbuf(buffer);
    std::istream iss(&strmbuf);

    m_sample_list.set_sample_list_name(sample_list_file);
    m_sample_list.load(iss, *m_comm, true);
  }
  else {
    m_sample_list.load(sample_list_file, *m_comm, true);
  }

  double tm2 = get_time();

  if (get_comm()->am_world_master()) {
    std::cout << "Time to load sample list '" << sample_list_file
              << "': " << tm2 - tm1 << std::endl;
  }

  /// Merge all the sample list pieces from the workers within the trainer
  m_sample_list.all_gather_packed_lists(*m_comm);
  set_file_dir(m_sample_list.get_samples_dirname());

  double tm3 = get_time();
  if (get_comm()->am_world_master()) {
    std::cout << "Time to gather sample list '" << sample_list_file
              << "': " << tm3 - tm2 << std::endl;
  }
  buffer.clear();
  buffer.shrink_to_fit();

  std::vector<char> empty_buffer;
  load_labels(empty_buffer);
}

void image_data_reader::load_list_of_samples_from_archive(
  const std::string& sample_list_archive)
{
  // load the sample list
  double tm1 = get_time();
  std::stringstream ss(sample_list_archive); // any stream can be used

  cereal::BinaryInputArchive iarchive(ss); // Create an input archive

  iarchive(m_sample_list); // Read the data from the archive
  double tm2 = get_time();

  if (get_comm()->am_world_master()) {
    std::cout << "Time to load sample list from archive: " << tm2 - tm1
              << std::endl;
  }
}

/**
 * Similar to `load_list_of_samples()` but generates the sample list header
 * on-the-fly, and reuse the original imagenet data list file for loading both
 * the sample list and the label list, of which path is specified via the
 * prototext variable `data_filedir`. This is for the backward compatibility
 * and allows users to use the old data reader prototext without preparing a
 * sample list and modifying the prototext. The base location of data files
 * is specified via `data_filedir` prototext variable as it was.
 */
void image_data_reader::gen_list_of_samples()
{
  // load the sample list
  double tm1 = get_time();

  // The original imagenet data file specified via the prototext variable
  // `data_filename`
  const std::string imageListFile = get_data_filename();

  sample_list_header header; // A sample list header being generated
  header.set_sample_list_type(lbann::single_sample);
  header.set_data_file_dir(get_file_dir());
  header.set_label_filename(imageListFile);
  const std::string sample_list_file = imageListFile;
  header.set_sample_list_name(sample_list_file);

  auto& arg_parser = global_argument_parser();

  if (m_keep_sample_order ||
      arg_parser.get<bool>(LBANN_OPTION_KEEP_SAMPLE_ORDER)) {
    m_sample_list.keep_sample_order(true);
  }
  else {
    m_sample_list.keep_sample_order(false);
  }

  if (arg_parser.get<bool>(LBANN_OPTION_CHECK_DATA)) {
    m_sample_list.set_data_file_check();
  }

  std::vector<char> buffer;

  if (arg_parser.get<bool>(LBANN_OPTION_LOAD_FULL_SAMPLE_LIST_ONCE)) {
    // The trainer master loads the entire file into a buffer in the memory
    if (m_comm->am_trainer_master()) {
      load_file(imageListFile, buffer);
    }
    // Broadcast the buffer to workers within this trainer
    m_comm->trainer_broadcast(m_comm->get_trainer_master(), buffer);

    // The trainer master counts the number of samples (lines) and broadcasts
    // the result
    size_t num_samples = 0ul;
    if (m_comm->am_trainer_master()) {
      vectorwrapbuf<char> strmbuf(buffer);
      std::istream iss(&strmbuf);
      num_samples = determine_num_of_samples(iss);
    }
    m_comm->trainer_broadcast(m_comm->get_trainer_master(), num_samples);
    header.set_sample_count(std::to_string(num_samples));

    // Populate the sample list using the generated header and the preloaded
    // buffer
    vectorwrapbuf<char> strmbuf(buffer);
    std::istream iss(&strmbuf);
    m_sample_list.load(header, iss, *m_comm, true);
  }
  else {
    // The trainer master counts the number of samples (lines) and broadcasts
    // the result
    size_t num_samples = 0ul;
    if (m_comm->am_trainer_master()) {
      std::ifstream iss(imageListFile);
      num_samples = determine_num_of_samples(iss);
    }
    m_comm->trainer_broadcast(m_comm->get_trainer_master(), num_samples);
    header.set_sample_count(std::to_string(num_samples));

    // Populate the sample list using the generated header and the original
    // imagenet data list file
    std::ifstream iss(imageListFile);
    m_sample_list.load(header, iss, *m_comm, true);
  }

  double tm2 = get_time();

  if (get_comm()->am_world_master()) {
    std::cout << "Time to load sample list '" << sample_list_file
              << "': " << tm2 - tm1 << std::endl;
  }

  /// Merge all the sample list pieces from the workers within the trainer
  m_sample_list.all_gather_packed_lists(*m_comm);

  double tm3 = get_time();
  if (get_comm()->am_world_master()) {
    std::cout << "Time to gather sample list '" << sample_list_file
              << "': " << tm3 - tm2 << std::endl;
  }
  // Reuse the preloaded buffer for obtaining labels when possible
  load_labels(buffer);
}

/// Populate the sample label vector out of the given input stream
void image_data_reader::read_labels(std::istream& istrm)
{
  const std::string whitespaces(" \t\f\v\n\r");
  const size_t num_samples = m_sample_list.size();

  // To help populating the label list, build a map from a sample name to
  // the index of the corresponding item in the sample list
  m_sample_list.build_sample_map_from_name_to_index();

  auto& arg_parser = global_argument_parser();
  const bool check_data = arg_parser.get<bool>(LBANN_OPTION_CHECK_DATA);

  m_labels.clear();
  m_labels.resize(num_samples);
  std::unordered_set<sample_idx_t> idx_set;

  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }

    // clear trailing spaces for accurate parsing
    std::stringstream sstr(line.substr(0, end_of_str + 1));
    std::string sname;
    label_t label;

    sstr >> sname >> label;

    // Translate the sample name into the index into the sample list
    const auto sample_idx =
      m_sample_list.get_sample_index(sample_name_t(sname));
    if (sample_idx >= num_samples) {
      continue;
    }
    if (check_data) {
      idx_set.insert(sample_idx);
    }
    m_labels[sample_idx] = label;
  }

  // Free the memory of the temporary map
  m_sample_list.clear_sample_map_from_name_to_index();

  if (check_data && (num_samples != idx_set.size())) {
    LBANN_ERROR(
      "The number of samples is different from the number of labels: ",
      std::to_string(num_samples),
      " != ",
      std::to_string(idx_set.size()));
  }
}

/**
 * Load the sample labels either from a file or from a preloaded buffer.
 * If the buffer given is empty, the label file specified in the sample list
 * header is used.
 */
void image_data_reader::load_labels(std::vector<char>& preloaded_buffer)
{
  const std::string imageListFile = m_sample_list.get_label_filename();

  double tm1 = get_time();

  if (preloaded_buffer.empty()) { // read labels from a file
    std::string line;
    std::ifstream is;
    is.open(imageListFile);
    if (is.fail()) {
      LBANN_ERROR("failed to open: " + imageListFile + " for reading");
    }
    read_labels(is);
  }
  else { // read labels from a preloaded buffer
    vectorwrapbuf<char> strmbuf(preloaded_buffer);
    std::istream is(&strmbuf);
    read_labels(is);
  }

  if (get_comm()->am_world_master()) {
    std::cout << "Time to load label file '" << imageListFile
              << "': " << get_time() - tm1 << std::endl;
  }
}

size_t image_data_reader::determine_num_of_samples(std::istream& istrm) const
{
  const std::string whitespaces(" \t\f\v\n\r");
  size_t cnt = 0ul;
  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    cnt++;
  }
  return cnt;
}

} // namespace lbann
