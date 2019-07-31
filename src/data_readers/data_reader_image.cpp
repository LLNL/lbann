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
// data_reader_image .hpp .cpp - generic data reader class for image dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_image.hpp"
#include "lbann/utils/image.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/utils/file_utils.hpp"
#include <fstream>

namespace lbann {

image_data_reader::image_data_reader(bool shuffle)
  : generic_data_reader(shuffle) {
  set_defaults();
}

image_data_reader::image_data_reader(const image_data_reader& rhs)
  : generic_data_reader(rhs)
{
  copy_members(rhs);
}

image_data_reader::image_data_reader(const image_data_reader& rhs,const std::vector<int>& ds_sample_move_list, std::string role)
  : generic_data_reader(rhs)
{
  set_role(role);
  copy_members(rhs, ds_sample_move_list);
}

image_data_reader::image_data_reader(const image_data_reader& rhs,const std::vector<int>& ds_sample_move_list)
  : generic_data_reader(rhs)
{
  copy_members(rhs, ds_sample_move_list);
}

image_data_reader& image_data_reader::operator=(const image_data_reader& rhs) {
  generic_data_reader::operator=(rhs);
  m_image_dir = rhs.m_image_dir;
  m_image_list = rhs.m_image_list;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_image_linearized_size = rhs.m_image_linearized_size;
  m_num_labels = rhs.m_num_labels;

  return (*this);
}

void image_data_reader::copy_members(const image_data_reader &rhs, const std::vector<int>& ds_sample_move_list) {

  if(rhs.m_data_store != nullptr) {
    if(ds_sample_move_list.size() == 0) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
    } else {
      m_data_store = new data_store_conduit(rhs.get_data_store(), ds_sample_move_list);
    }
    m_data_store->set_data_reader_ptr(this);
  }

  m_image_dir = rhs.m_image_dir;
  m_image_list = rhs.m_image_list;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_image_linearized_size = rhs.m_image_linearized_size;
  m_num_labels = rhs.m_num_labels;
  //m_thread_cv_buffer = rhs.m_thread_cv_buffer
}


void image_data_reader::set_linearized_image_size() {
  m_image_linearized_size = m_image_width * m_image_height * m_image_num_channels;
}

void image_data_reader::set_defaults() {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
}

void image_data_reader::set_input_params(const int width, const int height, const int num_ch, const int num_labels) {
  if ((width > 0) && (height > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
  } else if (!((width == 0) && (height == 0))) { // set but not valid
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader setup error: invalid input image sizes";
    throw lbann_exception(err.str());
  }
  if (num_ch > 0) {
    m_image_num_channels = num_ch;
  } else if (num_ch < 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader setup error: invalid number of channels of input images";
    throw lbann_exception(err.str());
  }
  set_linearized_image_size();
  if (num_labels > 0) {
    m_num_labels = num_labels;
  } else if (num_labels < 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: Imagenet data reader setup error: invalid number of labels";
    throw lbann_exception(err.str());
  }
}

bool image_data_reader::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  const label_t label = m_image_list[data_id].second;
  Y.Set(label, mb_idx, 1);
  return true;
}

void image_data_reader::load() {
  options *opts = options::get();

  const std::string imageListFile = get_data_filename();

  // load image list
  m_image_list.clear();
  FILE *fplist = fopen(imageListFile.c_str(), "rt");
  if (!fplist) {
    LBANN_ERROR("failed to open: " + imageListFile + " for reading");
  }
  while (!feof(fplist)) {
    char imagepath[512];
    label_t imagelabel;
    if (fscanf(fplist, "%s%d", imagepath, &imagelabel) <= 1) {
      break;
    }
    m_image_list.emplace_back(imagepath, imagelabel);
  }
  fclose(fplist);

  // TODO: this will probably need to change after sample_list class
  //       is modified
  
  std::vector<int> local_list_sizes;
  if (opts->get_bool("preload_data_store") || opts->get_bool("data_store_cache")) {
    int np = m_comm->get_procs_per_trainer();
    int base_files_per_rank = m_image_list.size() / np;
    int extra = m_image_list.size() - (base_files_per_rank*np);
    if (extra > np) {
      LBANN_ERROR("extra > np");
    }
    local_list_sizes.resize(np, 0);
    for (int j=0; j<np; j++) {
      local_list_sizes[j] = base_files_per_rank;
      if (j < extra) {
        local_list_sizes[j] += 1;
      }
    }
  }

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_image_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  opts->set_option("node_sizes_vary", 1);
  instantiate_data_store(local_list_sizes);

  select_subset_of_data();
}

//void read_raw_data(const std::string &filename, std::vector<conduit::uint8> &data) {
void read_raw_data(const std::string &filename, std::vector<char> &data) {
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

void image_data_reader::preload_data_store() {
  double tm1 = get_time();
  m_data_store->set_preload();

  conduit::Node node;
  int rank = m_comm->get_rank_in_trainer();
  for (size_t data_id=0; data_id<m_shuffled_indices.size(); data_id++) {
    if (m_data_store->get_index_owner(data_id) != rank) {
      continue;
    }
    load_conduit_node_from_file(data_id, node);
    m_data_store->set_preloaded_conduit_node(data_id, node);
  }

  if (is_master()) {
    std::cout << "image_data_reader::preload_data_store time: " << (get_time() - tm1) << "\n";
  }  
}

void image_data_reader::setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) {
  generic_data_reader::setup(num_io_threads, io_thread_pool);
   m_transform_pipeline.set_expected_out_dims(
    {static_cast<size_t>(m_image_num_channels),
     static_cast<size_t>(m_image_height),
     static_cast<size_t>(m_image_width)});
}

std::vector<image_data_reader::sample_t> image_data_reader::get_image_list_of_current_mb() const {
  std::vector<sample_t> ret;
  ret.reserve(m_mini_batch_size);
  return ret;
}

void image_data_reader::load_conduit_node_from_file(int data_id, conduit::Node &node) {
  node.reset();
  const std::string filename = get_file_dir() + m_image_list[data_id].first;
  int label = m_image_list[data_id].second;
  std::vector<char> data;
  read_raw_data(filename, data);
  node[LBANN_DATA_ID_STR(data_id) + "/label"].set(label);
  node[LBANN_DATA_ID_STR(data_id) + "/buffer"].set(data);
  node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"] = data.size();
}


}  // namespace lbann
