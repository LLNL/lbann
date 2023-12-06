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
// data_reader_imagenet .hpp .cpp - data reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_ingestion/readers/data_reader_imagenet.hpp"
#include "lbann/data_ingestion/readers/sample_list_impl.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/image.hpp"

namespace lbann {

imagenet_reader::imagenet_reader(bool shuffle) : image_data_reader(shuffle)
{
  set_defaults();
}

imagenet_reader::~imagenet_reader() {}

void imagenet_reader::set_defaults()
{
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
  m_supported_input_types[INPUT_DATA_TYPE_LABELS] = true;
}

CPUMat imagenet_reader::create_datum_view(CPUMat& X,
                                          const uint64_t mb_idx) const
{
  return El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
}

bool imagenet_reader::fetch_datum(CPUMat& X, uint64_t data_id, uint64_t mb_idx)
{
  El::Matrix<uint8_t> image;
  std::vector<size_t> dims;
  const auto file_id = m_sample_list[data_id].first;
  const std::string filename = m_sample_list.get_samples_filename(file_id);
  const std::string image_path = get_file_dir() + filename;

  if (m_data_store != nullptr) {
    bool have_node = true;
    conduit::Node node;
    if (m_data_store->is_local_cache()) {
      if (m_data_store->has_conduit_node(data_id)) {
        const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
        node.set_external(ds_node);
      }
      else {
        load_conduit_node_from_file(data_id, node);
        m_data_store->set_conduit_node(data_id, node);
      }
    }
    else if (data_store_active()) {
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
    }
    else if (priming_data_store()) {
      load_conduit_node_from_file(data_id, node);
      m_data_store->set_conduit_node(data_id, node);
    }
    else {
      if (get_role() != "test") {
        LBANN_ERROR("you shouldn't be here; please contact Dave Hysom");
      }
      if (m_issue_warning) {
        if (get_comm()->am_world_master()) {
          LBANN_WARNING("m_data_store != nullptr, but we are not retrivieving "
                        "a node from the store; role: " +
                        get_role() +
                        "; this is probably OK for test mode, but may be an "
                        "error for train or validate modes");
        }
      }
      m_issue_warning = false;
      load_image(image_path, image, dims);
      have_node = false;
    }

    if (have_node) {
      char* buf = node[LBANN_DATA_ID_STR(data_id) + "/buffer"].value();
      size_t size = node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();
      El::Matrix<uint8_t> encoded_image(size,
                                        1,
                                        reinterpret_cast<uint8_t*>(buf),
                                        size);
      decode_image(encoded_image, image, dims);
    }
  }

  // this block fires if not using data store
  else {
    load_image(image_path, image, dims);
  }

  auto X_v = create_datum_view(X, mb_idx);
  m_transform_pipeline.apply(image, X_v, dims);

  return true;
}

} // namespace lbann
