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
// data_reader_imagenet .hpp .cpp - data reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/utils/file_utils.hpp"
#include <omp.h>

namespace lbann {

imagenet_reader::imagenet_reader(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : image_data_reader(shuffle) {
  set_defaults();

  if (!pp) {
    LBANN_ERROR("construction error: no image processor");
  }

  m_master_pps = lbann::make_unique<cv_process>(*pp);

  if (is_master()) std::cout << "XX imagenet_reader ctor, pp, shuffle\n";
}

imagenet_reader::imagenet_reader(const imagenet_reader& rhs)
  : image_data_reader(rhs) {
  if (!rhs.m_master_pps) {
    LBANN_ERROR("construction error: no image processor");
  }
  m_master_pps = lbann::make_unique<cv_process>(*rhs.m_master_pps);
  if (is_master()) std::cout << "XX imagenet_reader copy ctor\n";
}


imagenet_reader::imagenet_reader(const imagenet_reader& rhs, const std::vector<int>& ds_sample_move_list, std::string role)
  : image_data_reader(rhs, ds_sample_move_list) {
  if (!rhs.m_master_pps) {
    LBANN_ERROR("construction error: no image processor");
  }
  m_master_pps = lbann::make_unique<cv_process>(*rhs.m_master_pps);
  set_role(role);

  if (is_master()) std::cout << "XX imagenet_reader copy ctor, ds_sample_list size: " << ds_sample_move_list.size() << "\n";
}

imagenet_reader::imagenet_reader(const imagenet_reader& rhs, const std::vector<int>& ds_sample_move_list)
  : image_data_reader(rhs, ds_sample_move_list) {
  if (!rhs.m_master_pps) {
    LBANN_ERROR("construction error: no image processor");
  }
  m_master_pps = lbann::make_unique<cv_process>(*rhs.m_master_pps);

  if (is_master()) std::cout << "XX imagenet_reader copy ctor, ds_sample_list size: " << ds_sample_move_list.size() << "\n";
}

imagenet_reader& imagenet_reader::operator=(const imagenet_reader& rhs) {
  if (is_master()) std::cout << "XX imagenet_reader operator=\n";
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  image_data_reader::operator=(rhs);

  if (!rhs.m_master_pps) {
    LBANN_ERROR("construction error: no image processor");
  }
  m_master_pps = lbann::make_unique<cv_process>(*rhs.m_master_pps);
  return (*this);
}

imagenet_reader::~imagenet_reader() {
}

void imagenet_reader::set_defaults() {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
}

void imagenet_reader::setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) {
  image_data_reader::setup(num_io_threads, io_thread_pool);
  replicate_processor(*m_master_pps, num_io_threads);
}

/// Replicate image processor for each I/O thread
bool imagenet_reader::replicate_processor(const cv_process& pp, const int nthreads) {
  m_pps.resize(nthreads);

  // Construct thread private preprocessing objects out of a shared pointer
  for (int i = 0; i < nthreads; ++i) {
    m_pps[i] = lbann::make_unique<cv_process>(pp);
  }

  bool ok = true;
  for (int i = 0; ok && (i < nthreads); ++i) {
    if (!m_pps[i]) ok = false;
  }

  if (!ok || (nthreads <= 0)) {
    LBANN_ERROR("cannot replicate image processor");
  }

  const std::vector<unsigned int> dims = pp.get_data_dims();
  if ((dims.size() == 2u) && (dims[0] != 0u) && (dims[1] != 0u)) {
    m_image_width = static_cast<int>(dims[0]);
    m_image_height = static_cast<int>(dims[1]);
    set_linearized_image_size();
  }

  return true;
}

CPUMat imagenet_reader::create_datum_view(CPUMat& X, const int mb_idx) const {
  return El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
}

bool imagenet_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  int width=0, height=0, img_type=0;
  int tid = m_io_thread_pool->get_local_thread_id();
  CPUMat X_v = create_datum_view(X, mb_idx);
  bool ret;
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  bool have_node = true;
  if (m_data_store != nullptr) {
    conduit::Node node;
    if (m_data_store->is_local_cache()) {
      if (m_data_store->has_conduit_node(data_id)) {
        const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
        node.set_external(ds_node);
      } else {
        load_conduit_node_from_file(data_id, node);
        m_data_store->set_conduit_node(data_id, node);
      }
    } else if (data_store_active()) {
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
    } else if (priming_data_store()) {
      load_conduit_node_from_file(data_id, node);
      m_data_store->set_conduit_node(data_id, node);
    } else {
      if (get_role() != "test") {
        LBANN_ERROR("you shouldn't be here; please contact Dave Hysom");
      }
      if (m_issue_warning) {
        if (is_master()) {
          LBANN_WARNING("m_data_store != nullptr, but we are not retrivieving a node from the store; role: " + get_role() + "; this is probably OK for test mode, but may be an error for train or validate modes");
        }  
        m_issue_warning = false;
      }
      ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_pps[tid]), X_v, m_thread_buffer[tid], &m_thread_cv_buffer[tid]);
      have_node = false;
    }

    if (have_node) {
      char *buf = node[LBANN_DATA_ID_STR(data_id) + "/buffer"].value();
      size_t size = node[LBANN_DATA_ID_STR(data_id) + "/buffer_size"].value();
      std::vector<unsigned char> v2(size);
      for (size_t j=0; j<size; j++) {
        v2[j] = buf[j];
      }
      ret = lbann::image_utils::load_image(v2, width, height, img_type, *(m_pps[tid]), X_v, &m_thread_cv_buffer[tid]);
      //ret = lbann::image_utils::load_image(v2, width, height, img_type, *(m_pps[tid]), X_v, m_thread_buffer[tid], &m_thread_cv_buffer[tid]);
    }
  }
  
  // not using data store
  else {
    ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_pps[tid]), X_v, m_thread_buffer[tid], &m_thread_cv_buffer[tid]);
  }

  if(!ret) {
    LBANN_ERROR(get_type() + ": image_utils::load_image failed to load - " + imagepath);
  }
  if((width * height * CV_MAT_CN(img_type)) != m_image_linearized_size) {
    LBANN_ERROR( get_type() + ": mismatch data size -- either width, height or channel - " + imagepath + "[w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(CV_MAT_CN(img_type)) + "]");
  } 

  return true;
}

}  // namespace lbann
