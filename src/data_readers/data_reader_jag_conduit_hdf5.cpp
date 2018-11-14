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

#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include "lbann/utils/file_utils.hpp" // for add_delimiter() in load()
#include "lbann/utils/options.hpp" // for add_delimiter() in load()
#include "lbann/data_store/jag_store.hpp"
#include "lbann/models/model.hpp"

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv_extensions.hpp"
#include <memory>
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/glob.hpp"
#include <thread>


// This macro may be moved to a global scope
#define _THROW_LBANN_EXCEPTION_(_CLASS_NAME_,_MSG_) { \
  std::stringstream err; \
  err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG_); \
  throw lbann_exception(err.str()); \
}

#define _THROW_LBANN_EXCEPTION2_(_CLASS_NAME_,_MSG1_,_MSG2_) { \
  std::stringstream err; \
  err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG1_) << (_MSG2_); \
  throw lbann_exception(err.str()); \
}

// This comes after all the headers, and is only visible within the current implementation file.
// To make sure, we put '#undef _CN_' at the end of this file
#define _CN_ "data_reader_jag_conduit_hdf5"

namespace lbann {

data_reader_jag_conduit_hdf5::data_reader_jag_conduit_hdf5(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : generic_data_reader(shuffle),
    m_jag_store(nullptr),
    m_owns_jag_store(false),
    m_primary_reader(nullptr) {

  set_defaults();

  if (!pp) {
    _THROW_LBANN_EXCEPTION_(get_type(), " construction error: no image processor");
  }

  replicate_processor(*pp);
}

void data_reader_jag_conduit_hdf5::copy_members(const data_reader_jag_conduit_hdf5& rhs) {
  m_jag_store = rhs.m_jag_store;
  m_owns_jag_store = rhs.m_owns_jag_store;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  m_is_data_loaded = rhs.m_is_data_loaded;
  m_scalar_keys = rhs.m_scalar_keys;
  m_input_keys = rhs.m_input_keys;
  m_success_map = rhs.m_success_map;

  if (rhs.m_pps.size() == 0u || !rhs.m_pps[0]) {
    _THROW_LBANN_EXCEPTION_(get_type(), " construction error: no image processor");
  }

  replicate_processor(*rhs.m_pps[0]);
  m_uniform_input_type = rhs.m_uniform_input_type;
}


data_reader_jag_conduit_hdf5::data_reader_jag_conduit_hdf5(const data_reader_jag_conduit_hdf5& rhs)
  : generic_data_reader(rhs) {
  copy_members(rhs);
}

data_reader_jag_conduit_hdf5& data_reader_jag_conduit_hdf5::operator=(const data_reader_jag_conduit_hdf5& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  generic_data_reader::operator=(rhs);

  copy_members(rhs);

  return (*this);
}

data_reader_jag_conduit_hdf5::~data_reader_jag_conduit_hdf5() {
  if (m_owns_jag_store) {
    delete m_jag_store;
  }
}

void data_reader_jag_conduit_hdf5::set_defaults() {
  m_image_width = 0;
  m_image_height = 0;
  m_image_num_channels = 1;
  m_num_labels = 0;
}

/// Replicate image processor for each OpenMP thread
bool data_reader_jag_conduit_hdf5::replicate_processor(const cv_process& pp) {
  const int nthreads = omp_get_max_threads();
  m_pps.resize(nthreads);

  // Construct thread private preprocessing objects out of a shared pointer
  #pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < nthreads; ++i) {
    std::unique_ptr<cv_process> ppu(new cv_process(pp));
    m_pps[i] = std::move(ppu);
  }

  bool ok = true;
  for (int i = 0; ok && (i < nthreads); ++i) {
    if (!m_pps[i]) ok = false;
  }

  if (!ok || (nthreads <= 0)) {
    _THROW_LBANN_EXCEPTION_(get_type(), " cannot replicate image processor");
    return false;
  }

  const std::vector<unsigned int> dims = pp.get_data_dims();
  if ((dims.size() == 2u) && (dims[0] != 0u) && (dims[1] != 0u)) {
    m_image_width = static_cast<int>(dims[0]);
    m_image_height = static_cast<int>(dims[1]);
  }

  return true;
}

void data_reader_jag_conduit_hdf5::set_image_dims(const int width, const int height, const int ch) {
  m_image_width = width;
  m_image_height = height;
  m_image_num_channels = ch;
}

bool data_reader_jag_conduit_hdf5::fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) {
  m_jag_store->load_data(data_id, tid);

  std::vector<size_t> sizes = get_linearized_data_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);

  size_t i = 0;
  std::vector<cv::Mat> images = get_cv_images(data_id, tid);

  for(size_t k=0u; k < get_num_img_srcs(); ++k) {
    int width, height, img_type;
    image_utils::process_image(images[k], width, height, img_type, *(m_pps[tid]), X_v[i++]);
   }

  const std::vector<data_reader_jag_conduit_hdf5::scalar_t> &scalars = m_jag_store->fetch_scalars(data_id, tid);
  set_minibatch_item<data_reader_jag_conduit_hdf5::scalar_t>(X_v[i++], 0, scalars.data(), m_jag_store->get_linearized_scalar_size());

  const std::vector<data_reader_jag_conduit_hdf5::input_t> &inputs = m_jag_store->fetch_inputs(data_id, tid);
  set_minibatch_item<data_reader_jag_conduit_hdf5::input_t>(X_v[i++], 0, inputs.data(), m_jag_store->get_linearized_input_size());
  return true;
}

void data_reader_jag_conduit_hdf5::load() {
  if(m_gan_labelling) {
    m_num_labels=2;
  }

  if (is_master()) {
    std::cout << "JAG load GAN m_gan_labelling : label_value "
              << m_gan_labelling <<" : " << m_gan_label_value << std::endl;
  }

  bool setup_jag_store = true;

  if (setup_jag_store) {
    m_jag_store = new jag_store;
  
    m_jag_store->set_comm(m_comm);
    if (is_master()) std::cerr << "calling: m_jag_store->set_image_size\n";
    m_jag_store->set_image_size(m_image_height * m_image_width);

    if (m_first_n > 0) {
      _THROW_LBANN_EXCEPTION_(_CN_, "load() does not support first_n feature.");
    }

    if (is_master()) std::cerr << "data_reader_jag_conduit_hdf5: calling m_jag_store->setup()\n";
    m_jag_store->setup(this);
  }

  m_is_data_loaded = true;

  // reset indices
  m_shuffled_indices.resize(get_num_samples());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();

  if (is_master()) {
    std::cout << "\n" << get_description() << "\n\n";
  }
}

size_t data_reader_jag_conduit_hdf5::get_num_samples() const {
  return m_jag_store->get_num_samples();
}

unsigned int data_reader_jag_conduit_hdf5::get_num_img_srcs() const {
  return m_jag_store->get_num_img_srcs();
}

unsigned int data_reader_jag_conduit_hdf5::get_num_channels() const {
  return m_jag_store->get_num_channels_per_view();
}

size_t data_reader_jag_conduit_hdf5::get_linearized_channel_size() const {
  return m_jag_store->get_linearized_channel_size();
}

size_t data_reader_jag_conduit_hdf5::get_linearized_image_size() const {
  return m_jag_store->get_linearized_image_size();
}

size_t data_reader_jag_conduit_hdf5::get_linearized_scalar_size() const {
  return m_jag_store->get_linearized_scalar_size();
}

size_t data_reader_jag_conduit_hdf5::get_linearized_input_size() const {
  return m_jag_store->get_linearized_input_size();
}


int data_reader_jag_conduit_hdf5::get_linearized_data_size() const {
  return m_jag_store->get_linearized_data_size();
}

int data_reader_jag_conduit_hdf5::get_linearized_response_size() const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  return 0;
  return 0;
}

std::vector<size_t> data_reader_jag_conduit_hdf5::get_linearized_data_sizes() const {
  return m_jag_store->get_linearized_data_sizes();
}

std::vector<size_t> data_reader_jag_conduit_hdf5::get_linearized_response_sizes() const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  std::vector<size_t> r;
  return r;
}

const std::vector<int> data_reader_jag_conduit_hdf5::get_data_dims() const {
  return {get_linearized_data_size()};
}

int data_reader_jag_conduit_hdf5::get_num_labels() const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  return m_num_labels;
}

int data_reader_jag_conduit_hdf5::get_linearized_label_size() const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  return m_num_labels;
  return 0;
}

std::string data_reader_jag_conduit_hdf5::get_description() const {
/*
  std::vector<size_t> s = get_linearized_data_sizes();
  std::string ret = std::string("data_reader_jag_conduit_hdf5:\n")
    + " - independent: " + data_reader_jag_conduit_hdf5::to_string(m_independent) + "\n"
    + " - dependent: " + data_reader_jag_conduit_hdf5::to_string(m_dependent) + "\n"
    + " - images: "   + std::to_string(m_num_img_srcs) + 'x'
                      + std::to_string(m_image_width) + 'x'
                      + std::to_string(m_image_height) + "\n"
    + " - scalars: "  + std::to_string(get_linearized_scalar_size()) + "\n"
    + " - inputs: "   + std::to_string(get_linearized_input_size()) + "\n"
    + " - linearized data size: "   + std::to_string(get_linearized_data_size()) + "\n"

    + " - uniform_input_type: " + (m_uniform_input_type? "true" : "false") + '\n';
    ret += '\n';
  return ret;
  */
  return "";
}


bool data_reader_jag_conduit_hdf5::check_sample_id(const size_t sample_id) const {
  m_jag_store->check_sample_id(sample_id);
  return true;
}

cv::Mat data_reader_jag_conduit_hdf5::cast_to_cvMat(const std::pair<size_t, const ch_t*> img, const int height) {
  const int num_pixels = static_cast<int>(img.first);
  const ch_t* ptr = img.second;

  // add a zero copying view to data
  using InputBuf_T = cv_image_type<ch_t>;
  const cv::Mat image(num_pixels, 1, InputBuf_T::T(1u),
                      reinterpret_cast<void*>(const_cast<ch_t*>(ptr)));
  // reshape the image. Furter need to clone (deep-copy) the image
  // to preserve the constness of the original data
  return (image.reshape(0, height));
}

std::vector<cv::Mat> data_reader_jag_conduit_hdf5::get_cv_images(const size_t sample_id, int tid) const {
  const std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>> &raw_images = m_jag_store->fetch_views(sample_id, tid);
  std::vector< std::pair<size_t, const ch_t*> > img_ptrs(raw_images.size());
  size_t num_pixels = get_linearized_channel_size();
  for (size_t h=0; h<raw_images.size(); h++) {
    img_ptrs[h] = std::make_pair(num_pixels, raw_images[h].data());
  }

  std::vector<cv::Mat> images;
  images.reserve(img_ptrs.size());

  for (const auto& img: img_ptrs) {
    images.emplace_back(cast_to_cvMat(img, m_image_height).clone());
  }
  return images;
}

std::vector<CPUMat>
data_reader_jag_conduit_hdf5::create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const {
  std::vector<CPUMat> X_v(sizes.size());
  El::Int h = 0;

  for(size_t i=0u; i < sizes.size(); ++i) {
    const El::Int h_end =  h + static_cast<El::Int>(sizes[i]);
    El::View(X_v[i], X, El::IR(h, h_end), El::IR(mb_idx, mb_idx + 1));
    h = h_end;
  }
  return X_v;
}

bool data_reader_jag_conduit_hdf5::fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) {
  if(m_gan_label_value) Y.Set(m_gan_label_value,mb_idx,1); //fake sample is set to 1; adversarial model
  else { //fake sample (second half of minibatch is set to 0;discriminator model
    //mb_idx < (m_mb_size/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
    mb_idx < (get_current_mini_batch_size()/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
  }
  //Y.Set(m_gan_label_value, mb_idx, 1);
  return true;
}

void data_reader_jag_conduit_hdf5::setup_data_store(model *m) {
  if (m_data_store != nullptr) {
    //delete m_data_store;
  }
/*
  m_data_store = new data_store_jag_conduit(this, m);
  if (m_data_store != nullptr) {
    m_data_store->setup();
  }
*/
}

void data_reader_jag_conduit_hdf5::post_update() {
  return;
}

} // end of namespace lbann

#undef _CN_
#endif // #ifdef LBANN_HAS_CONDUIT
