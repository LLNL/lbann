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

#ifndef _JAG_OFFLINE_TOOL_MODE_
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include "lbann/utils/file_utils.hpp" // for add_delimiter() in load()
#include "lbann/utils/options.hpp" // for add_delimiter() in load()
#include "lbann/data_store/jag_store.hpp"
#else
#include "data_reader_jag_conduit_hdf5.hpp"
#endif // _JAG_OFFLINE_TOOL_MODE_

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv_extensions.hpp"
#include <memory>
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/glob.hpp"


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
    m_owns_jag_store(false) {

  set_defaults();

  if (!pp) {
    _THROW_LBANN_EXCEPTION_(get_type(), " construction error: no image processor");
  }

  replicate_processor(*pp);
}

void data_reader_jag_conduit_hdf5::copy_members(const data_reader_jag_conduit_hdf5& rhs) {
  //todo: make m_jag_store a shared pointer
  m_jag_store = rhs.m_jag_store;
  m_owns_jag_store = rhs.m_owns_jag_store;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  //set_linearized_image_size();
  //m_num_img_srcs = rhs.m_num_img_srcs;
  m_is_data_loaded = rhs.m_is_data_loaded;
  m_scalar_keys = rhs.m_scalar_keys;
  m_input_keys = rhs.m_input_keys;
  m_success_map = rhs.m_success_map;

  if (rhs.m_pps.size() == 0u || !rhs.m_pps[0]) {
    _THROW_LBANN_EXCEPTION_(get_type(), " construction error: no image processor");
  }

  replicate_processor(*rhs.m_pps[0]);

  //m_data = rhs.m_data;
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
/*
  m_independent.assign(1u, Undefined);
  m_dependent.assign(1u, Undefined);
  set_linearized_image_size();
  m_num_img_srcs = 1u;
  m_is_data_loaded = false;
  m_num_labels = 0;
  m_scalar_keys.clear();
  m_input_keys.clear();
  m_uniform_input_type = false;
*/
}

/// Replicate image processor for each OpenMP thread
bool data_reader_jag_conduit_hdf5::replicate_processor(const cv_process& pp) {
  const int nthreads = omp_get_max_threads();
  m_pps.resize(nthreads);

  // Construct thread private preprocessing objects out of a shared pointer
  #pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < nthreads; ++i) {
    //auto ppu = std::make_unique<cv_process>(pp); // c++14
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

void data_reader_jag_conduit_hdf5::load() {
  if(m_gan_labelling) {
    m_num_labels=2;
  }

  if (is_master()) {
    std::cout << "JAG load GAN m_gan_labelling : label_value "
              << m_gan_labelling <<" : " << m_gan_label_value << std::endl;
  }

  bool setup_jag_store = true;
  options *opts = options::get();
  if (is_master()) std::cerr << "data_reader_jag_conduit_hdf5::load() - getting ptrs to data_readers\n";
  std::vector<void*> p = opts->get_ptrs();
  for (auto t : p) {
    data_reader_jag_conduit_hdf5 *other = static_cast<data_reader_jag_conduit_hdf5*>(t);
    if (other == nullptr) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: dynamic_cast<data_reader_jag_conduit_hdf5*> failed");
    }
    if (other->get_role() == get_role()) {
      if (is_master()) std::cerr << "data_reader_jag_conduit_hdf5::load() - found compatible reader; role: " <<  get_role() << "\n";
      m_jag_store = other->get_jag_store();
      m_owns_jag_store = false;
      setup_jag_store = false;
      break;
    }
  }

  if (setup_jag_store) {
    m_jag_store = new jag_store;
    //m_jag_store = std::make_shared<jag_store>(new jag_store);
  
    m_jag_store->set_image_size(m_image_height * m_image_width);
  
    // for selecting images, per Luc's advise
    m_emi_selectors.insert("(0.0, 0.0)");
    m_emi_selectors.insert("(90.0, 0.0)");
    m_emi_selectors.insert("(90.0, 78.0)");
  
    //const std::string data_dir = add_delimiter(get_file_dir());
    //const std::string conduit_file_name = get_data_filename();
    const std::string pattern = get_file_dir();
    std::vector<std::string> names = glob(pattern);
    if (names.size() < 1) {
      _THROW_LBANN_EXCEPTION_(get_type(), " failed to get data filenames");
    }
  
    if (m_first_n > 0) {
      _THROW_LBANN_EXCEPTION_(_CN_, "load() does not support first_n feature.");
    }
  
    if (m_max_files_to_load > 0) {
      if (m_max_files_to_load < names.size()) {
        names.resize(m_max_files_to_load);
      }
    }
  
    m_jag_store->set_comm(m_comm);
    m_jag_store->load_inputs();
    //m_jag_store.load_scalars();
  
    std::vector<std::string> image_names;
    for (auto t : m_emi_selectors) {
      image_names.push_back(t);
    }
    m_jag_store->load_images(image_names);
    m_jag_store->setup(names);
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
#if 0
  size_t sz = 0u;
  for (const auto t: m_dependent) {
    if (t == Undefined) {
      continue;
    }
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
#endif
  return 0;
}

std::vector<size_t> data_reader_jag_conduit_hdf5::get_linearized_data_sizes() const {
  return m_jag_store->get_linearized_data_sizes();
}

std::vector<size_t> data_reader_jag_conduit_hdf5::get_linearized_response_sizes() const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  std::vector<size_t> r;
  return r;
#if 0
  std::vector<size_t> all_dim;
  all_dim.reserve(m_dependent.size());
  for (const auto t: m_dependent) {
    if (t == Undefined) {
      continue;
    }
    all_dim.push_back(get_linearized_size(t));
  }
  return all_dim;
#endif
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
}


std::string data_reader_jag_conduit_hdf5::to_string(const variable_t t) {
  switch (t) {
    case Undefined:  return "Undefined";
    case JAG_Image:  return "JAG_Image";
    case JAG_Scalar: return "JAG_Scalar";
    case JAG_Input:  return "JAG_Input";
  }
  return "Undefined";
}

std::string data_reader_jag_conduit_hdf5::to_string(const std::vector<data_reader_jag_conduit_hdf5::variable_t>& vec) {
  std::string str("[");
  for (const auto& el: vec) {
    str += ' ' + data_reader_jag_conduit_hdf5::to_string(el);
  }
  str += " ]";
  return str;
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
  return m_jag_store->check_sample_id(sample_id);
}

std::vector< std::pair<size_t, const data_reader_jag_conduit_hdf5::ch_t*> >
data_reader_jag_conduit_hdf5::get_image_ptrs(const size_t sample_id) const {
  if (sample_id >= m_success_map.size()) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_images() : invalid sample index");
  }

  std::vector< std::pair<size_t, const ch_t*> >image_ptrs;
#if 0
  std::unordered_map<int, std::string>::const_iterator it = m_success_map.find(sample_id);

  for (auto t : m_emi_selectors) {
    std::string img_key = it->second + "/outputs/images/" + t + "/0.0/emi";
    const conduit::Node & n_image = get_conduit_node(img_key);
    conduit::float32_array emi = n_image.value();
    const size_t num_pixels = emi.number_of_elements();
    const ch_t* emi_data = n_image.value();
    image_ptrs.push_back(std::make_pair(num_pixels, emi_data));
  }  
#endif
  return image_ptrs;
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

std::vector<cv::Mat> data_reader_jag_conduit_hdf5::get_cv_images(const size_t sample_id) const {
  const std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>> &raw_images = m_jag_store->fetch_images(sample_id);
  std::vector< std::pair<size_t, const ch_t*> > img_ptrs(raw_images.size());
  size_t num_pixels = get_linearized_image_size();
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

std::vector<data_reader_jag_conduit_hdf5::ch_t> data_reader_jag_conduit_hdf5::get_images(const size_t sample_id) const {
  std::vector< std::pair<size_t, const ch_t*> > img_ptrs(get_image_ptrs(sample_id));
  std::vector<ch_t> images;
  images.reserve(get_linearized_image_size());

  for (const auto& img: img_ptrs) {
    const size_t num_pixels = img.first;
    const ch_t* ptr = img.second;
    images.insert(images.end(), ptr, ptr + num_pixels);
  }

  return images;
}

std::vector<data_reader_jag_conduit_hdf5::scalar_t> data_reader_jag_conduit_hdf5::get_scalars(const size_t sample_id) const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  std::vector<data_reader_jag_conduit_hdf5::scalar_t> r;
  return r;
#if 0
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_scalars() : invalid sample index");
  }

  std::vector<scalar_t> scalars;
  scalars.reserve(m_scalar_keys.size());

  for(const auto key: m_scalar_keys) {
    std::unordered_map<int, std::string>::const_iterator t2 = m_success_map.find(sample_id);
    std::string scalar_key = t2->second + "/outputs/scalars/" + key;
    const conduit::Node & n_scalar = get_conduit_node(scalar_key);
    // All the scalar output currently seems to be scalar_t
    //add_val(key, n_scalar, scalars);
    scalars.push_back(static_cast<scalar_t>(n_scalar.to_value()));
  }
  return scalars;
#endif
}

std::vector<data_reader_jag_conduit_hdf5::input_t> data_reader_jag_conduit_hdf5::get_inputs(const size_t sample_id) const {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  std::vector<data_reader_jag_conduit_hdf5::input_t> r;
  return r;
#if 0
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_inputs() : invalid sample index");
  }

  std::vector<input_t> inputs;
  inputs.reserve(m_input_keys.size());

  // automatically determine which method to use based on if all the variables are of input_t
  if (m_uniform_input_type) {
    for(const auto key: m_input_keys) {
      std::unordered_map<int, std::string>::const_iterator t2 = m_success_map.find(sample_id);
      std::string input_key = t2->second + "/inputs/" + key;
      const conduit::Node & n_input = get_conduit_node(input_key);
      inputs.push_back(n_input.value()); // less overhead
    }
  } else {
    for(const auto key: m_input_keys) {
      std::unordered_map<int, std::string>::const_iterator t2 = m_success_map.find(sample_id);
      std::string input_key = t2->second + "/inputs/" + key;
      //const conduit::Node & n_input = get_conduit_node(input_key);
      //add_val(key, n_input, inputs); // more overhead but general
    }
  }
  return inputs;
#endif
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

bool data_reader_jag_conduit_hdf5::fetch(CPUMat& X, int data_id, int mb_idx, int tid,
  const data_reader_jag_conduit_hdf5::variable_t vt, const std::string tag) {
  switch (vt) {
    case JAG_Image: {
      const std::vector<size_t> sizes(get_num_img_srcs(), get_linearized_image_size());
      std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
      std::vector<cv::Mat> images = get_cv_images(data_id);

      if (images.size() != get_num_img_srcs()) {
        _THROW_LBANN_EXCEPTION2_(_CN_, "fetch() : the number of images is not as expected", \
          std::to_string(images.size()) + "!=" + std::to_string(get_num_img_srcs()));
      }

      for(size_t i=0u; i < get_num_img_srcs(); ++i) {
        int width, height, img_type;
        image_utils::process_image(images[i], width, height, img_type, *(m_pps[tid]), X_v[i]);
      }
      break;
    }
    case JAG_Scalar: {
      const std::vector<scalar_t> scalars(get_scalars(data_id));
      set_minibatch_item<scalar_t>(X, mb_idx, scalars.data(), get_linearized_scalar_size());
      break;
    }
    case JAG_Input: {
      const std::vector<input_t> inputs(get_inputs(data_id));
      set_minibatch_item<input_t>(X, mb_idx, inputs.data(), get_linearized_input_size());
      break;
    }
    default: { // includes Undefined case
      _THROW_LBANN_EXCEPTION_(_CN_, "fetch_" + tag + "() : unknown or undefined variable type");
    }
  }
  return true;
}

bool data_reader_jag_conduit_hdf5::fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) {
  bool ok = true;

  const std::vector<size_t> & sizes = get_linearized_data_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);

  size_t i = 0;
  const std::vector<data_reader_jag_conduit_hdf5::input_t> &inputs = m_jag_store->fetch_inputs(data_id);
  set_minibatch_item<data_reader_jag_conduit_hdf5::input_t>(X_v[i++], 0, inputs.data(), m_jag_store->get_linearized_input_size());

  std::vector<cv::Mat> images = get_cv_images(data_id);

  if (images.size() != get_num_img_srcs()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: the number of images is not as expected " + std::to_string(images.size()) + "!=" + std::to_string(get_num_img_srcs()));
  }  

  for(size_t k=0u; k < get_num_img_srcs(); ++k) {
    int width, height, img_type;
    image_utils::process_image(images[k], width, height, img_type, *(m_pps[tid]), X_v[i]);
   }

  return ok;
}

bool data_reader_jag_conduit_hdf5::fetch_response(CPUMat& X, int data_id, int mb_idx, int tid) {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: not implemented");
  return true;
#if 0
  std::vector<size_t> sizes = get_linearized_response_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    ok = fetch(X_v[i], data_id, 0, tid, m_dependent[i], "response");
  }
  return ok;
#endif
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


} // end of namespace lbann

#undef _CN_
#endif // #ifdef LBANN_HAS_CONDUIT
