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

#include "lbann/models/model.hpp"
#include "lbann/data_readers/image_utils.hpp"

namespace lbann {

template<class Ch_t, class Conduit_ch_t , class Scalar_t, class Input_t, class TimeSeries_t>
data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::data_reader_jag_conduit(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : data_reader_conduit(pp, shuffle) {
  set_defaults();
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
void data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::copy_members
(const data_reader_jag_conduit& rhs, const std::vector<int>& ds_sample_move_list) {
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::data_reader_jag_conduit(const data_reader_jag_conduit& rhs)
  : data_reader_conduit(rhs) {
  copy_members(rhs);
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::data_reader_jag_conduit(const data_reader_jag_conduit& rhs, const std::vector<int>& ds_sample_move_list)
  : data_reader_conduit(rhs) {
  copy_members(rhs, ds_sample_move_list);
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>  &data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::operator=(const data_reader_jag_conduit& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  data_reader_conduit::operator=(rhs);

  copy_members(rhs);

  return (*this);
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::~data_reader_jag_conduit() {
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
void data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::set_defaults() {
}


template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
void data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::check_image_data() {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  if (m_sample_list.empty()) {
    return;
  }

  size_t first_idx = (m_sample_list[0]).first;
  if (!has_conduit_path(first_idx, "")) {
    LBANN_ERROR("check_image_data() : no sample by " + m_sample_list[first_idx].second);
    return;
  }
  conduit::Node n_imageset;
  load_conduit_node(first_idx, m_output_image_prefix, n_imageset);
  if (static_cast<size_t>(n_imageset.number_of_children()) == 0u) {
    return;
  }
  if (m_emi_image_keys.size() == 0u) {
    return;
  }
  for (const auto& emi_tag: m_emi_image_keys) {
    if (!has_conduit_path(first_idx, m_output_image_prefix + emi_tag)) {
      LBANN_ERROR("check_image_data() : no emi image by " + emi_tag);
      return;
    }
  }
  conduit::Node n_image;
  load_conduit_node(first_idx, m_output_image_prefix + m_emi_image_keys[0], n_image);
  Conduit_ch_t emi = n_image.value();

  if (m_image_linearized_size != static_cast<size_t>(emi.number_of_elements())) {
    if ((m_image_width == 0) && (m_image_height == 0)) {
      m_image_height = 1;
      m_image_width = static_cast<int>(emi.number_of_elements());
      m_image_num_channels = 1;
      set_linearized_image_size();
    } else {
      std::string msg = "expected linearized emi image size: "
                      + std::to_string(emi.number_of_elements()) + '\n';
      LBANN_ERROR(msg + get_description());
    }
  }

  if (m_image_normalization_params.empty()) {
    m_image_normalization_params.assign(m_emi_image_keys.size()*m_image_num_channels, linear_transform_t(1.0, 0.0));
  } else if (m_image_normalization_params.size() != static_cast<size_t>(m_image_num_channels)) {
    LBANN_ERROR("Incorrect number of image normalization parameter sets!" \
                                + std::to_string(m_image_normalization_params.size()) + " != " \
                                + std::to_string(m_image_num_channels));
  }
#if defined(LBANN_DEBUG)
  std::cout << "image normalization parameters: " << std::endl;
  for (size_t i = 0u, s = 0u; s < m_emi_image_keys.size(); ++s) {
    for (int c = 0; c < m_image_num_channels; ++c) {
      const auto& param = m_image_normalization_params[i*m_image_num_channels + c];
      std::cout << " scale: \t" << param.first << " \tbias: \t" << param.second
                << " \t" << m_emi_image_keys[s] << ":C" << c << std::endl;
    }
  }
#endif
}


template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
std::vector< std::vector<Ch_t> >
data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::get_image_data(const size_t sample_id, conduit::Node& sample) const {
  float x = 1.0;
  Ch_t *x2 = &x;
  std::cout << "x2: " << *x2 << std::endl;

  std::vector< std::vector<Ch_t> > image_ptrs;
  image_ptrs.reserve(m_emi_image_keys.size());

  for (const auto& emi_tag : m_emi_image_keys) {
    const std::string conduit_field = m_output_image_prefix + emi_tag;
    const std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
    if(sample[conduit_obj].schema().dtype().is_empty()) {
      if (data_store_active()) {
        LBANN_ERROR("Unable to find field " + conduit_obj
                    + " in conduit node: " + std::to_string(sample_id));
      }
      conduit::Node n_image;
      bool from_file = load_conduit_node(sample_id, conduit_field, n_image);
      if (from_file) {
        sample[conduit_obj].set(n_image);
      } else {
        sample = n_image;
      }
    }

 //   Conduit_ch_t emi = sample[conduit_obj].value();
//    const size_t num_vals = emi.number_of_elements();
 //   const DataType dt = sample.dtype();
    //std::cout << "dtype name: " << dt.name() << " n elts: " << dt.number_of_elements() << " is_float: " << dt.is_float() << " is double: " << dt.is_double() << "\n";
    //const void* emi_data = sample[conduit_obj].as_float32_ptr();
    //const Ch_t* data = emi_data;
    //const Ch_t* emi_data = sample[conduit_obj].value();
    //image_ptrs.emplace_back(emi_data, emi_data + num_vals);
  }

  return image_ptrs;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
cv::Mat data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::cast_to_cvMat(
  const std::pair<size_t, const Ch_t*> img, const int height, const int num_ch) {
  const int num_pixels = static_cast<int>(img.first);
  const Ch_t* ptr = img.second;

  // add a zero copying view to data
  using InputBuf_T = cv_image_type<Ch_t>;
  const cv::Mat image(num_pixels, 1, InputBuf_T::T(1u),
                      reinterpret_cast<void*>(const_cast<Ch_t*>(ptr)));
  // reshape the image. Furter need to clone (deep-copy) the image
  // to preserve the constness of the original data
  return (image.reshape(num_ch, height));
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
std::vector<cv::Mat> data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::get_cv_images(const size_t sample_id, conduit::Node& sample) const {
  const std::vector< std::vector<Ch_t> > img_data(get_image_data(sample_id, sample));
  std::vector<cv::Mat> images;

  if (m_split_channels) {
    images.reserve(img_data.size()*m_image_num_channels);
    for (size_t i = 0u; i < img_data.size(); ++i) {
      const auto& img = img_data[i];
      cv::Mat ch[m_image_num_channels];
      cv::split(cast_to_cvMat(std::make_pair(img.size(), img.data()), m_image_height, m_image_num_channels), ch);
      for(int c = 0; c < m_image_num_channels; ++c) {
    #if 1 // with normalization
        image_normalization(ch[c], i, static_cast<size_t>(c));
    #endif
        images.emplace_back(ch[c].clone());
      }
    }
  } else {
    images.reserve(img_data.size());
    for (size_t i = 0u; i < img_data.size(); ++i) {
      const auto& img = img_data[i];
    #if 1 // with normalization
      cv::Mat ch[m_image_num_channels];
      cv::split(cast_to_cvMat(std::make_pair(img.size(), img.data()), m_image_height, m_image_num_channels), ch);
      for(int c = 0; c < m_image_num_channels; ++c) {
        image_normalization(ch[c], i, static_cast<size_t>(c));
      }
      cv::Mat img_normalized;
      cv::merge(ch, m_image_num_channels, img_normalized);
      images.emplace_back(img_normalized);
    #else
      images.emplace_back(cast_to_cvMat(std::make_pair(img.size(), img.data()), m_image_height, m_image_num_channels).clone());
    #endif
    }
  }
  return images;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
std::vector<Ch_t> data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::get_images(const size_t sample_id, conduit::Node& sample) const {
  std::vector< std::vector<Ch_t> > img_data(get_image_data(sample_id, sample));
  std::vector<Ch_t> images;

  if (m_split_channels) {
    images.resize(get_linearized_size(JAG_Image));
    size_t i = 0u;
    size_t j = 0u;
    for (const auto& img: img_data) {
      const Ch_t * const ptr_end = img.data() + img.size();
      for (int c=0; c < m_image_num_channels; ++c) {
        const auto& tr = m_image_normalization_params.at(c);
        for (const Ch_t* ptr = img.data() + c; ptr < ptr_end; ptr += m_image_num_channels) {
        #if 1 // with normalization
          images[i++] = cv::saturate_cast<Ch_t>(*ptr * tr.first + tr.second);
        #else
          images[i++] = *ptr;
        #endif
        }
      }
      j ++;
    }
  } else {
    images.reserve(get_linearized_size(JAG_Image));
    for (const auto& img: img_data) {
    #if 1 // with normalization
      // TODO: normalization needed
      LBANN_ERROR("get_images() : normalization not implemented yet");
      (void) img;
    #else
      images.insert(images.end(), img.cbegin(), ptr + img.cend());
    #endif
    }
  }

  return images;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
std::vector<Scalar_t> data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::get_scalars(const size_t sample_id, conduit::Node& sample) const {
  std::vector<Scalar_t> scalars;
  scalars.reserve(m_scalar_keys.size());

  auto tr = m_scalar_normalization_params.cbegin();

  for(const auto key: m_scalar_keys) {
    std::string conduit_field = m_output_scalar_prefix + key;
    std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
    if(sample[conduit_obj].schema().dtype().is_empty()) {
      if (data_store_active()) {
        LBANN_ERROR("Unable to find field " + conduit_obj
                    + " in conduit node: " + std::to_string(sample_id));
      }
      conduit::Node n_scalar;
      bool from_file = load_conduit_node(sample_id, conduit_field, n_scalar);
      if (from_file) {
        sample[conduit_obj].set(n_scalar);
      } else {
        sample = n_scalar;
      }
    }
    const Scalar_t val_raw = static_cast<Scalar_t>(sample[conduit_obj].to_value());
    const Scalar_t val = static_cast<Scalar_t>(val_raw * tr->first + tr->second);
    scalars.push_back(val);
    tr ++;
  }
  return scalars;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
std::vector<Input_t> data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::get_inputs(const size_t sample_id, conduit::Node& sample) const {
  std::vector<Input_t> inputs;
  inputs.reserve(m_input_keys.size());

  // The sequence of normalization parameters should follow the same order as
  // that of the variable keys.
  auto tr = m_input_normalization_params.cbegin();

  // automatically determine which method to use based on if all the variables are of Input_t
  if (m_uniform_input_type) {
    // avoid some overhead by taking advantage of the fact that all the variables are of the same type
    for(const auto key: m_input_keys) {
      const std::string conduit_field = m_input_prefix + key;
      const std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
      if(sample[conduit_obj].schema().dtype().is_empty()) {
        if (data_store_active()) {
          LBANN_ERROR("Unable to find field " + conduit_obj
                      + " in conduit node: " + std::to_string(sample_id));
        }
        conduit::Node n_input;
        bool from_file = load_conduit_node(sample_id, conduit_field, n_input);
        if (from_file) {
          sample[conduit_obj].set(n_input);
        } else {
          sample = n_input;
        }
      }
      const Input_t val_raw = static_cast<Input_t>(sample[conduit_obj].value());
      const Input_t val = static_cast<Input_t>(val_raw * tr->first + tr->second);
      inputs.push_back(val);
      tr ++;
    }
  } else {
    for(const auto key: m_input_keys) {
      const std::string conduit_field = m_input_prefix + key;
      const std::string conduit_obj = '/' + LBANN_DATA_ID_STR(sample_id) + '/' + conduit_field;
      if(sample[conduit_obj].schema().dtype().is_empty()) {
        if (data_store_active()) {
          LBANN_ERROR("Unable to find field " + conduit_obj
                      + " in conduit node: " + std::to_string(sample_id));
        }
        conduit::Node n_input;
        bool from_file = load_conduit_node(sample_id, conduit_field, n_input);
        if (from_file) {
          sample[conduit_obj].set(n_input);
        } else {
          sample = n_input;
        }
      }
      add_val(key, sample[conduit_obj], inputs); // more overhead but general
      Input_t& val = inputs.back();
      val = static_cast<Input_t>(val * tr->first + tr->second);
      tr ++;
    }
  }

  return inputs;
}


template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
bool data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::fetch(CPUMat& X, int data_id, conduit::Node& sample, int mb_idx, int tid,
  const data_reader_jag_conduit::variable_t vt, const std::string tag) {
  switch (vt) {
    case JAG_Image: {
      const size_t num_images = get_num_img_srcs()
                              * static_cast<size_t>(m_split_channels? m_image_num_channels : 1u);
      const size_t image_size = m_split_channels? get_linearized_1ch_image_size() : get_linearized_image_size();
      const std::vector<size_t> sizes(num_images, image_size);
      std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
      std::vector<cv::Mat> images = get_cv_images(data_id, sample);

      if (images.size() != num_images) {
        LBANN_ERROR("fetch() : the number of images is not as expected" + 
          std::to_string(images.size()) + "!=" + std::to_string(num_images));
      }

      for(size_t i=0u; i < num_images; ++i) {
        int width, height, img_type;
        image_utils::process_image(images[i], width, height, img_type, *(m_pps[tid]), X_v[i]);
      }
      break;
    }
    case JAG_Scalar: {
      const std::vector<Scalar_t> scalars(get_scalars(data_id, sample));
      set_minibatch_item<Scalar_t>(X, mb_idx, scalars.data(), get_linearized_scalar_size());
      break;
    }
    case JAG_Input: {
      const std::vector<Input_t> inputs(get_inputs(data_id, sample));
      set_minibatch_item<Input_t>(X, mb_idx, inputs.data(), get_linearized_input_size());
      break;
    }
    default: { // includes Undefined case
      LBANN_ERROR("fetch_" + tag + "() : unknown or undefined variable type");
    }
  }
  return true;
}


template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
bool data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  int tid = m_io_thread_pool->get_local_thread_id();
  std::vector<size_t> sizes = get_linearized_data_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  // Create a node to hold all of the data
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  }else {
    m_sample_list.open_samples_hdf5_handle(data_id);
  }

  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    // The third argument mb_idx below is 0 because it is for the view of X not X itself
    ok = fetch(X_v[i], data_id, node, 0, tid, m_independent[i], "datum");
  }

  if (priming_data_store()) {
    // Once the node has been populated save it in the data store
    m_data_store->set_conduit_node(data_id, node);
  }

  m_sample_list.close_if_done_samples_hdf5_handle(data_id);
  m_using_random_node.erase(m_io_thread_pool->get_local_thread_id());
  return ok;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
bool data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::fetch_response(CPUMat& X, int data_id, int mb_idx) {
  int tid = m_io_thread_pool->get_local_thread_id();
  std::vector<size_t> sizes = get_linearized_response_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  // Create a node to hold all of the data
  conduit::Node node;
  if (m_data_store != nullptr && m_model->get_epoch() > 0) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  }
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    ok = fetch(X_v[i], data_id, node, 0, tid, m_dependent[i], "response");
  }
  if (m_data_store != nullptr && m_model->get_epoch() == 0) {
    // Once the node has been populated save it in the data store
    if (m_data_store != nullptr) {
      m_data_store->set_conduit_node(data_id, node);
    }
  }
  return ok;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
bool data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  if(m_gan_label_value) Y.Set(m_gan_label_value,mb_idx,1); //fake sample is set to 1; adversarial model
  else { //fake sample (second half of minibatch is set to 0;discriminator model
    //mb_idx < (m_mb_size/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
    mb_idx < (get_current_mini_batch_size()/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
  }
  //Y.Set(m_gan_label_value, mb_idx, 1);
  return true;
}

template<class Ch_t, class Conduit_ch_t, class Scalar_t, class Input_t, class TimeSeries_t>
void data_reader_jag_conduit<Ch_t,Conduit_ch_t,Scalar_t,Input_t,TimeSeries_t>::check_input_keys() {
  //@TODO revisit later -- don't know how to handle this yet
  if (m_data_store != nullptr) {
    return;
  }

  if (m_input_keys.empty()) {
    return;
  }
  if (!m_is_data_loaded) {
    return;
  }
  if (m_sample_list.empty()) {
    //m_input_keys.clear();
    return;
  }

  // If this call is made after loading data, check if the keys

  size_t num_found = 0u;
  std::vector<bool> found(m_input_keys.size(), false);
  std::map<std::string, TypeID> keys_conduit;

  conduit::Node n_input;
  size_t first_idx = (m_sample_list[0]).first;
  load_conduit_node(first_idx, "/inputs", n_input);
  conduit::NodeConstIterator itr = n_input.children();

  while (itr.has_next()) {
    const conduit::Node & n = itr.next();
    keys_conduit.insert(std::pair<std::string, TypeID>(itr.name(), static_cast<TypeID>(n.dtype().id())));
  }

  bool is_Input_t = true;

  for (size_t i=0u; i < m_input_keys.size(); ++i) {
    std::map<std::string, TypeID>::const_iterator it = keys_conduit.find(m_input_keys[i]);
    if (it != keys_conduit.cend()) {
      num_found ++;
      found[i] = true;
      is_Input_t = is_Input_t && is_same_type<Input_t>(it->second);
    }
  }

  if (num_found != m_input_keys.size()) {
    std::string msg = "keys not found:";
    for (size_t i=0u; i < m_input_keys.size(); ++i) {
      if (!found[i]) {
        msg += ' ' + m_input_keys[i];
      }
    }
    LBANN_ERROR("check_input_keys() : " + msg);
  }

  m_uniform_input_type = (m_input_keys.size() == 0u)? false : is_Input_t;

  if (m_input_normalization_params.empty()) {
    m_input_normalization_params.assign(m_input_keys.size(), linear_transform_t(1.0, 0.0));
  } else if (m_input_normalization_params.size() != m_input_keys.size()) {
     LBANN_ERROR("Incorrect number of input normalization parameter sets! " \
                                 + std::to_string(m_input_normalization_params.size()) + " != " \
                                 + std::to_string(m_input_keys.size()));
  }
#if defined(LBANN_DEBUG)
  std::cout << "input normalization parameters: " << std::endl;
  for (size_t i = 0u; i < m_input_normalization_params.size(); ++i) {
    const auto& param = m_input_normalization_params[i];
    std::cout << " scale: \t" << param.first << " \tbias: \t" << param.second << " \t" << m_input_keys[i] << std::endl;
  }
#endif
}

} // end of namespace lbann

