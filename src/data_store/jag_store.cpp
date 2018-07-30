#include "lbann/data_store/jag_store.hpp"
#include "lbann/utils/exception.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include "hdf5.h"
#include <unordered_set>

namespace lbann {

jag_store::jag_store() 
  : m_test_mode_is_set(false), 
    m_is_setup(false),
    m_load_inputs(false),
    m_load_scalars(false),
    m_image_size(0)
  {}

void jag_store::load_inputs() {
  m_load_inputs = true;
}

void jag_store::load_scalars() {
  m_load_scalars = true;
}


void jag_store::load_images(const std::vector<std::string> &keys) {
  for (auto t : keys) {
    m_images_to_load.insert(t);
  }
}

jag_store::input_t jag_store::fetch_input(size_t sample_id, const std::string &key) const {
  std::unordered_map<std::string, std::vector<input_t>>::const_iterator it = m_data_inputs.find(key);
  if (it == m_data_inputs.end()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: nothing known about the requested input key: " + key);
  }
  if (sample_id >= it->second.size()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: largest sample_id we know about is " + std::to_string(it->second.size()) + "; you asked for sample_id: " + std::to_string(sample_id));
  }
  return it->second[sample_id];
}


jag_store::scalar_t jag_store::fetch_scalar(size_t sample_id, const std::string &key) const {
  std::unordered_map<std::string, std::vector<scalar_t>>::const_iterator it = m_data_scalars.find(key);
  if (it == m_data_scalars.end()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: nothing known about the requested scalar key: " + key);
  }
  if (sample_id >= it->second.size()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: largest sample_id we know about is " + std::to_string(it->second.size()) + "; you asked for sample_id: " + std::to_string(sample_id));
  }
  return it->second[sample_id];
}

void jag_store::setup(
  const std::vector<std::string> conduit_filenames,
  bool num_stores,
  int my_rank) {

  bool master = m_comm->am_world_master();
  if (master) std::cerr << "starting jag_store::setup\n";

  std::string test_file;
  std::string test_sample_id;

  // get the sample_ids of successful samples; mostly need to do this
  // to figure out the proper memory allocation for m_data
  std::unordered_set<std::string> use_me;
  conduit::Node n_ok;
  size_t failed = 0;
  for (size_t j = my_rank; j<conduit_filenames.size(); j+= num_stores) {
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (auto t : cnames) {
      const std::string key = "/" + t + "/performance/success";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
      int success = n_ok.to_int64();
      if (success == 1) {
        use_me.insert(t);
        if (test_file == "") {
          test_file = conduit_filenames[j];
          test_sample_id = "/" + t;
        }
      } else {
        ++failed;
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
  m_num_samples = use_me.size();
  if (master) {
    std::cout << "jag_store::setup; successful samples: " << m_num_samples << " failed samples: " << failed << std::endl;
  }

  //optionally get input and scalar names
  if (m_load_inputs || m_load_scalars) {
    if (test_file == "" || test_sample_id == "") {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to find any good samples");
    }
    if (m_load_inputs) {
      get_default_keys(test_file, test_sample_id, "inputs", master);
    }
    if (m_load_scalars) {
      get_default_keys(test_file, test_sample_id, "scalars", master);
    }
  }

  //allocate some memory
  for (auto t : m_inputs_to_load) {
    m_data_inputs[t].reserve(m_num_samples);
  }
  for (auto t : m_scalars_to_load) {
    m_data_scalars[t].reserve(m_num_samples);
  }

  //load the data
  conduit::Node node;
  for (size_t j = my_rank; j<conduit_filenames.size(); j+= num_stores) {
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (auto t : cnames) {
      if (use_me.find(t) != use_me.end()) {
        for (auto input_name : m_inputs_to_load) {
          const std::string key = "/" + t + "/inputs/" + input_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if scalar_t changes
          m_data_inputs[input_name].push_back( node.to_float64() );
        }
        for (auto scalar_name : m_scalars_to_load) {
          const std::string key = "/" + t + "/inputs/" + scalar_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if input_t changes
          m_data_inputs[scalar_name].push_back( node.to_float64() );
        }
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
}

void jag_store::get_default_keys(std::string &filename, std::string &sample_id, std::string key1, bool master) {
  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(filename);
  conduit::Node n2;

  const std::string key = "/" + sample_id + "/" + key1;
  std::vector<std::string> children;
  conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, key, children);
  for (auto t : children) {
    if (master) std::cout << "next key: " << t << std::endl;
    if (key1 == "inputs") {
      m_inputs_to_load.insert(t);
    } else {
      m_scalars_to_load.insert(t);
    }
  }
  conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
}

size_t jag_store::get_linearized_data_size() const {
std::cerr << "jag_store::get_linearized_data_size: " << m_inputs_to_load.size()
<< "  m_scalars_to_load.size(): " << m_scalars_to_load.size()
<< " m_images_to_load.size(): " <<m_images_to_load.size()
<< " get_linearized_image_size: " << get_linearized_image_size() << "\n";

  return m_inputs_to_load.size()
           + m_scalars_to_load.size()
           + m_images_to_load.size() * get_linearized_image_size();
}

void jag_store::set_image_size(size_t n) {
  m_image_size = n;
}

size_t jag_store::get_linearized_image_size() const {
  return m_image_size;
}

size_t jag_store::get_linearized_scalar_size() const {
  return m_scalars_to_load.size();
}

size_t jag_store::get_linearized_input_size() const {
  return m_inputs_to_load.size();
}

std::vector<size_t> jag_store::get_linearized_data_sizes() const {
  std::vector<size_t> all_dim;
  for (auto t : m_images_to_load) {
    all_dim.push_back(get_linearized_image_size());
  }
  for (auto t : m_inputs_to_load) {
    all_dim.push_back(get_linearized_input_size());
  }
  for (auto t : m_scalars_to_load) {
    all_dim.push_back(get_linearized_scalar_size());
  }
  return all_dim;
}

bool jag_store::check_sample_id(const size_t sample_id) const {
 return sample_id <= m_num_samples;
}

size_t jag_store::get_num_img_srcs() const {
  return m_data_images.size();
}

} // namespace lbann
