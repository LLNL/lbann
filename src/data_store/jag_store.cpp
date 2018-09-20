#include "lbann/data_store/jag_store.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/options.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include "hdf5.h"
#include <unordered_set>

namespace lbann {

jag_store::jag_store() 
  : m_is_setup(false),
    m_load_inputs(false),
    m_load_scalars(false),
    m_image_size(0)
  { 
  }

void jag_store::load_inputs() {
  m_load_inputs = true;
}

void jag_store::load_scalars() {
  m_load_scalars = true;
}


void jag_store::load_images(const std::vector<std::string> &keys) {
  for (auto t : keys) {
    m_images_to_load.push_back(t);
  }
}

void jag_store::setup(
  const std::vector<std::string> conduit_filenames,
  bool num_stores,
  int my_rank) {
  bool master = m_comm->am_world_master();
  double tm2 = get_time();
  if (master) std::cerr << "starting jag_store::setup for " << conduit_filenames.size() << " conduit files\n";

  m_conduit_filenames = conduit_filenames;

  // quick hack to get every processor to read a unique
  // subset of the data - under development; not tested
  if (options::get()->has_string("every_n")) {
    my_rank = m_comm->get_rank_in_world();
    num_stores = m_comm->get_procs_in_world();
  }

  if (m_image_size == 0) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: image_size = 0; probably set_image_size() has not been called");
  }

  std::string test_file;
  std::string test_sample_id;

  // get the sample_ids of successful samples; mostly need to do this
  // to figure out the proper memory allocation for m_data
  //
  // todo TODO p_0 should compute and bcast, instead of having all 
  // procs read the same files
  double tm1 = get_time();
  conduit::Node n_ok;
  size_t failed = 0;
  std::unordered_set<std::string> valid_samples;
  if (master) std::cerr << "jag_store::setup; scanning data files for success flag\n";
  for (size_t j = my_rank; j<conduit_filenames.size(); j+= num_stores) {
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (auto t : cnames) {
      const std::string key = "/" + t + "/performance/success";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
      int success = n_ok.to_int64();
      if (success == 1) {
        valid_samples.insert(t);
        m_sample_id_to_filename_idx.push_back(j);
        m_sample_id_to_string_id.push_back(t);
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
  m_num_samples = valid_samples.size();
  if (master) {
    std::cerr << "jag_store::setup; successful samples: " << m_num_samples << " failed samples: " << failed << " time to test for success: " << get_time() - tm1 << std::endl;
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
      get_default_keys(test_file, test_sample_id, "outputs/scalars", master);
    }
  }

  //allocate memory
  size_t nthreads = omp_get_max_threads();
  m_data_inputs.resize(nthreads);
  m_data_scalars.resize(nthreads);
  m_data_images.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    m_data_inputs[j].reserve(m_inputs_to_load.size());
    m_data_scalars[j].reserve(m_scalars_to_load.size());
    m_data_images[j].resize(m_images_to_load.size());
  }
  for (size_t j=0; j<m_data_images.size(); j++) {
    m_data_images[j].resize(m_images_to_load.size());
    for (size_t i=0; i<m_images_to_load.size(); i++) {
      m_data_images[j][i].resize(m_image_size);
    }
  }

#if 0
  //load the data
  size_t idx = 0;
  conduit::Node node;
  for (size_t j = my_rank; j<conduit_filenames.size(); j+= num_stores) {
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (auto t : cnames) {
      m_id_to_name[idx] = t;
      if (m_valid_samples.find(t) != m_valid_samples.end()) {
        for (auto input_name : m_inputs_to_load) {
          const std::string key = "/" + t + "/inputs/" + input_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if scalar_t changes
          m_data_inputs[idx].push_back( node.to_float64() );
        }
        for (auto scalar_name : m_scalars_to_load) {
          const std::string key = "/" + t + "/outputs/scalars/" + scalar_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if input_t changes
          m_data_inputs[idx].push_back( node.to_float64() );
        }
        size_t k = 0;
        for (auto image_name : m_images_to_load) {
          const std::string key = "/" + t + "/outputs/images/" + image_name + "/0.0/emi";
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);


          conduit::float32_array emi = node.value();
          const size_t image_size = emi.number_of_elements();

          //conduit::DataType dtype = node.dtype();
          //size_t image_size = dtype.number_of_elements();
          if (image_size != m_image_size) {
            throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: m_image_size = " + std::to_string(m_image_size) + " but conduit image size is " + std::to_string(image_size));
          }
          //this is fragile; will break if ch_t changes
          //float *p = node.value();
          for (size_t h=0; h<image_size; h++) {
            m_data_images[idx][k][h] = emi[h];
            //m_data_images[idx][k][h] = p[h];
          }
          ++k;
        }
        ++idx;
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
#endif
  build_data_sizes();
  m_is_setup = true;
  if (master) {
    std::cerr << "jag_store::setup time: " << get_time() - tm2 << std::endl;
  }
}

void jag_store::get_default_keys(std::string &filename, std::string &sample_id, std::string key1, bool master) {
  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(filename);
  conduit::Node n2;

  const std::string key = "/" + sample_id + "/" + key1;

  std::vector<std::string> children;
  conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, key, children);
  for (auto t : children) {
    if (key1 == "inputs") {
      m_inputs_to_load.push_back(t);
    } else {
      m_scalars_to_load.push_back(t);
    }
  }
  conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
}

size_t jag_store::get_linearized_data_size() const {
  return m_inputs_to_load.size()
           + m_scalars_to_load.size()
           + m_images_to_load.size() * get_linearized_image_size();
}

void jag_store::build_data_sizes() {
  for (auto t : m_inputs_to_load) {
    m_data_sizes.push_back(get_linearized_input_size());
  }
  for (auto t : m_scalars_to_load) {
    m_data_sizes.push_back(get_linearized_scalar_size());
  }
  for (auto t : m_images_to_load) {
    m_data_sizes.push_back(get_linearized_image_size());
  }
}

void jag_store::load_data(int data_id, int tid) {
  check_sample_id(data_id);
  std::string &filename = m_conduit_filenames[data_id];
  std::string &sample_name = m_sample_id_to_string_id[data_id];

  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename );
  conduit::Node node;
  for (auto input_name : m_inputs_to_load) {
    const std::string key = "/" + sample_name + "/inputs/" + input_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if scalar_t changes
    m_data_inputs[tid].push_back( node.to_float64() );
  }
  for (auto scalar_name : m_scalars_to_load) {
    const std::string key = "/" + sample_name + "/outputs/scalars/" + scalar_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if input_t changes
    m_data_inputs[tid].push_back( node.to_float64() );
  }
  size_t k = 0;
  for (auto image_name : m_images_to_load) {
    const std::string key = "/" + sample_name + "/outputs/images/" + image_name + "/0.0/emi";
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    conduit::float32_array emi = node.value();
    const size_t image_size = emi.number_of_elements();
    if (image_size != m_image_size) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: m_image_size = " + std::to_string(m_image_size) + " but conduit image size is " + std::to_string(image_size));
    }
    //this is fragile; will break if ch_t changes
    //can probably eliminate this next loop ...
    for (size_t h=0; h<image_size; h++) {
      m_data_images[tid][k][h] = emi[h];
    }
    ++k;
  }
  conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
}

} // namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT
