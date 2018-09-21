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
    m_image_size(0),
    m_num_samples(0),
    m_comm(nullptr),
    m_master(false)
  { 
  }


void load_keys(std::vector<std::string> &v, const std::string &keys) {
   std::stringstream s;
   s << keys;
   std::string key;
   while (s >> key) {
     v.push_back(key);
   }
}

void jag_store::load_scalars(const std::string &keys) {
   load_keys(m_scalars_to_use, keys);
}

void jag_store::load_inputs(const std::string &keys) {
   load_keys(m_inputs_to_use, keys);
}

void jag_store::load_image_views(const std::string &keys) {
   //load_keys(m_image_views_to_use, keys);
   size_t last = 0;
   while (true) {
     size_t j1 = keys.find('(', last);
     size_t j2 = keys.find(')', last);
     if (j1 == std::string::npos || j2 == std::string::npos) {
       break;
     }
     std::string key = keys.substr(j1, j2-j1+1);
     m_image_views_to_use.push_back(key);
     last = j2+1;
   }
}

void jag_store::load_image_channels(const std::string &keys) {
   std::stringstream s;
   s << keys;
   int channel;
   while (s >> channel) {
     m_image_channels_to_use.push_back(channel);
   }
}


void jag_store::setup(
  const std::vector<std::string> conduit_filenames,
  data_reader_jag_conduit_hdf5 *reader,
  bool num_stores,
  int my_rank) {

  m_master = m_comm->am_world_master();
  if (m_master) std::cerr << "starting jag_store::setup for " << conduit_filenames.size() << " conduit files\n";
  double tm1 = get_time();
  int num_ranks = m_comm->get_procs_in_world();
  m_reader = reader;
  m_conduit_filenames = conduit_filenames;

  // holding off on my bcast scheme for now, since Brian has a better idea ...
  //bool bcast = true;
  bool bcast = false;

  load_inputs(m_reader->m_input_keys);
  load_scalars(m_reader->m_scalar_keys);
  load_image_views(m_reader->m_image_views);
  load_image_channels(m_reader->m_image_channels);

  if (m_master) {
    std::cout << "\n======================================================\n"
              << "using these inputs: ";
    for (auto t : m_inputs_to_use) {
      std::cout << t << " ";
    }
    std::cout << "\n\nusing these scalars: ";
    for (auto t : m_scalars_to_use) {
      std::cout << t << " ";
    }
    std::cout << "\n\nusing these views: ";
    for (auto t : m_image_views_to_use) {
      std::cout << t << " ";
    }
    std::cout << "\n\nusing these image channels: ";
    for (auto t : m_image_channels_to_use) {
      std::cout << t << " ";
    }
    std::cout << "\n======================================================\n";
  }

  // hack to get every processor to read a unique
  // subset of the data - under development; not tested
  if (options::get()->has_string("every_n") && num_ranks > 1) {
    my_rank = m_comm->get_rank_in_world();
    num_stores = m_comm->get_procs_in_world();
    bcast = false;
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
  // procs read the same files. However, if using "every_n" option
  // this comment is NA
  conduit::Node n_ok;
  size_t failed = 0;
  std::unordered_set<std::string> valid_samples;
  if (m_master) std::cerr << "jag_store::setup; scanning data files for success flag\n";

  if (bcast && !m_master) {
    goto BCAST;
  }
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
        m_data_id_to_filename_idx.push_back(j);
        m_data_id_to_string_id.push_back(t);
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
  if (m_master) {
    std::cerr << "jag_store::setup; successful samples: " << m_num_samples << " failed samples: " << failed << " time to test for success: " << get_time() - tm1 << std::endl;
  }

BCAST:
  if (bcast) {
    //todo
  }

  //allocate memory
  size_t nthreads = omp_get_max_threads();
  m_data_inputs.resize(nthreads);
  m_data_scalars.resize(nthreads);
  m_data_images.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    m_data_inputs[j].reserve(m_inputs_to_use.size());
    m_data_scalars[j].reserve(m_scalars_to_use.size());
    m_data_images[j].resize(m_image_views_to_use.size());
  }
  for (size_t j=0; j<m_data_images.size(); j++) {
    m_data_images[j].resize(m_image_views_to_use.size());
    for (size_t i=0; i<m_image_views_to_use.size(); i++) {
      m_data_images[j][i].resize(m_image_size);
    }
  }

  build_data_sizes(); //todo -- needs fixing

  if (m_master) {
    std::cout << "\n======================================================\n"
              << "data sizes: ";
    const std::vector<size_t> & s = get_linearized_data_sizes();
    size_t total = 0;
    for (auto t : s) {
      total += t;
      std::cout << t << " ";
    }
    std::cout << "\n======================================================\n\n";
  }

  m_is_setup = true;
  if (m_master) {
    std::cerr << "jag_store::setup time: " << get_time() - tm1 << std::endl;
  }
}

  
size_t jag_store::get_linearized_data_size() const {
  return m_inputs_to_use.size()
           + m_scalars_to_use.size()
           + m_image_views_to_use.size() * get_linearized_image_size();
}

void jag_store::build_data_sizes() {
  for (auto t : m_inputs_to_use) {
    m_data_sizes.push_back(get_linearized_input_size());
  }
  for (auto t : m_scalars_to_use) {
    m_data_sizes.push_back(get_linearized_scalar_size());
  }
  for (auto t : m_image_views_to_use) {
    m_data_sizes.push_back(get_linearized_image_size());
  }
}

void jag_store::load_data(int data_id, int tid) {
  check_sample_id(data_id);
  int idx = m_data_id_to_filename_idx[data_id];
  //if (m_master) std::cerr << "idx: " << idx << "\n";
  std::string &filename = m_conduit_filenames[idx];
  std::string &sample_name = m_data_id_to_string_id[idx];

  //if (m_master) std::cerr << "data_id: " << data_id << " m_conduit_filenames.size: " << m_conduit_filenames.size() << " filename: " << filename << " sample_name: " << sample_name << " idx: " << idx << "\n";

  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filename );
  conduit::Node node;
  for (auto input_name : m_inputs_to_use) {
    const std::string key = "/" + sample_name + "/inputs/" + input_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if scalar_t changes
    m_data_inputs[tid].push_back( node.to_float64() );
  }
  for (auto scalar_name : m_scalars_to_use) {
    const std::string key = "/" + sample_name + "/outputs/scalars/" + scalar_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if input_t changes
    m_data_inputs[tid].push_back( node.to_float64() );
  }
  size_t k = 0;
  for (auto image_name : m_image_views_to_use) {
    const std::string key = "/" + sample_name + "/outputs/images/" + image_name + "/0.0/emi";
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    conduit::float32_array emi = node.value();
    const size_t image_size = emi.number_of_elements();
    if (image_size != m_image_size*4) {  //MAGIC NUMBER!
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: m_image_size = " + std::to_string(m_image_size) + " but conduit image size is " + std::to_string(image_size));
    }
    //this is fragile; will break if ch_t changes
    // (also, can probably eliminate this loop ...)
    for (size_t h=0; h<image_size; h++) {
      m_data_images[tid][k][h] = emi[h];
    }
    ++k;
  }
  conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
}

} // namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT
