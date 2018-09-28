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
    m_master(false),
    m_use_conduit(false)
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
  m_scalars_to_use.clear();
  load_keys(m_scalars_to_use, keys);
}

void jag_store::load_inputs(const std::string &keys) {
  m_inputs_to_use.clear();
  load_keys(m_inputs_to_use, keys);
}

void jag_store::load_image_views(const std::string &keys) {
  m_image_views_to_use.clear();
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
  const std::vector<std::string> filenames,
  data_reader_jag_conduit_hdf5 *reader,
  bool num_stores,
  int my_rank) {

  //magic numbers (from Rushil)
  m_normalize.push_back(0.035550589898738466);
  m_normalize.push_back(0.0012234476453273034);
  m_normalize.push_back(1.0744965260584181e-05);
  m_normalize.push_back(2.29319120949361e-07);

  m_master = m_comm->am_world_master();
  if (m_master) std::cerr << "starting jag_store::setup for " << filenames.size() << " conduit files\n";
  double tm1 = get_time();
  m_reader = reader;
  m_conduit_filenames = filenames;

  options *opts = options::get();
  if (opts->has_string("convert_conduit")) {
    convert_conduit(filenames);
    return;
  }

  load_variable_names();
  build_data_sizes(); 
  report_linearized_sizes();
  allocate_memory(); 

  if (opts->has_bool("use_conduit")) {
    m_use_conduit = true;
    if (m_master) {
      std::cerr << "USING CONDUIT DIRECTLY\n";
      setup_conduit(tm1);
      return;
    }
  }

  size_t nthreads = omp_get_max_threads();
  m_stream.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    m_stream[j].resize(filenames.size());
  }

  read_key_map(reader->get_data_filename());

  m_num_samples = 0;
  std::string line;
  int global_idx = 0;
  int file_idx = -1;
  for (auto t : filenames) {
    ++file_idx;
    size_t j = t.rfind(".bin");
    if (j == std::string::npos) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: t.rfind('.bin') failed");
    }
    std::stringstream s;
    s << t.substr(0, j) << "_names.txt";
    std::cerr << "opening: " << s.str() << "\n";
    std::ifstream in(s.str().c_str());
    if (!in.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + s.str() + " for reading");
    }
    int idx = 0;
    while (!in.eof()) {
      getline(in, line);
      if (line == "") {
        break;
      }
      ++m_num_samples;
      m_sample_map[global_idx++] = std::make_pair(file_idx, idx++);
    }
    for (size_t i=0; i<nthreads; i++) {
      m_stream[i][file_idx] = new std::ifstream(t, std::ios::in | std::ios::binary);
      if (! m_stream[i][file_idx]->good()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + t + " for reading");
      }
    }
  }

  m_scratch.resize(nthreads);
  for (size_t i=0; i<m_scratch.size(); i++) {
    m_scratch[i].resize(m_sample_len);
  }

  m_is_setup = true;
  if (m_master) {
    std::cerr << "jag_store::setup time: " << get_time() - tm1 << std::endl;
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

void jag_store::setup_conduit(double tm1) {
  m_is_setup = true;
  bool master = true;

  std::string test_file;
  std::string test_sample_id;

  // get the sample_ids of successful samples; mostly need to do this
  // to figure out the proper memory allocation for m_data
  conduit::Node n_ok;
  size_t failed = 0;
  for (size_t j = 0; j<m_conduit_filenames.size(); ++j) {
  std::cerr << "ATTEMPTING to open: " << m_conduit_filenames[j] << "\n";
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( m_conduit_filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (auto t : cnames) {
      const std::string key = "/" + t + "/performance/success";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
      int success = n_ok.to_int64();
      if (success == 1) {
        m_valid_samples.insert(t);
        m_data_id_to_string_id.push_back(t);
        m_data_id_to_filename_idx.push_back(j);
        if (test_file == "") {
          test_file = m_conduit_filenames[j];
          test_sample_id = "/" + t;
        }
      } else {
        ++failed;
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
  m_num_samples = m_valid_samples.size();
  if (master) {
    std::cout << "jag_store::setup; successful samples: " << m_num_samples << " failed samples: " << failed << " time to test for success: " << get_time() - tm1 << std::endl;
  }
  //double tm2 = get_time();

  #if 0
  get_default_keys(test_file, test_sample_id, "inputs", master);
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

  m_data_inputs.resize(m_num_samples);
  m_data_scalars.resize(m_num_samples);
  m_data_images.resize(m_num_samples);
  for (size_t j=0; j<m_num_samples; j++) {
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

  //load the data
  size_t idx = 0;
  conduit::Node node;
  for (size_t j = 0; j<m_conduit_filenames.size(); ++j) {
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( m_conduit_filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (auto t : cnames) {
      m_id_to_name[idx] = t;
      if (m_valid_samples.find(t) != m_valid_samples.end()) {
std::cerr << "m_inputs_to_load.size(): " << m_inputs_to_load.size() << "\n";
        for (auto input_name : m_inputs_to_load) {
          const std::string key = "/" + t + "/inputs/" + input_name;
std::cerr << "key: " << key << "\n";
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if scalar_t changes
std::cerr << "m_data_inputs.size(): " << m_data_inputs.size() << "\n";
          m_data_inputs[idx].push_back( node.to_float64() );
        }
        for (auto scalar_name : m_scalars_to_load) {
          const std::string key = "/" + t + "/outputs/scalars/" + scalar_name;
std::cerr << "scalar key: " << key << "\n";
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
            m_data_images_2[idx][k][h] = emi[h];
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
  for (size_t j=0; j<m_image_views_to_use.size(); j++) {
    for (size_t k=0; k<m_image_channels_to_use.size(); k++) {
      m_data_sizes.push_back(IMAGE_SIZE_PER_CHANNEL);
    }
  }
  m_data_sizes.push_back(get_linearized_scalar_size());
  m_data_sizes.push_back(get_linearized_input_size());
}

void jag_store::load_data_from_conduit(int data_id, int tid) {
  conduit::Node node;
  const int filename_idx = m_data_id_to_filename_idx[data_id];
  const std::string &sample_id = m_data_id_to_string_id[data_id];
  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( m_conduit_filenames[filename_idx] );

  for (auto input_name : m_inputs_to_load) {
    const std::string key = "/" + sample_id + "/inputs/" + input_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if scalar_t changes
    m_data_inputs[tid].push_back( node.to_float64() );
  }

  for (auto scalar_name : m_scalars_to_load) {
    const std::string key = "/" + sample_id + "/outputs/scalars/" + scalar_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if input_t changes
    m_data_inputs[tid].push_back( node.to_float64() );
  }

  size_t image_num = 0;
  for (auto image_name : m_images_to_load) {
    const std::string key = "/" + sample_id + "/outputs/images/" + image_name + "/0.0/emi";
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
    for (size_t channel = 0; channel<4; channel++) {
       for (size_t h=channel; h<image_size; h+=4) {
        m_data_images[tid][image_num][channel][h] = emi[h];
      }
    }
    ++image_num;
  }
  conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
}

void jag_store::load_data(int data_id, int tid) {
  if (m_use_conduit) {
    load_data_from_conduit(data_id, tid);
    return;
  }
static int debug = 0;
  check_sample_id(data_id);
  int file_idx = m_sample_map[data_id].first;
  size_t n = m_sample_map[data_id].second;
  size_t offset = n * m_sample_len;
if (debug < 10) std::cerr << "file_idx: " << file_idx << " n: " << n << " offset: " << offset << "\n";
    m_stream[tid][file_idx]->seekg(offset);
    m_stream[tid][file_idx]->read((char*)m_scratch[tid].data(), m_sample_len);

  for (size_t j=0; j<m_inputs_to_use.size(); j++) {
    check_entry(m_inputs_to_use[j]);
std::cerr << "XX: m_inputs_to_use.size: "<<m_inputs_to_use.size() <<"  m_data_inputs.size(): " << m_data_inputs.size() << "\n";;
std::cerr << "XX: m_inputs_to_use.size: "<<m_inputs_to_use.size() <<"  m_data_inputs[0].size(): " << m_data_inputs[0].size() << "\n";;
std::cerr << "XX: m_scratch.size(): " << m_scratch.size() << "\n";
std::cerr << "XX: m_scratch[0].size(): " << m_scratch[0].size() << "\n";
    memcpy((void*)(m_data_inputs[tid].data()+j), (void*)(m_scratch[tid].data()+m_key_map[m_inputs_to_use[j]]), 8);
  }

if (debug< 10) {
  for (auto t : m_data_inputs[tid]) std::cerr << t << " ";
  std::cerr << "\n";
}

std::cerr << "XX m_scalars_to_use.size: " << m_scalars_to_use.size() << "  m_data_scalars.size(): " << m_data_scalars.size() << "\n";
std::cerr << "XX m_data_scalars[0].size(): " << m_data_scalars[0].size() << "\n";
  for (size_t j=0; j<m_scalars_to_use.size(); j++) {
    check_entry(m_scalars_to_use[j]);
    memcpy((void*)(m_data_scalars[tid].data()+j), (void*)(m_scratch[tid].data()+m_key_map[m_scalars_to_use[j]]), 8);
  }

if (debug < 10) {
  for (auto t : m_data_scalars[tid]) std::cerr << t << " ";
  std::cerr << "\n";
}

  for (size_t view=0; view<m_image_views_to_use.size(); view++) {
    check_entry(m_image_views_to_use[view]);
    for (size_t k=0; k<m_image_channels_to_use.size(); k++) {
      int channel = m_image_channels_to_use[k];
      if (m_key_map[m_image_views_to_use[view]] + channel*64*64*4 + 64*64*4 > m_scratch[tid].size()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: attempted out of bounds read in m_scratch buffer!\n");
      }
      memcpy((void*)m_data_images[tid][view][channel].data(), 
             (void*)(m_scratch[tid].data()+m_key_map[m_image_views_to_use[view]] + channel*64*64*4), 64*64*4);
    }
  }
++debug;
}

void jag_store::open_output_files(const std::string &dir) {
  if (m_name_file.is_open()) {
    m_name_file.close();
  }
  if (m_binary_file.is_open()) {
    m_binary_file.close();
  }
  m_cur_bin_count = 0;
  std::stringstream s;
  s << dir << "/" << BINARY_FILE_BASENAME << "_" << m_bin_file_count << "_names.txt";
  m_name_file.open(s.str().c_str());
  if (!m_name_file.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open: " + s.str());
  }  
  s.clear();
  s.str("");
  s << dir << "/" << BINARY_FILE_BASENAME << "_" << m_bin_file_count << ".bin";
  ++m_bin_file_count;
  m_binary_file.open(s.str().c_str(), std::ios::out | std::ios::binary);
  if (!m_binary_file.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open: " + s.str());
  }  
}

void jag_store::write_binary(const std::string &input, const std::string &dir) {
  if (m_cur_bin_count >= MAX_SAMPLES_PER_BINARY_FILE) {
    open_output_files(dir);
  }

  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( input );
  std::vector<std::string> cnames;
  conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
  conduit::Node n_ok;
  conduit::Node node;
  for (auto sample_name : cnames) {
    const std::string key_1 = "/" + sample_name + "/performance/success";
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key_1, n_ok);
    int success = n_ok.to_int64();
    if (success == 1) {
      m_name_file << sample_name << "\n";

      for (auto input_name : m_inputs_to_use) {
        const std::string key = "/" + sample_name + "/inputs/" + input_name;
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
        //this is fragile; will break if input_t changes
        double tmp = node.to_float64();
        m_binary_file.write((char*)&tmp, sizeof(data_reader_jag_conduit_hdf5::input_t));
      }

      for (auto scalar_name : m_scalars_to_use) {
        const std::string key = "/" + sample_name + "/outputs/scalars/" + scalar_name;
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
        //this is fragile; will break if scalar_t changes
        double tmp = node.to_float64();
        m_binary_file.write((char*)&tmp, sizeof(data_reader_jag_conduit_hdf5::scalar_t));
      }

      for (auto image_name : m_image_views_to_use) {
        const std::string key = "/" + sample_name + "/outputs/images/" + image_name + "/0.0/emi";
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
        conduit::float32_array emi = node.value();
        const size_t image_size = emi.number_of_elements();
        //this is fragile; will break if ch_t changes
        for (int channel=0; channel<4; channel++) {
          for (size_t j=channel; j<image_size; j+=4) {
            m_binary_file.write((char*)&emi[j], sizeof(data_reader_jag_conduit_hdf5::ch_t));
            //m_binary_file.write((char*)&emi[0], image_size*sizeof(data_reader_jag_conduit_hdf5::ch_t));
          }
        }
      }

      ++m_cur_bin_count;
      if (m_cur_bin_count >= MAX_SAMPLES_PER_BINARY_FILE) {
        open_output_files(dir);
      }
    }
  }
}

void jag_store::test_conversion(const std::string &input_fn, const std::string &output_dir) {
  std::cerr << "\nstarting jag_store::test_conversion\n";

  std::stringstream s;
  s << output_dir << "/" << METADATA_FN;
  read_key_map(s.str());

  s.clear();
  s.str("");
  s << output_dir << "/" << BINARY_FILE_BASENAME << "_0_names.txt";
  std::ifstream in_names(s.str().c_str());
  if (!in_names.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open name file: " + s.str());
  }
  std::cerr << "OPENED: " << s.str() << "\n";

  s.clear();
  s.str("");
  s << output_dir << "/" << BINARY_FILE_BASENAME << "_0.bin";
  std::ifstream in(s.str().c_str(), std::ios::out | std::ios::binary);
  if (!in.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open: " + s.str());
  }

  std::cerr << "loading conduit file\n";
  conduit::Node data;
  conduit::relay::io::load(input_fn, "hdf5", data);
  s.clear();

  std::cerr << "testing inputs\n";
  std::string sample_id;
  int k = 0;
  while (! in_names.eof()) {
    getline(in_names, sample_id);
    if (sample_id == "") {
      break;
    }
    std::cout << "============================================================\n\n";
    for (auto input_key : m_inputs_to_use) {
      double v1;
      if (m_key_map.find(input_key) == m_key_map.end()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to find key: '" + input_key + "' in m_key_map");
      }
      int offset = m_key_map[input_key] + (m_sample_len * k);
      in.seekg(offset);
      in.read((char*)&v1, sizeof(double));
      std::string key = "/" + sample_id + "/inputs/" + input_key;
      std::cout << "getting node from conduit bundle: " << key << "\n";
      const conduit::Node nd = data[key];
      double v2 = nd.as_float64();
      if (v1 != v2) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: input key: " + input_key + " value from binary file: " + std::to_string(v1) + " doesn't match values from conduit file: " + std::to_string(v2));
      }
    }
    for (auto scalar_key : m_scalars_to_use) {
      double v1;
      if (m_key_map.find(scalar_key) == m_key_map.end()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to find key: '" + scalar_key + "' in m_key_map");
      }
      int offset = m_key_map[scalar_key] + (m_sample_len * k);
      in.seekg(offset);
      in.read((char*)&v1, sizeof(double));
      std::string key = "/" + sample_id + "/outputs/scalars/" + scalar_key;
      std::cout << "getting node from conduit bundle: " << key << "\n";
      const conduit::Node nd = data[key];
      double v2 = nd.as_float64();
      if (v1 != v2) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: scalar key: " + scalar_key + " value from binary file: " + std::to_string(v1) + " doesn't match values from conduit file: " + std::to_string(v2));
      }
    }
    for (auto image_key : m_image_views_to_use) {
      if (m_key_map.find(image_key) == m_key_map.end()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to find key: '" + image_key + "' in m_key_map");
      }
      std::string key = "/" + sample_id + "/outputs/images/" + image_key + "/0.0/emi";
      std::cout << "getting node from conduit bundle: " << key << "\n";
      const conduit::Node nd = data[key];
      conduit::float32_array emi = nd.value();
      const size_t image_size = emi.number_of_elements();
      std::vector<float> v(image_size);
      int offset = m_key_map[image_key] + (m_sample_len * k);
      in.seekg(offset);
      in.read((char*)v.data(), sizeof(float)*image_size);

      #if 0 
      //note: don't do this -- because I de-interlaced the channels on disk
      for (size_t i=0; i<image_size; i++) {
        if (emi[i] != v[i]) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: image key: " + image_key + " values do not match for index " + std::to_string(i) + " of " + std::to_string(image_size));
        }
      }
      #endif
    }

    ++k;
    if (k == 10) {
      break;
    }
  }
}

void jag_store::read_key_map(const std::string &filename) {
  std::ifstream in(filename.c_str());
  if (!in.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open: " + filename);
  }

  std::string line;
  getline(in, line);
  getline(in, line);
  getline(in, line);

  std::string key;
  int n;
  m_inputs_to_use.clear();
  m_scalars_to_use.clear();
  m_image_views_to_use.clear();
  for (int k=0; k<3; k++) {
    getline(in, line);
    std::stringstream s;
    s << line;
    s >> key >> n;
    std::cerr << key << " " << n << "\n";
    for (int j=0; j<n; j++) {
      getline(in, line);
      size_t j2 = line.rfind(" ");
      if (j2 == std::string::npos) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to rfind space for this line: " + line);
      }
      int k2 = atoi(&line[j2+1]);
      m_key_map[line.substr(0, j2)] = k2;
      switch (k) {
        case 0 : m_inputs_to_use.push_back(line.substr(0, j2));
                 break;
        case 1 : m_scalars_to_use.push_back(line.substr(0, j2));
                 break;
        case 2 : m_image_views_to_use.push_back(line.substr(0, j2));
                 break;
      }
    }
  }
  getline(in, line);
  if (line.find("TOTAL") == std::string::npos) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: missing TOTAL field");
  }
  size_t j3 = line.rfind(" ");
  if (j3 == std::string::npos) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to rfind space for this line: " + line);
  }
  m_sample_len = atoi(&line[j3+1]);
  in.close();

  for (auto t : m_key_map) {
    std::cerr << "key: " << t.first << " offset: " << t.second << "\n";
  }
}

void jag_store::write_binary_metadata(std::string dir) {
  std::stringstream s;
  s << dir << "/" << METADATA_FN;
  std::ofstream out(s.str().c_str());
  out << "input_t " << sizeof(data_reader_jag_conduit_hdf5::input_t) << "\n";
  out << "scalar_t " << sizeof(data_reader_jag_conduit_hdf5::scalar_t) << "\n";
  out << "ch_t " << sizeof(data_reader_jag_conduit_hdf5::ch_t) << "\n";

  out << "INPUTS " << m_inputs_to_use.size() << "\n";
  size_t offset = 0;
  for (auto t : m_inputs_to_use) {
    out << t << " " << offset << "\n";
    offset += sizeof(data_reader_jag_conduit_hdf5::input_t);
  }

  out << "SCALARS " << m_scalars_to_use.size() << "\n";
  for (auto t : m_scalars_to_use) {
    out << t << " " << offset << "\n";
    offset += sizeof(data_reader_jag_conduit_hdf5::scalar_t);
  }

  out << "VIEWS " << m_image_views_to_use.size() << "\n";
  for (auto t : m_image_views_to_use) {
    out << t << " " << offset << "\n";
    offset += sizeof(data_reader_jag_conduit_hdf5::ch_t)*128*128; //magic number!
  }
  out << "TOTAL " << offset << "\n";
  out.close();
}

void jag_store::convert_conduit(const std::vector<std::string> &conduit_filenames) {
  load_inputs(m_reader->m_input_keys);
  load_scalars(m_reader->m_scalar_keys);
  load_image_views(m_reader->m_image_views);
  load_image_channels(m_reader->m_image_channels);
  m_image_size = m_image_channels_to_use.size() * IMAGE_SIZE_PER_CHANNEL;

  options *opts = options::get();
  if (!opts->has_string("output_dir")) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: missing --output_dir=<string> which is needed for conversion");
  }  
  std::string output_dir = opts->get_string("output_dir");
  if (m_master) {
    std::cerr << "\nConverting conduit files; num files: " << conduit_filenames.size() << "\n";
    char b[128];
    sprintf(b, "mkdir -p %s", output_dir.c_str());
    system(b);
    write_binary_metadata(output_dir);
  }
  m_comm->global_barrier();

  m_cur_bin_count = MAX_SAMPLES_PER_BINARY_FILE;
  m_bin_file_count = 0;

  int np = m_comm->get_procs_in_world();
  int me = m_comm->get_rank_in_world();
  for (size_t j=me; j<conduit_filenames.size(); j += np) {
    size_t n1 = conduit_filenames[j].rfind("/");
    if (n1 == std::string::npos) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: rfind failed.\n");
    }
    write_binary(conduit_filenames[j], output_dir);
  }
  m_comm->global_barrier();
  if (m_name_file.is_open()) {
  std::cerr << "CLOSING:  m_name_file\n";
    m_name_file.close();
  }
  if (m_binary_file.is_open()) {
  std::cerr << "CLOSING:  m_binary_file\n";
    m_binary_file.close();
  }
  if (opts->has_bool("test_conversion")) {
    if (m_master) {
      test_conversion(conduit_filenames[0], output_dir);
    }
  }
  m_comm->global_barrier();
}

void jag_store::load_variable_names() {
  //bad naming: these methods don't load anything -- they just setup
  //the names of the keys for the data that will be loaded later
  load_inputs(m_reader->m_input_keys);
  load_scalars(m_reader->m_scalar_keys);
  load_image_views(m_reader->m_image_views);
  load_image_channels(m_reader->m_image_channels);
  m_image_size = m_image_channels_to_use.size() * IMAGE_SIZE_PER_CHANNEL;

  if (m_master) {
    std::cerr << "\n======================================================\n"
              << "using these inputs: ";
    for (auto t : m_inputs_to_use) {
      std::cerr << t << " ";
    }
    std::cerr << "\n\nusing these scalars: ";
    for (auto t : m_scalars_to_use) {
      std::cerr << t << " ";
    }
    std::cerr << "\n\nusing these views: ";
    for (auto t : m_image_views_to_use) {
      std::cerr << t << " ";
    }
    std::cerr << "\n\nusing these image channels: ";
    for (auto t : m_image_channels_to_use) {
      std::cerr << t << " ";
    }
    std::cerr << "\n======================================================\n";
  }
}

void jag_store::allocate_memory() {
std::cerr << "XX jag_store::allocate_memory; m_inputs_to_use.size: " << m_inputs_to_use.size() << "\n";
  size_t nthreads = omp_get_max_threads();
  m_data_inputs.resize(nthreads);
  m_data_scalars.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    m_data_inputs[j].resize(m_inputs_to_use.size());
    m_data_scalars[j].resize(m_scalars_to_use.size());
  }

  m_data_images.resize(nthreads);  
  for (size_t j=0; j<m_data_images.size(); j++) {
    m_data_images[j].resize(m_image_views_to_use.size());
    for (size_t i=0; i<m_data_images[j].size(); i++) {
      m_data_images[j][i].resize(m_image_channels_to_use.size());
      for (size_t h=0; h<m_data_images[j][i].size(); h++) {
        m_data_images[j][i][h].resize(IMAGE_SIZE_PER_CHANNEL);
      }
    }
  }
}

void jag_store::report_linearized_sizes() {
  if (m_master) {
    std::cerr << "\n======================================================\n"
              << "data sizes: ";
    const std::vector<size_t> & s = get_linearized_data_sizes();
    size_t total = 0;
    for (auto t : s) {
      total += t;
      std::cerr << t << " ";
    }
    std::cerr << "\nget_linearized_data_size: " << get_linearized_data_size() << "\n"
              << "get_linearized_image_size: " << get_linearized_image_size() << "\n"
              << "get_linearized_scalar_size_size: " << get_linearized_scalar_size() << "\n"
              << "get_linearized_input_size_size: " << get_linearized_input_size() << "\n"
              << "get_num_img_srcs: " << get_num_img_srcs() << "\n";
    std::cerr << "\n======================================================\n\n";
  }
}
} // namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT
