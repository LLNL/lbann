#include "lbann/data_store/jag_store.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/options.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include "lbann/utils/glob.hpp"
#include "hdf5.h"
#include <unordered_set>

namespace lbann {

jag_store::jag_store() 
  : m_is_setup(false),
    m_image_size(0),
    m_run_tests(false),
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
  const std::vector<std::string> conduit_filenames,
  data_reader_jag_conduit_hdf5 *reader,
  bool num_stores,
  int my_rank) {
  double tm1 = get_time();

<<<<<<< HEAD
  // quick hack to get every processor to read a unique
  // subset of the data
  if (options::get()->has_string("every_n") || options::get()->has_string("jag_partitioned")) {
    my_rank = m_comm->get_rank_in_world();
    num_stores = m_comm->get_procs_in_world();
  }
=======
  //magic numbers (from Rushil)
  m_normalize.push_back(0.035550589898738466);
  m_normalize.push_back(0.0012234476453273034);
  m_normalize.push_back(1.0744965260584181e-05);
  m_normalize.push_back(2.29319120949361e-07);
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53

  m_reader = reader;
  m_master = m_comm->am_world_master();
  m_conduit_filenames = conduit_filenames;

  options *opts = options::get();

  size_t max_samples = INT_MAX;
  if (opts->has_int("max_samples")) {
    max_samples = opts->get_int("max_samples");
  }

<<<<<<< HEAD
=======
  load_variable_names();
  report_linearized_sizes();
  build_data_sizes();
  allocate_memory();

>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
  if (m_master) std::cerr << "starting jag_store::setup for " << conduit_filenames.size() << " conduit files\n";

  if (m_image_size == 0) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: image_size = 0; probably set_image_size() has not been called");
  }

  size_t nthreads = omp_get_max_threads();
  m_stream.resize(nthreads);

<<<<<<< HEAD
  // get the sample_ids of successful samples; mostly need to do this
  // to figure out the proper memory allocation for m_data
  double tm1 = get_time();
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
        m_valid_samples.insert(t);
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
  m_num_samples = m_valid_samples.size();
  if (m_master) {
    std::cout << "jag_store::setup; successful samples: " << m_num_samples << " failed samples: " << failed << " time to test for success: " << get_time() - tm1 << std::endl;
=======
  read_key_map(reader->get_data_filename());

  if (m_master) std::cerr << "calling:  glob(pattern)\n";
  const std::string pattern("/p/lscratchh/brainusr/datasets/1MJAG_converted/*.bin");
  std::vector<std::string> names = glob(pattern);

  for (size_t j=0; j<nthreads; j++) {
    m_stream[j].resize(names.size());
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
  }

  std::string line;
  int global_idx = 0;
  int file_idx = -1;
  m_num_samples = 0;

  for (auto t : names) {
    ++file_idx;
    if (m_num_samples == max_samples) {
      break;
    }
    size_t j = t.rfind(".bin");
    if (j == std::string::npos) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: t.rfind('.bin') failed");
    }
<<<<<<< HEAD
    if (m_load_inputs) {
      get_default_keys(test_file, test_sample_id, "inputs");
    }
    if (m_load_scalars) {
      get_default_keys(test_file, test_sample_id, "scalars");
=======
    std::stringstream s;
    s << t.substr(0, j) << "_names.txt";
    std::ifstream in(s.str().c_str());
    if (!in.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + s.str() + " for reading");
    }
    size_t idx = 0;
    while (!in.eof()) {
      getline(in, line);
      if (line == "") {
        break;
      }

      m_sample_map[global_idx] = std::make_pair(file_idx, idx++);
      m_sample_id_map[global_idx] = line;
      ++global_idx;

      ++m_num_samples;
      if (m_num_samples == max_samples) {
        break;
      }
    }
    in.close();

    for (size_t i=0; i<nthreads; i++) {
      m_stream[i][file_idx] = new std::ifstream(t, std::ios::in | std::ios::binary);
      if (! m_stream[i][file_idx]->good()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + t + " for reading");
      }
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
    }
  }

  m_scratch.resize(nthreads);
  for (size_t i=0; i<m_scratch.size(); i++) {
    m_scratch[i].resize(m_sample_len);
  }

  m_is_setup = true;
  if (m_master) {
    std::cerr << "jag_store::setup time: " << get_time() - tm1 << "; num samples: " << m_num_samples << std::endl;
  }
}

size_t jag_store::get_linearized_data_size() const {
  size_t n = m_image_views_to_use.size() * m_image_channels_to_use.size() * get_linearized_channel_size()
           + m_scalars_to_use.size()
           + m_inputs_to_use.size();
  return n;
}

void jag_store::build_data_sizes() {
  size_t total_channels = m_image_channels_to_use.size() * m_image_views_to_use.size();
  for (size_t i=0; i<total_channels; i++) {
    m_data_sizes.push_back(get_linearized_channel_size());
  }
  for (auto t : m_scalars_to_use) {
    m_data_sizes.push_back(get_linearized_scalar_size());
  }
  for (auto t : m_inputs_to_use) {
    m_data_sizes.push_back(get_linearized_input_size());
  }
}

void jag_store::report_linearized_sizes() {
  if (m_master) {
    std::cerr << "\ndata sizes: ";
    const std::vector<size_t> & s = get_linearized_data_sizes();
    size_t total = 0;
    for (auto t : s) {
      total += t;
      std::cerr << t << " ";
    }
    std::cerr << "\nget_linearized_data_size:  " << get_linearized_data_size() << "\n"
              << "get_linearized_image_size:   " << get_linearized_image_size() << "\n"
              << "get_linearized_channel_size: " << get_linearized_channel_size() << "\n"
              << "get_num_channels: " << get_num_channels_per_view() << "\n"
              << "get_linearized_scalar_size:  " << get_linearized_scalar_size() << "\n"
              << "get_linearized_input_size:   " << get_linearized_input_size() << "\n"
              << "get_num_img_srcs:            " << get_num_img_srcs() << "\n";
  }
}

void jag_store::load_data(int data_id, int tid) {
  check_sample_id(data_id);
  int file_idx = m_sample_map[data_id].first;
  size_t n = m_sample_map[data_id].second;
  size_t offset = n * m_sample_len;

  m_stream[tid][file_idx]->seekg(offset);
  m_stream[tid][file_idx]->read((char*)m_scratch[tid].data(), m_sample_len);

  for (size_t j=0; j<m_inputs_to_use.size(); j++) {
    check_entry(m_inputs_to_use[j]);
    memcpy((void*)(m_data_inputs[tid].data()+j), (void*)(m_scratch[tid].data()+m_key_map[m_inputs_to_use[j]]), 8);
  }
  /* todo: normalize
  for (size_t j=0; j<m_data_inputs[tid].size(); j++) {
    // normalize m_data_inputs[tid][j]
  }
  */

  for (size_t j=0; j<m_scalars_to_use.size(); j++) {
    check_entry(m_scalars_to_use[j]);
    memcpy((void*)(m_data_scalars[tid].data()+j), (void*)(m_scratch[tid].data()+m_key_map[m_scalars_to_use[j]]), 8);
  }
  /* todo: normalize
  for (size_t j=0; j<m_data_scalars[tid].size(); j++) {
    // normalize m_data_scalars[tid][j]
  }
  */

  size_t y = 0;
  for (size_t view=0; view<m_image_views_to_use.size(); view++) {
    check_entry(m_image_views_to_use[view]);
    for (size_t k=0; k<m_image_channels_to_use.size(); k++) {
      int channel = m_image_channels_to_use[k];
      memcpy((void*)m_data_images[tid][y].data(), 
             (void*)(m_scratch[tid].data()+m_key_map[m_image_views_to_use[view]] + channel*get_linearized_channel_size()), get_linearized_channel_size());
      /*
      for (size_t h=0; h<m_data_images[tid][y].size(); h++) {
        m_data_images[tid][y][h] /= m_normalize[channel];
      }
      */
      ++y;
    }
  }
}

<<<<<<< HEAD
  build_data_sizes();
  m_is_setup = true;
  if (m_master) {
    std::cerr << "jag_store::setup; time to load the data: " << get_time() - tm2 << std::endl;
=======
void jag_store::open_output_files(const std::string &dir) {
  if (m_name_file.is_open()) {
    m_name_file.close();
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
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

<<<<<<< HEAD
void jag_store::get_default_keys(std::string &filename, std::string &sample_id, std::string key1) {
  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(filename);
  conduit::Node n2;
=======
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
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53

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
  for (int k=0; k<3; k++) {
    getline(in, line);
    std::stringstream s;
    s << line;
    s >> key >> n;
    for (int j=0; j<n; j++) {
      getline(in, line);
      size_t j2 = line.rfind(" ");
      if (j2 == std::string::npos) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to rfind space for this line: " + line);
      }
      int k2 = atoi(&line[j2+1]);
      m_key_map[line.substr(0, j2)] = k2;
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

<<<<<<< HEAD
void jag_store::run_tests(const std::vector<std::string> &conduit_filenames) {
  conduit::Node node;
  if (m_master) {
    std::cout << "\n=======================================================================\n"
              << "starting jag_store::run_tests\n";
  }
  for (auto t : conduit_filenames) {
    if (m_master) {
      std::cout << "  loading conduit::Node from file: " << t << "\n";
=======
  if (m_master) {
    for (auto t : m_key_map) {
      std::cerr << "key: " << t.first << " offset: " << t.second << "\n";
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
    }
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

<<<<<<< HEAD
    const std::vector<data_reader_jag_conduit_hdf5::input_t> & inputs = fetch_inputs(j);
    size_t i = 0;
    for (auto input_name : m_inputs_to_load) {
      const std::string key = "/" + sample_id + "/inputs/" + input_name;
      if (m_master) {
        //std::cout << "  next key: " << key << "\n";
      }
      const conduit::Node& nd = node[key];
      if (inputs[i] != nd.to_float64()) {
        std::cout << "FAILED; id: " << j << " sample name: " << sample_id << " key: " << key << " value from jag_store: " << inputs[j] << " value from conduit: " << nd.to_float64() << "\n";
      }
      ++i;
    }
  }
  if (m_master) {
    std::cout << "all inputs match!\n";
  }

  std::cout << "jag_store::run_tests; testing images\n";
  for (size_t j=0; j<get_num_samples(); j++) {
    const std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>> & images = fetch_images(j);
    std::string sample_id = m_id_to_name[j];
    //if (m_master) std::cout << "  next sample: " << sample_id << "\n";
    for (size_t k=0; k<m_images_to_load.size(); k++) {
      //if (m_master) std::cerr << "  next image: " << m_images_to_load[k] << "\n";
      std::string img_key = sample_id + "/outputs/images/" + m_images_to_load[k] + "/0.0/emi";
      //if (m_master) std::cerr << "  next key: " << img_key << "\n";
      const conduit::Node & n_image = node[img_key];
      conduit::float32_array emi = n_image.value();
      const size_t num_pixels = emi.number_of_elements();
      if (num_pixels != images[k].size()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: jag_store::run_tests() - image size mismatch");
      }
      const data_reader_jag_conduit_hdf5::ch_t* emi_data = n_image.value();
      for (size_t i=0; i<num_pixels; i++) {
        if (images[k][i] != emi_data[i]) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: jag_store::run_tests() - images don't match for sample " + std::to_string(j) + " image " + std::to_string(k) + " byte number " + std::to_string(i));
        }
      }
=======
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
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
    }
    write_binary(conduit_filenames[j], output_dir);
  }
<<<<<<< HEAD
  if (m_master) {
    std::cout << "all images match!\n";
    std::cout << "=======================================================================\n";
=======
  m_comm->global_barrier();
  if (m_name_file.is_open()) {
  std::cerr << "CLOSING:  m_name_file\n";
    m_name_file.close();
>>>>>>> ef9f22783716ee1bcd4c85b826824b78a786ad53
  }
  if (m_binary_file.is_open()) {
  std::cerr << "CLOSING:  m_binary_file\n";
    m_binary_file.close();
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

  if (m_master) {
    std::cerr << "using these inputs: ";
    for (auto t : m_inputs_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\n\nusing these scalars: ";
    for (auto t : m_scalars_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\n\nusing these views: ";
    for (auto t : m_image_views_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\n\nusing these image channels: ";
    for (auto t : m_image_channels_to_use) {
      std::cerr << t << " ";
    }
  }
}

void jag_store::allocate_memory() {
  size_t nthreads = omp_get_max_threads();
  m_data_inputs.resize(nthreads);
  m_data_scalars.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    m_data_inputs[j].resize(m_inputs_to_use.size());
    m_data_scalars[j].resize(m_scalars_to_use.size());
  }

  m_data_images.resize(nthreads);  
  for (size_t j=0; j<m_data_images.size(); j++) {
    m_data_images[j].resize(get_total_num_channels());
    for (size_t i=0; i<m_data_images[j].size(); i++) {
      m_data_images[j][i].resize(get_linearized_channel_size());
    }  
  }
}

} // namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT
