#include "lbann/data_store/jag_store.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "lbann/data_readers/data_reader_jag_conduit_hdf5.hpp"
#include "lbann/utils/glob.hpp"
#include <cmath>
#include <limits>
#include "hdf5.h"
#include <unordered_set>

namespace lbann {

jag_store::jag_store()
  : m_image_size(0),
    m_comm(nullptr),
    m_master(false),
    m_max_samples(INT_MAX)
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

void jag_store::load_scalars_to_use(const std::string &keys) {
  m_scalars_to_use.clear();
  load_keys(m_scalars_to_use, keys);
}

void jag_store::load_inputs_to_use(const std::string &keys) {
  m_inputs_to_use.clear();
  load_keys(m_inputs_to_use, keys);
}

void jag_store::load_image_views_to_use(const std::string &keys) {
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

void jag_store::load_image_channels_to_use(const std::string &keys) {
   std::stringstream s;
   s << keys;
   int channel;
   while (s >> channel) {
     m_image_channels_to_use.push_back(channel);
   }
}

void jag_store::build_conduit_index(const std::vector<std::string> &filenames) {
  options *opts = options::get();
  if (!opts->has_string("base_dir")) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must pass --base_dir=<string> on the cmd line");
  }
  const std::string base_dir = opts->get_string("base_dir");
  const std::string output_fn = opts->get_string("build_conduit_index");
  std::stringstream ss;
  ss << output_fn << "." << m_rank_in_world;
  std::ofstream out(ss.str().c_str());
  if (!out.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + output_fn + " for writing");
  }
  if (m_master) std::cerr << "writing index file: " << output_fn << "\n";
  if (m_rank_in_world == 0) {
    out << base_dir << "\n";
  }
  if (m_master) std::cerr << "base dir: " << base_dir << "\n";

  int global_num_samples = 0;
  for (size_t j=m_rank_in_world; j<filenames.size(); j+=m_num_procs_in_world) {
    out << filenames[j] << " ";
    std::string fn(base_dir);
    fn += '/';
    fn += filenames[j];
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( fn );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    size_t is_good = 0;
    size_t is_bad = 0;
    std::stringstream s5;
    conduit::Node n_ok;
    for (size_t h=0; h<cnames.size(); h++) {
      const std::string key_1 = "/" + cnames[h] + "/performance/success";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key_1, n_ok);
      int success = n_ok.to_int64();
      if (success == 1) {
        ++is_good;
      } else {
        s5 << h << " ";
        ++is_bad;
      }
    }
    global_num_samples += is_good;
    out << is_good << " " << is_bad << " " << s5.str() << "\n";
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
  out.close();
  m_comm->global_barrier();

  int num_samples;
  MPI_Reduce(&global_num_samples, &num_samples, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  //m_comm->reduce<int>(&global_num_samples, 1, 0, m_comm->get_world_comm(), El::mpi::SUM);
  //

  if (m_master) {
    std::stringstream s3;
    s3 << "echo " << num_samples << " " << filenames.size() << " >  num_samples_tmp";
    system(s3.str().c_str());
    s3.clear();
    s3.str("");
    s3 << "cat num_samples_tmp ";
    for (int k=0; k<m_num_procs_in_world; k++) {
      s3 << output_fn << "." << k << " ";
    }
    s3 << "> " << output_fn;
    system(s3.str().c_str());
    s3.clear();
    s3.str("");
    s3 << "chmod 660 " << output_fn;
    system(s3.str().c_str());
    s3.clear();
    s3.str("");
    s3 << "rm -f num_samples_tmp ";
    for (int k=0; k<m_num_procs_in_world; k++) {
      s3 << output_fn << "." << k << " ";
    }
    system(s3.str().c_str());
  }
  m_comm->global_barrier();
}

void jag_store::setup_testing() {
  setup_conduit();
  setup_binary();
}

void jag_store::setup(
  data_reader_jag_conduit_hdf5 *reader,
  bool num_stores,
  int my_rank) {
  double tm1 = get_time();

  m_master = m_comm->am_world_master();
  options *opts = options::get();
  m_reader = reader;

  m_max_samples = INT_MAX;
  if (opts->has_int("max_samples")) {
    m_max_samples = (size_t)opts->get_int("max_samples");
  }

  bool has_conduit_filenames = false;
  if (opts->has_string("conduit_filelist")) {
    std::string f = opts->get_string("conduit_filelist");
    std::ifstream in(f.c_str());
    if (!in) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + f + " for reading");
    }
    std::string line;
    while (getline(in, line)) {
      m_conduit_filenames.push_back(line);
    }
    in.close();
    if (m_max_samples < m_conduit_filenames.size()) {
      m_conduit_filenames.resize(m_max_samples);
    }
    has_conduit_filenames = true;
  }

  if (m_image_size == 0) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: image_size = 0; probably set_image_size() has not been called");
  }

  // optionally build an index file, then exit. Each line of the file will
  // contain a conduit filename, followed by the valid sample_ids in
  // the conduit file
  if (opts->has_string("build_conduit_index")) {
    if (! has_conduit_filenames) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must pass --conduit_filenames=<string> on the cmd line when building a conduit index");
    }
    build_conduit_index(m_conduit_filenames);
    exit(0);
  }

  load_variable_names();
  build_data_sizes();
  report_linearized_sizes();
  allocate_memory();
  load_normalization_values();

  if (!opts->has_int("mode")) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must pass --mode=<int> on cmd line, where <int> is 1 (to use conduit files) or 2 or 3 (for testing) (to use binary files)");
  }
  m_mode = opts->get_int("mode");
  if (! (m_mode == 1 || m_mode == 2 || m_mode == 3)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must pass --mode=<int> on cmd line, where <int> is 1 (to use conduit files) or 2 (to use binary files); or 4 (for testing) you passed: " + std::to_string(m_mode));
  }
  if (m_master) std::cerr << "Running in mode: " << m_mode << "\n";

  // optionally convert conduit files to our binary format, then exit
  if (opts->has_string("convert_conduit")) {
    if (! has_conduit_filenames) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must pass --conduit_filenames=<string> on the cmd line when converting conduit filenames to binary");
    }
    setup_conduit();
    convert_conduit_to_binary(m_conduit_filenames);
    exit(0);
  }

  if (m_mode == 1) {
    setup_conduit();
  } else if (m_mode == 2) {
    setup_binary();
  } else {
    setup_testing();
  }

  if (m_master) {
    std::cerr << "jag_store::setup time: " << get_time() - tm1 << "; num samples: " << m_num_samples << std::endl;
  }

  if (m_mode == 3) {
    test_converted_files();
    m_comm->global_barrier();
    exit(0);
  }

  // optionally compute min/max values, then exit.
  // This is only needed for one-time computation of normalization values
  if (opts->has_string("compute_min_max")) {
    compute_min_max();
    exit(0);
  }

  // optionally check bandwidth (sort of), then exit
  if (opts->has_int("bandwidth")) {
    if (m_mode == 0) {
      compute_bandwidth();
    } else {
      compute_bandwidth_binary();
    }
    exit(0);
  }
}

size_t jag_store::get_linearized_data_size() const {
  size_t n = m_image_views_to_use.size() * m_image_channels_to_use.size() * get_linearized_channel_size()
           + m_scalars_to_use.size()
           + m_inputs_to_use.size();
  return n;
}

void jag_store::build_data_sizes() {
  for (size_t i=0; i<get_total_num_channels(); i++) {
    m_data_sizes.push_back(get_linearized_channel_size());
  }
  if (get_linearized_scalar_size() > 0.0) {
    m_data_sizes.push_back(get_linearized_scalar_size());
  }
  if (get_linearized_input_size() > 0.0) {
    m_data_sizes.push_back(get_linearized_input_size());
  }
}

void jag_store::report_linearized_sizes() {
  if (! m_master) {
    return;
  }
  std::cerr
    << "===================================================================\n"
    << "LINEARIZED SIZES REPORT:\n"
    << "get_linearized_data_size:  " << get_linearized_data_size() << "\n"
    << "get_linearized_image_size:   " << get_linearized_image_size() << "\n"
    << "get_linearized_channel_size: " << get_linearized_channel_size() << "\n"
    << "get_num_channels: " << get_num_channels_per_view() << "\n"
    << "get_linearized_scalar_size:  " << get_linearized_scalar_size() << "\n"
    << "get_linearized_input_size:   " << get_linearized_input_size() << "\n"
    << "get_num_img_srcs:            " << get_num_img_srcs() << "\n"
    << "sizes vector: ";
  size_t total = 0;
  for (auto t : m_data_sizes) {
    std::cerr << t << " ";
    total += t;
  }
  std::cerr << "\n";
  std::cerr << "total, from m_data_sizes; should be same as above: "
    << total << "\n"
    << "===================================================================\n";
}

void jag_store::load_data_binary(int data_id, int tid) {
  const int file_idx = m_sample_map[data_id].first;
//  std::string fn = m_binary_filenames[file_idx];
  const int sample_idx = m_sample_map[data_id].second;

 // std::ifstream in(fn.c_str(), std::ios::out | std::ios::binary);
  /*
  if (!in.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open: " + fn + " for reading; data_id: " + std::to_string(data_id) + " tid: " + std::to_string(tid));
  }
  */

//  in.seekg(sample_idx*m_sample_len);
  m_streams[tid][file_idx]->seekg(sample_idx*m_sample_len);
  m_streams[tid][file_idx]->read((char*)m_scratch[tid].data(), m_sample_len);
  //in.read((char*)m_scratch[tid].data(), m_sample_len);
//  in.close();

//  size_t offset = sample_idx * m_sample_len;

//  in.seekg(offset);
 // in.read((char*)m_scratch[tid].data(), m_sample_len);

  for (size_t j=0; j<m_inputs_to_use.size(); j++) {
    check_entry(m_inputs_to_use[j]);
    memcpy((void*)(m_data_inputs[tid].data()+j), (void*)(m_scratch[tid].data()+m_key_map[m_inputs_to_use[j]]), 8);
  }
  for (size_t j=0; j<m_data_inputs[tid].size(); j++) {
    m_data_inputs[tid][j] = m_data_inputs[tid][j]*m_normalize_inputs[j].first - m_normalize_inputs[j].second;
  }

  for (size_t j=0; j<m_scalars_to_use.size(); j++) {
    check_entry(m_scalars_to_use[j]);
    memcpy((void*)(m_data_scalars[tid].data()+j), (void*)(m_scratch[tid].data()+m_key_map[m_scalars_to_use[j]]), 8);
  }
  for (size_t j=0; j<m_data_scalars[tid].size(); j++) {
    m_data_scalars[tid][j] = m_data_scalars[tid][j]*m_normalize_scalars[j].first - m_normalize_scalars[j].second;
  }

  size_t y = 0;
  for (size_t view=0; view<m_image_views_to_use.size(); view++) {
    check_entry(m_image_views_to_use[view]);
    for (size_t k=0; k<m_image_channels_to_use.size(); k++) {
      int channel = m_image_channels_to_use[k];

      memcpy((void*)m_data_images[tid][y].data(),
             (void*)(m_scratch[tid].data()+m_key_map[m_image_views_to_use[view]] + channel*get_linearized_channel_size()*sizeof(data_reader_jag_conduit_hdf5::ch_t)), get_linearized_channel_size());
      for (size_t x=0; x<m_data_images[tid][y].size(); x++) {
        m_data_images[tid][y][x] = m_data_images[tid][y][x]*m_normalize_views[channel].first - m_normalize_views[channel].second;
      }
      ++y;
    }
  }
}

void jag_store::load_data_conduit(int data_id, int tid) {
  //map data_id to the correct file and sample_id
  int idx = m_data_id_to_conduit_filename_idx[data_id];
  const std::string &filename = m_conduit_filenames[idx];
  const std::string sample_id = m_data_id_to_sample_id[data_id];
  hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(filename);

  conduit::Node node;

  size_t j = 0;
  for (auto input_name : m_inputs_to_use) {
    const std::string key = sample_id + "/inputs/" + input_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if input_t changes
    double d = node.to_float64();
    d = d*m_normalize_inputs[j].first - m_normalize_inputs[j].second;
    m_data_inputs[tid][j++] = d;
  }

  j = 0;
  for (auto scalar_name : m_scalars_to_use) {
    const std::string key = sample_id + "/outputs/scalars/" + scalar_name;
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    //this is fragile; will break if scalar_t changes
    double d = node.to_float64();
    d = d*m_normalize_scalars[j].first - m_normalize_scalars[j].second;
    m_data_scalars[tid][j++] = d;
  }

  j = 0;
  for (auto image_name : m_image_views_to_use) {
    const std::string key = sample_id + "/outputs/images/" + image_name + "/0.0/emi";
    conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
    conduit::float32_array emi = node.value();
    const size_t image_size = emi.number_of_elements();
    //this is fragile; will break if ch_t changes
    for (size_t h=0; h<m_image_channels_to_use.size(); h++) {
      int channel = m_image_channels_to_use[h];
      int k = 0;
      for (size_t i=channel; i<image_size; i+=4) {
        float d = emi[i];
        d = d*m_normalize_views[channel].first - m_normalize_views[channel].second;
        m_data_images[tid][j][k++] = d;
      }
    }
    ++j;
  }
  conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
}

void jag_store::open_binary_file_for_output(const std::string &dir) {
  if (m_binary_output_file.is_open()) {
    m_binary_output_file.close();
    m_binary_output_file_names.close();
    ++m_global_file_idx;
  }

  std::stringstream s;
  s << dir << "/" << BINARY_FILE_BASENAME << "_" << m_global_file_idx << ".bin";
  m_binary_output_file.open(s.str().c_str(), std::ios::out | std::ios::binary);
  if (!m_binary_output_file) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + s.str() + " for writing");
  }
  m_binary_output_filename = s.str();
  std::cerr << "opened for writing: " << s.str() << "\n";

  s.clear();
  s.str("");
  s << dir << "/" << BINARY_FILE_BASENAME << "_" << m_global_file_idx << "_names.txt";
  m_binary_output_file_names.open(s.str());
  if (!m_binary_output_file_names) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + s.str() + " for writing");
  }
  std::cerr << "opened for writing: " << s.str() << "\n";
}

void jag_store::write_binary(const std::vector<std::string> &filenames, const std::string &dir) {
  if (m_master) std::cerr << "starting jag_store::write_binary\n";
  options *opts = options::get();
  const std::string output_dir = opts->get_string("convert_conduit");

  m_global_file_idx = 0;
  m_num_converted_samples = 0;
  m_binary_output_filename = "";
  open_binary_file_for_output(output_dir);

  size_t num_samples_written = 0;
  std::string fn;
  for (size_t k=0; k<filenames.size(); ++k) {
    std::stringstream s2;
    s2 << filenames[k];
    s2 >> fn;
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( fn );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    if (m_master) std::cerr << "  num samples this file: " << cnames.size() << "\n";

    conduit::Node n_ok;
    conduit::Node node;
    for (auto sample_name : cnames) {
      const std::string key_1 = "/" + sample_name + "/performance/success";
      conduit::relay::io::hdf5_read(hdf5_file_hnd, key_1, n_ok);
      int success = n_ok.to_int64();
      if (success == 1) {
        m_binary_output_file_names << sample_name << "\n";
        for (auto input_name : m_inputs_to_use) {
          const std::string key = "/" + sample_name + "/inputs/" + input_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if input_t changes
          double tmp = node.to_float64();
          m_binary_output_file.write((char*)&tmp, sizeof(data_reader_jag_conduit_hdf5::input_t));
        }

        for (auto scalar_name : m_scalars_to_use) {
          const std::string key = "/" + sample_name + "/outputs/scalars/" + scalar_name;
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          //this is fragile; will break if scalar_t changes
          double tmp = node.to_float64();
          m_binary_output_file.write((char*)&tmp, sizeof(data_reader_jag_conduit_hdf5::scalar_t));
        }

        for (auto image_name : m_image_views_to_use) {
          const std::string key = "/" + sample_name + "/outputs/images/" + image_name + "/0.0/emi";
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, node);
          conduit::float32_array emi = node.value();
          const size_t image_size = emi.number_of_elements();
          //this is fragile; will break if ch_t changes
          for (int channel=0; channel<4; channel++) {
            for (size_t j=channel; j<image_size; j+=4) {
              m_binary_output_file.write((char*)&emi[j], sizeof(data_reader_jag_conduit_hdf5::ch_t));

            }
          }
        }
        ++m_num_converted_samples;
        if (m_num_converted_samples >= m_max_samples) {
          conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
          goto EARLY_EXIT;
          break;
        }
        ++num_samples_written;
        if (num_samples_written == MAX_SAMPLES_PER_BINARY_FILE) {
          num_samples_written = 0;
          open_binary_file_for_output(output_dir);
        }
      }
    }
    conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
  }
EARLY_EXIT :
  m_binary_output_file.close();
  m_binary_output_file_names.close();
  if (m_master) std::cerr << "LEAVING jag_store::write_binary\n";
}

void jag_store::read_key_map(const std::string &filename) {
  if (m_master) std::cerr << "starting jag_store::read_key_map; opening file: " << filename << "\n";
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

  if (m_master) {
    for (auto t : m_key_map) {
      std::cerr << "key: " << t.first << " offset: " << t.second << "\n";
    }
  }
}

void jag_store::write_binary_metadata(std::string dir) {
  std::stringstream s;
  s << dir << "/" << METADATA_FN;
  std::ofstream out(s.str().c_str());
  if (m_master) std::cerr << "writing metadata for file: " << s.str() << "\n";
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

void jag_store::convert_conduit_to_binary(const std::vector<std::string> &conduit_filenames) {
  m_num_converted_samples = 0;

  if (m_comm->get_procs_in_world() != 1) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must run convert_conduit with a single processor");
  }

  options *opts = options::get();
  std::string output_dir = opts->get_string("convert_conduit");
  if (m_master) {
    char b[128];
    sprintf(b, "mkdir --mode=770 -p %s", output_dir.c_str());
    system(b);
    write_binary_metadata(output_dir);
  }
  write_binary(conduit_filenames, output_dir);
}

void jag_store::load_variable_names() {
  load_inputs_to_use(m_reader->m_input_keys);
  load_scalars_to_use(m_reader->m_scalar_keys);
  load_image_views_to_use(m_reader->m_image_views);
  load_image_channels_to_use(m_reader->m_image_channels);

  if (m_master) {
    std::cerr << "using these inputs:\n";
    for (auto t : m_inputs_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\nusing these scalars:\n";
    for (auto t : m_scalars_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\nusing these views:\n";
    for (auto t : m_image_views_to_use) {
      std::cerr << "    " << t << "\n";
    }
    std::cerr << "\nusing these image channels: ";
    for (auto t : m_image_channels_to_use) {
      std::cerr << t << " ";
    }
    std::cerr << "\n";
  }
}

void jag_store::allocate_memory() {
  size_t nthreads = omp_get_max_threads();
  if (m_master) std::cerr << "starting jag_store::allocate_memory; nthreads: " << nthreads << "\n";
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

void jag_store::test_converted_files() {
  int np = m_comm->get_procs_in_world();
  if (np != 1) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: mode 3 (test converted binary files) must be run with a single process");
  }
  std::cerr << "\nstarting jag_store::test_converted_files()\n";

  std::vector<std::vector<data_reader_jag_conduit_hdf5::input_t>> inputs;
  std::vector<std::vector<data_reader_jag_conduit_hdf5::scalar_t>> scalars;
  std::vector<std::vector<std::vector<data_reader_jag_conduit_hdf5::ch_t>>> images;

  int tid = 0;
  options *opts = options::get();
  if (!opts->has_int("num_to_test")) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: when running in test mode you must pass --num_to_test=<int> on the cmd line");
  }
  size_t num_to_test = opts->get_int("num_to_test");
  std::cerr << "\nnum to test: " << num_to_test << "\n";
  for (size_t data_id=0; data_id<num_to_test; data_id++) {

    // sanity checks
    if (data_id >= m_data_id_to_sample_id.size()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: data_id: " + std::to_string(data_id) + " >= m_data_id_to_sample_id.size(): " + std::to_string(m_data_id_to_sample_id.size()));
    }

    const std::string sample_id = m_data_id_to_sample_id[data_id];

    if (m_sample_id_to_global_idx.find(sample_id) == m_sample_id_to_global_idx.end()) {
    std::cerr << "discarding " << sample_id << " since it's not found in m_sample_id_to_global_idx\n";
      //throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to find " + sample_id + " in m_sample_id_to_global_idx; data_id: " + std::to_string(data_id));
    }

else {
    int global_id = m_sample_id_to_global_idx[sample_id];

    std::cerr << "testing sample: " << sample_id << " data_id: " << data_id << " global_id: " << global_id << "\n";

    load_data_conduit(data_id, tid);
    inputs = m_data_inputs;
    scalars = m_data_scalars;
    images = m_data_images;

    load_data_binary(global_id, tid);

    if (inputs != m_data_inputs) {
      std::cerr << "inputs for data_id " << data_id << " failed.\n"
                << "values from conduit: ";
      for (auto t : inputs[tid]) std::cerr << t << " ";
      std::cerr << "\nvalues from binary:  ";
      for (auto t : m_data_inputs[tid]) std::cerr << t << " ";
      std::cerr << "\n";
      exit(9);
    }
    if (scalars != m_data_scalars) {
      std::cerr << "scalars != m_data_scalars\n";
      exit(9);
    }

    std::cerr << "1. num channels: " << images[0].size() << "\n";
    std::cerr << "2. num channels: " << m_data_images[0].size() << "\n";
    for (size_t j=0; j<images[0].size(); j++) {
      if (images[0][j] != m_data_images[0][j]) {
        std::cerr << "FAILED: images[0][" << j << "] != m_data_images[0][" << j << "]\n";
        for (size_t x=0; x<images[0][j].size(); x++) {
          if (images[0][j][x] != m_data_images[0][j][x]) {
            bool testme = images[0][j][x] - m_data_images[0][j][x] <  std::numeric_limits<float>::epsilon();
            std::cerr << x << " " << images[0][j][x] << " " << m_data_images[0][j][x] << "  epsilon? " << testme << "\n";
          }
        }
        //exit(9);
      } else {
        std::cerr << "PASSED: images[0][" << j << "] == m_data_images[0][" << j << "]\n";
      }
    }
  }
  }
  std::cerr << "\ntested " << m_max_samples << "; all passed\n";
}

void jag_store::setup_conduit() {
  if (m_master) std::cerr << "starting jag_store::setup_conduit\n";

  std::string filename;
  std::string sample_id;
  int j = -1;
  std::vector<std::string> tmp;
  for (auto t : m_conduit_filenames) {
    if (m_data_id_to_sample_id.size() == m_max_samples) {
      break;
    }
    ++j;
    std::stringstream s(t);
    s >> filename;
    tmp.push_back(filename);
    while (s >> sample_id) {
      m_data_id_to_conduit_filename_idx.push_back(j);
      m_data_id_to_sample_id.push_back(sample_id);
      if (m_data_id_to_sample_id.size() == m_max_samples) {
        break;
      }
    }
  }
  m_conduit_filenames = tmp;
  m_num_samples = m_data_id_to_sample_id.size();
  if (m_master) std::cerr << "finished reading " << m_num_samples << " sample names\n";
}

void jag_store::setup_binary() {
  if (m_master) std::cerr << "starting jag_store::setup_binary\n";
  options *opts = options::get();
  if (!opts->has_string("binary_filelist")) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: you must pass --binary_filelist=<string> on the cmd line");
  }

  const std::string fn = opts->get_string("binary_filelist");
  std::ifstream in(fn.c_str());
  if (!in.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + fn + " for reading");
  }
  if (m_master) std::cerr << "opened " << fn << " for reading\n";

  std::string filename;
  size_t num_files = 0;
  while (in >> filename) {
    ++num_files;
  }
  in.close();

  in.open(fn.c_str());
  size_t nthreads = omp_get_max_threads();
  m_streams.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    m_streams[j].resize(num_files);
  }

  size_t global_idx = 0;
  int file_idx = -1;
  while (in >> filename) {
    if (m_master) std::cerr << "next binary filename: " << filename << "\n";
    ++file_idx;

    for (size_t tid=0; tid<nthreads; tid++) {
      m_streams[tid][file_idx] = new std::ifstream(filename.c_str(), std::ios::out | std::ios::binary);
    }

    if (global_idx == m_max_samples) {
      break;
    }

    m_binary_filenames.push_back(filename);

    size_t j = filename.rfind(".bin");
    if (j == std::string::npos) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: t.rfind('.bin') failed for filename: " + filename);
    }

    std::stringstream s;
    s << filename.substr(0, j) << "_names.txt";
    std::ifstream in2(s.str().c_str());
    if (!in2.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + s.str() + " for reading");
    }
    if (m_master) std::cerr << "opened " << s.str() << " for reading\n";

    size_t local_idx = 0;
    std::string sample_id;
    while (in2 >> sample_id) {
      //maps global index (shuffled index subscript) to <file_index,
      //num sample within the file
      m_sample_map[global_idx] = std::make_pair(file_idx, local_idx++);
      m_sample_id_to_global_idx[sample_id] = global_idx;

      //maps global index (shuffled index subscript) to sample id
      m_sample_id_map[global_idx] = sample_id;

      ++global_idx;
      if (global_idx == m_max_samples) {
        break;
      }
    }
    in2.close();
  }
  m_num_samples = m_sample_map.size();
  if (m_master) std::cerr << "num samples: " << m_num_samples << "\n";

  size_t jj = filename.rfind('/');
  if (jj == std::string::npos) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: " + filename + ".rfind('/') failed");
  }
  std::string key_fn = filename.substr(0, jj);
  key_fn += "/metadata.txt";
  read_key_map(key_fn);

  m_scratch.resize(nthreads);
  for (size_t j=0; j<nthreads; j++) {
    for (size_t i=0; i<m_scratch.size(); i++) {
      m_scratch[i].resize(m_sample_len);
    }
  }
}

void jag_store::compute_bandwidth_binary() {
  if (m_master) std::cerr << "starting bandwidth test (binary); num_samples: " << m_num_samples << " m_max_samples: " << m_max_samples << "\n";
  double tm1 = get_time();
  int me = get_rank_in_world();
  int np = m_comm->get_procs_in_world();

  LBANN_OMP_PARALLEL
  {
    const auto threadId = omp_get_thread_num();

    LBANN_OMP_PARALLEL_FOR
    for (size_t j = me; j<m_max_samples; j += np) {
      if (j % 1000 == 0 && m_master) std::cerr << "processed " << j/1000 << "K samples\n";
      load_data_binary(j, threadId);
    }
  }
  std::cerr << "P_" << me << " finished; time: " << get_time() - tm1 << "\n";
  m_comm->global_barrier();
  if (m_master) std::cerr << "time to load all data: " << get_time() - tm1 << "\n";
}

void jag_store::compute_bandwidth() {
  if (m_master) std::cerr << "starting bandwidth test\n";
  double tm1 = get_time();
  int me = get_rank_in_world();
  int np = m_comm->get_procs_in_world();
  size_t n = 0;
  for (size_t j = me; j<m_data_id_to_conduit_filename_idx.size(); j+= np) {
    if (j % 1000 == 0 && m_master) std::cerr << "processed " << j/1000 << "K samples\n";
    load_data(j, 0);
    n += np;
  }
  std::cerr << "P_" << me << " finished; time: " << get_time() - tm1 << "\n";
  m_comm->global_barrier();
  if (m_master) std::cerr << "time to load all data: " << get_time() - tm1 << "\n";
}

void jag_store::compute_min_max() {
  std::vector<double> inputs_max(m_inputs_to_use.size(), DBL_MIN);
  std::vector<double> inputs_min(m_inputs_to_use.size(), DBL_MAX);
  std::vector<double> inputs_avg(m_inputs_to_use.size(), 0.);
  std::vector<double> scalars_max(m_scalars_to_use.size(), DBL_MIN);;
  std::vector<double> scalars_min(m_scalars_to_use.size(), DBL_MAX);;
  std::vector<double> scalars_avg(m_scalars_to_use.size(), 0.);;

  for (size_t j = 0; j<m_data_id_to_conduit_filename_idx.size(); j++) {
    if (j == m_max_samples) {
      break;
    }
    if (j % 1000 == 0) std::cerr << "processed " << j/1000 << "K samples\n";
    load_data(j, 0);
    const std::vector<data_reader_jag_conduit_hdf5::input_t> &t1 = fetch_inputs(j, 0);
    for (size_t h=0; h<t1.size(); h++) {
      if (j == 0) {
        inputs_min[h] = t1[h];
        inputs_max[h] = t1[h];
        inputs_avg[h] += t1[h];
      } else {
        inputs_avg[h] += t1[h];
        if (t1[h] > inputs_max[h]) inputs_max[h] = t1[h];
        if (t1[h] < inputs_min[h]) inputs_min[h] = t1[h];
      }
    }

    const std::vector<data_reader_jag_conduit_hdf5::scalar_t> &t2 = fetch_scalars(j, 0);
    for (size_t h=0; h<t2.size(); h++) {
      scalars_avg[h] += t2[h];
      if (t2[h] > scalars_max[h]) scalars_max[h] = t2[h];
      if (t2[h] < scalars_min[h]) scalars_min[h] = t2[h];
    }
  }
  std::cerr << "\n\ninputs min: ";
  for (auto t : inputs_min) std::cerr << t << " ";
  std::cerr << "\ninputs max: ";
  for (auto t : inputs_max) std::cerr << t << " ";
  std::cerr << "\ninputs avg: ";
  for (auto t : inputs_avg) std::cerr << t/m_data_id_to_conduit_filename_idx.size() << " ";
  std::cerr << "\n\n";
  std::cerr << "\n\nscalars min: ";
  for (auto t : scalars_min) std::cerr << t << " ";
  std::cerr << "\nscalars max: ";
  for (auto t : scalars_max) std::cerr << t << " ";
  std::cerr << "\nscalars avg: ";
  for (auto t : scalars_avg) std::cerr << t/m_data_id_to_conduit_filename_idx.size() << " ";
  std::cerr << "\n\n";
}

void jag_store::load_normalization_values_impl(
    std::vector<std::pair<double, double>> &values,
    const std::vector<std::string> &variables) {
  values.resize(variables.size());
  for (size_t j=0; j<values.size(); j++) {
    values[j] = std::make_pair(1.0, 0.0);
  }

  options *opts = options::get();
  if (!opts->has_string("normalization_fn")) {
    if (m_master) {
      std::cerr << "\nWARNING! missing --normalization_fn option on command line; inputs, scalars, and possibly images will not be normalized. This is probably a bad thing.\n";
    }
  } else {
    const std::string fn = opts->get_string("normalization_fn");
    std::unordered_map<std::string, std::pair<double, double>> m;
    std::ifstream in(fn.c_str());
    if (!in.good()) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + fn + " for reading");
    }
    std::string variable;
    double scale;
    double bias;
    while (in >> variable >> scale >> bias) {
      m[variable] = std::make_pair(scale, bias);
    }
    in.close();
    for (size_t j=0; j<variables.size(); j++) {
      if (m.find(variables[j]) == m.end()) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed find scale and bias value for: " + variables[j]);
      }
      values[j] = m[variables[j]];
    }
  }
}

void jag_store::load_normalization_values() {
  load_normalization_values_impl(m_normalize_inputs, m_inputs_to_use);
  load_normalization_values_impl(m_normalize_scalars, m_scalars_to_use);
  std::vector<std::string> channels_to_use;
  for (int j=0; j<4; j++) {
    std::string s = "C" + std::to_string(j);
    channels_to_use.push_back(s);
  }
  load_normalization_values_impl(m_normalize_views, channels_to_use);
}


} // namespace lbann
#endif //ifdef LBANN_HAS_CONDUIT
