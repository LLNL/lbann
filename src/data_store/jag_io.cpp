#include "lbann_config.hpp" // may define LBANN_HAS_CONDUIT

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_store/jag_io.hpp"
#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann_config.hpp" 
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <cstdlib>
#include <algorithm>

namespace lbann {

jag_io::~jag_io() {
  if (m_data_stream != nullptr && m_data_stream->is_open()) {
    m_data_stream->close();
    delete m_data_stream;
  }
}

jag_io::jag_io() : m_data_stream(nullptr) {}

void jag_io::get_hierarchy(conduit::Node &nd, std::string parent) {
  std::string parent_2 = parent;
  if (parent.find('/') != std::string::npos) {
    // hack to discard keys that vary between samples;
    // will fix later, when we have the samples we're going
    // to actually use, and guidance as to which outputs
    // we should use.
    if (parent.find("outputs/scalars") == std::string::npos) {
      m_keys.push_back( parent.substr(2));
    }  
  } 
  conduit::Node nd2 = nd[parent];
  const std::vector<std::string> &children_names = nd2.child_names();
  for (auto t : children_names) {
    m_parent_to_children[parent_2].insert(t);
    std::string p = parent + '/' + t;
    get_hierarchy(nd, p);
  }
}

void jag_io::convert(std::string conduit_pathname, std::string base_dir) {
  std::stringstream err;

  //create the output directory (if it doesn't already exist)
  create_dir(base_dir);

  // load the conduit bundle
  std::cerr << "Loading conduit file ...\n";
  double tm1 = get_time();
  conduit::Node head;
  conduit::relay::io::load(conduit_pathname, "hdf5", head);
  std::cerr << "time to load: " << get_time() - tm1 << "\n";
  m_num_samples = head.number_of_children();
  std::cerr << "\nconversion in progress for " << m_num_samples << " samples\n";

  // get the hierarchy (get all keys in the hierarchy); this fills in m_keys 
  // and m_parent_to_children
  get_hierarchy(head, "6");

  // fill in m_metadata
  m_sample_offset = 0;
  for (auto t : m_keys) {
    conduit::Node nd = head["0/" + t];
    conduit::DataType m = nd.dtype();
    m_metadata[t] = MetaData((TypeID)m.id(), m.number_of_elements(), m.element_bytes(), m_sample_offset);
    m_sample_offset += (m.number_of_elements() * m.element_bytes());
  }

  // write metadata to file
  std::string fn = base_dir + "/metadata.txt";
  std::ofstream metadata_writer(fn.c_str());
  if (!metadata_writer.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for writing";
    throw lbann_exception(err.str());
  }

  metadata_writer << m_num_samples << "\n";
  metadata_writer << m_sample_offset << "\n";
  for (auto t : m_keys) {
    metadata_writer << m_metadata[t].dType << " " << m_metadata[t].num_elts
           << " " << m_metadata[t].num_bytes << " " << m_metadata[t].offset 
           << " " << t << "\n";
  }
  metadata_writer.close();
  std::cerr << "wrote file: " << fn << "\n";


  // open output file for binary data
  fn = base_dir + "/data.bin";
  std::ofstream bin_writer(fn.c_str(), std::ios::binary);
  if (!bin_writer.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for writing";
    throw lbann_exception(err.str());
  }

  // write binary data
  for (size_t j=0; j<m_num_samples; j++) {
    for (auto t2 : m_keys) {
      const conduit::Node d = head[std::to_string(j) + '/' + t2];
      const TypeID t = (TypeID)d.dtype().id();
      if (m_metadata.find(t2) == m_metadata.end()) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "key is missing from metadata map: " << t2;
        throw lbann_exception(err.str());
      }
      size_t total_bytes = m_metadata[t2].num_elts * m_metadata[t2].num_bytes;

      if (total_bytes) {
        //as of now I'm only coding for the dataTypes that are in our
        //current *.bundle files; we may need to add additional later,
        //if the schema changes
        switch (t) {
          case TypeID::CHAR8_STR_ID :
            bin_writer.write((char*)d.as_char8_str(), total_bytes);
            break;
          case TypeID::FLOAT64_ID :
            bin_writer.write((char*)d.as_float64_ptr(), total_bytes);
            break;
          case TypeID::UINT64_ID :
            bin_writer.write((char*)d.as_uint64_ptr(), total_bytes);
            break;
          case TypeID::INT64_ID :
            bin_writer.write((char*)d.as_int64_ptr(), total_bytes);
            break;
          default:
            err << __FILE__ << " " << __LINE__ << " :: "
                << "get_value() failed; dType: " << d.dtype().name()
                << " " << (std::to_string(j) + '/' + t2);
            throw lbann_exception(err.str());
        }
      }
    }
  }
  bin_writer.close();
  std::cerr << "wrote " << fn << "\n";

  #if 0
  //write scalar keys
  fn = base_dir + "/scalar_keys.txt";
  std::ofstream out_scalars(fn.c_str());
  if (!out_scalars.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for writing";
    throw lbann_exception(err.str());
  }
  const conduit::Node & n_scalar = head["0/outputs/scalars"];
  conduit::NodeConstIterator itr5 = n_scalar.children();
  while (itr5.has_next()) {
    itr5.next();
    out_scalars << itr5.name() << "\n";
  }
  out_scalars.close();
  std::cerr << "wrote " << fn << "\n";
  #endif

  //write input keys
  fn = base_dir + "/input_keys.txt";
  std::ofstream out_inputs(fn.c_str());
  if (!out_inputs.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for writing";
    throw lbann_exception(err.str());
  }
  const conduit::Node & n_input = head["0/inputs"];
  conduit::NodeConstIterator itr6 = n_input.children();
  while (itr6.has_next()) {
    itr6.next();
    out_inputs << itr6.name() << "\n";
  }
  out_inputs.close();
  std::cerr << "wrote " << fn << "\n";

  //write the parent_to_child mapping
  fn = base_dir + "/parent_to_child.txt";
  std::ofstream out(fn.c_str());
  if (!out.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for writing";
    throw lbann_exception(err.str());
  }
  for (auto t : m_parent_to_children) {
    out << t.first << " ";
    for (auto t2 : t.second) {
      out << t2 << " ";
    }
    out << "\n";
  }
  out.close();
  std::cerr << "wrote " << fn << "\n";

  std::cerr << "finished conversion!\n";
}

void jag_io::load(std::string base_dir) {

  std::stringstream err;
  std::string fn;
  std::ifstream in;
  std::string key;

  // open the binary data file
  fn = base_dir + "/data.bin";
  m_data_stream = new std::ifstream(fn.c_str(), std::ios::in | std::ios::binary);
  if (! m_data_stream->good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }

  // fill in parent_to_child map
  fn = base_dir + "/parent_to_child.txt";
  in.open(fn.c_str());
  if (!in.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  std::string parent;
  std::string child;
  while (in >> parent >> child) {
    m_parent_to_children[parent].insert(child);
  }
  in.close();

  // open metadata file
  fn = base_dir + "/metadata.txt";
  in.open(fn.c_str(), std::ios::in | std::ios::binary);
  if (!in.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }

  // get num_samples, etc.
  in >> m_num_samples;
  in >> m_sample_offset;

  // fill in the metadata map
  uint64 dType;
  int num_elts;
  int bytes_per_elt;
  size_t offset;
  while (in >> dType >> num_elts >> bytes_per_elt >> offset >> key) {
    m_metadata[key] = MetaData((TypeID)dType, num_elts, bytes_per_elt, offset);
    m_keys.push_back(key);
  }
  in.close();

  #if 0
  // fill in m_scalar_keys
  fn = base_dir + "/scalar_keys.txt";
  in.open(fn.c_str());
  if (!in.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  while (in >> key) {
    m_scalar_keys.push_back(key);
  }
  in.close();
  #endif

  // fill in m_input_keys
  fn = base_dir + "/input_keys.txt";
  in.open(fn.c_str());
  if (!in.good()) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  while (in >> key) {
    m_input_keys.push_back(key);
  }
  in.close();
}

const std::unordered_set<std::string> & jag_io::get_children(std::string parent) const {
  std::unordered_map<std::string, std::unordered_set<std::string>>::const_iterator t;
  t = m_parent_to_children.find(parent);
  if (t == m_parent_to_children.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find " << parent << " in m_parent_to_children map\n"
        << "m_parent_to_children.size(): " << m_parent_to_children.size();
    throw lbann_exception(err.str());
  }
  return (*t).second;
}

size_t jag_io::get_sample_id(std::string node_name) const {
  std::stringstream err;
  size_t j = node_name.find('/');
  if (j == std::string::npos) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find '/' in node_name: " << node_name;
    throw lbann_exception(err.str());
  }
  for (size_t i=0; i<j; i++) {
    if (! isdigit(node_name[i])) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "isdigit(" << node_name << "[" << i << "] failed";
      throw lbann_exception(err.str());
    }
  }
  return atoi(node_name.data());
}

std::string jag_io::get_metadata_key(std::string node_name) const {
  std::stringstream err;
  size_t j = node_name.find('/');
  if (j == std::string::npos) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "failed to find '/' in node_name: " << node_name;
    throw lbann_exception(err.str());
  }
  std::string key = node_name.substr(j+1);
  key_exists(key);
  return key;
}

void jag_io::key_exists(std::string key) const {
  if (m_metadata.find(key) == m_metadata.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "the key: " << key << " is not valid for the metadata map";
    throw lbann_exception(err.str());
  }
}

void jag_io::get_data(std::string node_name, char * data_out, size_t num_bytes) {
  std::string key = get_metadata_key(node_name);
  size_t sample_id = get_sample_id(node_name);
  size_t offset = m_sample_offset * sample_id + m_metadata[key].offset;
  m_data_stream->seekg(offset);
  m_data_stream->read(data_out, num_bytes);
}

const std::vector<std::string>& jag_io::get_input_choices() const {
  return m_input_keys;
}

/*
const std::vector<std::string>& jag_io::get_scalar_choices() const {
  return m_scalar_keys;
}
*/


void jag_io::get_metadata(std::string key, size_t &num_elts_out, size_t &bytes_per_elt_out, size_t &total_bytes_out, conduit::DataType::TypeID &type_out) {
  num_elts_out = m_metadata[key].num_elts;
  bytes_per_elt_out = m_metadata[key].num_bytes;
  total_bytes_out = num_elts_out*bytes_per_elt_out;
  type_out = m_metadata[key].dType;
}

bool jag_io::has_key(std::string key) const {
  if (m_metadata.find(key) == m_metadata.end()) {
    return false;
  }  
  return true;
}

size_t jag_io::get_offset(std::string node_name) {
  std::string key = get_metadata_key(node_name);
  size_t sample_id = get_sample_id(node_name);
  return sample_id*m_sample_offset + m_metadata[key].offset;
}

void jag_io::print_metadata() {
  for (auto key : m_keys) {
    std::cerr << "type/num_elts/bytes_per_elt: " << m_metadata[key].dType
              << " " << m_metadata[key].num_elts << " " << m_metadata[key].num_bytes << " offset: " << m_metadata[key].offset << " :: " << key << "\n";
  }
}

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT

