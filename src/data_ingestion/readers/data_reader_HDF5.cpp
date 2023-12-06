////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
/////////////////////////////////////////////////////////////////////////////////
#include "conduit/conduit_relay_mpi.hpp"

#include "lbann/data_ingestion/readers/data_reader_HDF5.hpp"
#include "lbann/data_ingestion/readers/data_reader_sample_list_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list_open_files_impl.hpp"
#include "lbann/transforms/repack_HWC_to_CHW_layout.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {
namespace {

template <typename T>
void do_normalize(T* const data,
                  double const scale,
                  double const bias,
                  size_t const n_elts)
{
  std::for_each(data, data + n_elts, [scale, bias](auto& x) {
    x = x * scale + bias;
  });
}

template <typename T>
void do_normalize(T* const data,
                  const double* scale,
                  const double* bias,
                  size_t const n_elts,
                  size_t const n_channels)
{
  size_t const n_elts_per_channel = n_elts / n_channels;
  for (size_t j = 0; j < n_elts_per_channel; j++) {
    for (size_t k = 0; k < n_channels; k++) {
      size_t const idx = j * n_channels + k;
      data[idx] = (data[idx] * scale[k] + bias[k]);
    }
  }
}

// Pack 2-D tensor into a channels-first from a channels-last storage format
template <typename T>
void do_repack_tensor_HWC_to_CHW(T* const src_buf,
                                 size_t const n_elts,
                                 size_t const n_rows,
                                 size_t const n_cols,
                                 size_t const n_channels)
{
  std::vector<T> work;
  work.reserve(n_elts);
  T* const dst_buf = work.data();
  std::vector<size_t> dims = {n_channels, n_rows, n_cols};
  transform::repack_HWC_to_CHW(src_buf, dst_buf, dims);
  std::copy_n(dst_buf, n_elts, src_buf);
}

// Pack 3-D tensor into a channels-first from a channels-last storage format
template <typename T>
void do_repack_tensor_DHWC_to_CDHW(T* const src_buf,
                                   size_t const n_elts,
                                   size_t const depth,
                                   size_t const height,
                                   size_t const width,
                                   size_t const n_channels)
{
  std::vector<T> work;
  work.reserve(n_elts);
  T* const dst_buf = work.data();
  std::vector<size_t> dims = {n_channels, depth, height, width};
  transform::repack_DHWC_to_CDHW(src_buf, dst_buf, dims);
  std::copy_n(dst_buf, n_elts, src_buf);
}

} // namespace

template <typename T>
void hdf5_data_reader::pack(std::string const& group_name,
                            conduit::Node& node,
                            size_t const index)
{
  if (m_packing_groups.find(group_name) == m_packing_groups.end()) {
    LBANN_ERROR("(m_packing_groups.find(", group_name, ") failed");
  }
  const PackingGroup& g = m_packing_groups[group_name];
  std::vector<T> data(g.n_elts);
  size_t idx = 0;
  for (size_t k = 0; k < g.names.size(); k++) {
    size_t const n_elts = g.sizes[k];
    std::string path;
    if (node.name() == "") {
      path = build_string(node.child(0).name(), '/', g.names[k]);
    }
    else {
      path =
        build_string(node.name(), '/', node.child(0).name(), '/', g.names[k]);
    }
    if (!node.has_path(path)) {
      LBANN_ERROR("no leaf for path: ", path);
    }
    conduit::Node& leaf = node[path];
    memcpy(data.data() + idx, leaf.element_ptr(0), n_elts * sizeof(T));
    if (m_delete_packed_fields) {
      node.remove(path);
    }
    idx += n_elts;
  }
  if (idx != g.n_elts) {
    LBANN_ERROR("idx != g.n_elts*sizeof(T): ", idx, " ", g.n_elts * sizeof(T));
  }
  std::ostringstream ss;
  ss << '/' << LBANN_DATA_ID_STR(index) + '/' + group_name;
  node[ss.str()] = std::move(data);

  // this is clumsy and should be done better
  if (m_add_to_map.find(group_name) == m_add_to_map.end()) {
    m_add_to_map.insert(group_name);
    conduit::Node metadata;
    metadata[s_composite_node] = true;
    m_experiment_schema[group_name][s_metadata_node_name] = metadata;
    m_data_schema[group_name][s_metadata_node_name] = metadata;
    m_useme_node_map[group_name] = m_experiment_schema[group_name];
    m_useme_node_map_ptrs[group_name] = &(m_experiment_schema[group_name]);
  }
}

hdf5_data_reader::~hdf5_data_reader() {}

hdf5_data_reader::hdf5_data_reader(bool shuffle)
  : data_reader_sample_list(shuffle)
{}

hdf5_data_reader::hdf5_data_reader(const hdf5_data_reader& rhs)
  : data_reader_sample_list(rhs)
{
  copy_members(rhs);
}

hdf5_data_reader& hdf5_data_reader::operator=(const hdf5_data_reader& rhs)
{
  if (this == &rhs) {
    return (*this);
  }
  data_reader_sample_list::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void hdf5_data_reader::copy_members(const hdf5_data_reader& rhs)
{
  data_reader_sample_list::copy_members(rhs);
  m_data_dims_lookup_table = rhs.m_data_dims_lookup_table;
  m_linearized_size_lookup_table = rhs.m_linearized_size_lookup_table;
  m_experiment_schema_filename = rhs.m_experiment_schema_filename;
  m_data_schema_filename = rhs.m_data_schema_filename;
  m_delete_packed_fields = rhs.m_delete_packed_fields;
  m_packing_groups = rhs.m_packing_groups;
  m_experiment_schema = rhs.m_experiment_schema;
  m_data_schema = rhs.m_data_schema;
  m_useme_node_map = rhs.m_useme_node_map;
  // m_data_map should not be copied, as it contains pointers, and is only
  // needed for setting up other structures during load

  if (rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
    m_data_store->set_data_reader_ptr(this);
  }
}

void hdf5_data_reader::load()
{
  if (get_comm()->am_world_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }
  double tm1 = get_time();
  double tm11 = tm1;
  auto& arg_parser = global_argument_parser();

  if (arg_parser.get<bool>(LBANN_OPTION_KEEP_PACKED_FIELDS)) {
    m_delete_packed_fields = false;
  }

  // May go away; for now, this reader only supports preloading mode
  // with data store
  // TODO MRW
  // opts->set_option("preload_data_store", true);
  if (!arg_parser.get<bool>(LBANN_OPTION_USE_DATA_STORE)) {
    LBANN_ERROR(
      "HDF5 data reader requires the data store.",
      "Set command line arguments --use_data_store --preload_data_store");
  }

  // Load the sample list(s)
  data_reader_sample_list::load();
  if (get_comm()->am_world_master()) {
    std::cout << "time to load sample list: " << get_time() - tm11 << std::endl;
  }

  // the usual boilerplate (we should wrap this in a function)
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();

  // Load and parse the user-supplied schemas
  tm11 = get_time();
  if (get_data_schema_filename().empty()) {
    LBANN_ERROR("you must include 'data_schema_filename' in your data reader "
                "prototext file");
  }
  if (get_experiment_schema_filename().empty()) {
    LBANN_ERROR("you must include 'data_experiment_filename' in your data "
                "reader prototext file");
  }
  load_schema(get_data_schema_filename(), m_data_schema);
  load_schema(get_experiment_schema_filename(), m_experiment_schema);
  parse_schemas();

  if (get_comm()->am_world_master()) {
    std::cout << "time to load and parse the schemas: " << get_time() - tm11
              << " for role: " << get_role() << std::endl;
    std::cout << "hdf5_data_reader::load() time: " << (get_time() - tm1)
              << "; num samples: " << m_shuffled_indices.size() << std::endl;
  }

  if (!arg_parser.get<bool>(LBANN_OPTION_QUIET) &&
      get_comm()->am_world_master()) {
    print_metadata();
  }
}

// master loads the experiment-schema then bcasts to others
void hdf5_data_reader::load_schema(std::string filename, conduit::Node& schema)
{
  if (m_comm->am_trainer_master()) {
    conduit::relay::io::load(filename, schema);
  }

  conduit::relay::mpi::broadcast_using_schema(
    schema,
    m_comm->get_trainer_master(),
    m_comm->get_trainer_comm().GetMPIComm());
}

void hdf5_data_reader::do_preload_data_store()
{
  double tm1 = get_time();
  if (get_comm()->am_world_master()) {
    std::cout << "starting hdf5_data_reader::do_preload_data_store() for role: "
              << get_role() << std::endl;
  }

  for (size_t idx = 0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if (m_data_store->get_index_owner(index) !=
        get_comm()->get_rank_in_trainer()) {
      continue;
    }
    try {
      conduit::Node& node = m_data_store->get_empty_node(index);
      load_sample_from_sample_list(node, index);
      m_data_store->set_preloaded_conduit_node(index, node);
    }
    catch (conduit::Error const& e) {
      LBANN_ERROR("trying to load the node ",
                  index,
                  " and caught conduit exception: ",
                  e.what());
    }
  }
  // Once all of the data has been preloaded, close all of the file handles

  for (size_t idx = 0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if (m_data_store->get_index_owner(index) !=
        get_comm()->get_rank_in_trainer()) {
      continue;
    }
    close_file(index); // data_reader_sample_list::close_file
  }

  size_t nn = m_data_store->get_num_global_indices();
  if (get_comm()->am_world_master()) {
    std::cout << "loading data for role: " << get_role() << " took "
              << get_time() - tm1 << "s"
              << "num samples (local to this rank): "
              << m_data_store->get_data_size()
              << "; global to this trainer: " << nn << std::endl;
  }
}

// Loads the fields that are specified in the user supplied schema
void hdf5_data_reader::load_sample(conduit::Node& node,
                                   hid_t file_handle,
                                   const std::string& sample_name,
                                   bool ignore_failure)
{
  // BVE FIXME Note that this will load an HDF5 field by field.  If the entire
  // file is used, this will have sub-optimal performance.  It should be
  // optimized to have a cut-through for when the experiment schema matches the
  // data schema load data for the field names specified in the user's
  // experiment-schema
  for (auto& [pathname, path_node] : m_useme_node_map) {
    // do not load a "packed" field, as it doesn't exist on disk!
    if (!is_composite_node(path_node)) {

      // check that the requested data (pathname) exists on disk
      const std::string original_path = "/" + sample_name + "/" + pathname;
      if (!conduit::relay::io::hdf5_has_path(file_handle, original_path)) {
        if (ignore_failure) {
          continue;
        }
        LBANN_ERROR("hdf5_has_path failed for path: ", original_path);
      }

      // get the new path-name (prepend the index)
      std::ostringstream ss2;
      ss2 << '/' << pathname;
      const std::string new_pathname(ss2.str());

      // note: this will throw an exception if the child node doesn't exist
      const conduit::Node& metadata = path_node.child(s_metadata_node_name);

      // optionally coerce the data, e.g, from double to float, per settings
      // in the experiment_schema
      if (metadata.has_child(HDF5_METADATA_KEY_COERCE)) {
        coerce(metadata, file_handle, original_path, new_pathname, node);
      }
      else {
        conduit::relay::io::hdf5_read(file_handle,
                                      original_path,
                                      node[new_pathname]);
      }

      // check to see if there are integer types left in the sample and warn the
      // user
      auto dtype = node[new_pathname].dtype();
      // https://github.com/LLNL/conduit/blob/develop/src/libs/conduit/conduit_data_type.hpp
      if (!dtype.is_floating_point()) {
        LBANN_MSG("Ingesting sample field ",
                  pathname,
                  " which has unconventional data format ",
                  dtype.to_string());
      }

      // optionally normalize
      if (metadata.has_child(HDF5_METADATA_KEY_SCALE) ||
          metadata.has_child(HDF5_METADATA_KEY_BIAS)) {
        normalize(node, new_pathname, metadata);
      }

      // for images
      // Check to make sure that the image / data volume has more than one
      // channel and is in a channel's last format.  LBANN wants data in a
      // channels first format
      if (does_hdf5_field_require_repack_to_channels_first(metadata)) {
        repack_image(node, new_pathname, metadata);
      }
    }
  }
}

void hdf5_data_reader::load_sample_from_sample_list(conduit::Node& node,
                                                    size_t index,
                                                    bool ignore_failure)
{
  auto [file_handle, sample_name] = data_reader_sample_list::open_file(index);
  const std::string padded_index = LBANN_DATA_ID_STR(index);
  load_sample(node[padded_index], file_handle, sample_name, ignore_failure);
  pack(node, index);
}

void hdf5_data_reader::normalize(conduit::Node& node,
                                 const std::string& path,
                                 const conduit::Node& metadata)
{
  void* vals = node[path].element_ptr(0);
  size_t n_elements = node[path].dtype().number_of_elements();

  // treat this as a multi-channel image
  if (metadata.has_child(HDF5_METADATA_KEY_CHANNELS)) {

    // get number of channels, with sanity checking
    int64_t n_channels = metadata[HDF5_METADATA_KEY_CHANNELS].value();
    int sanity = metadata[HDF5_METADATA_KEY_SCALE].dtype().number_of_elements();
    if (sanity != n_channels) {
      LBANN_ERROR("sanity: ",
                  sanity,
                  " should equal ",
                  n_channels,
                  " but instead is: ",
                  n_channels);
    }

    // sanity check; TODO: implement for other formats when needed
    if (n_channels > 1 && !metadata.has_child("hwc")) {
      LBANN_ERROR("we only currently know how to deal with HWC input images");
    }

    // get the scale and bias arrays
    const double* scale = metadata[HDF5_METADATA_KEY_SCALE].as_double_ptr();
    std::vector<double> b(n_channels, 0);
    const double* bias = b.data();
    if (metadata.has_child(HDF5_METADATA_KEY_BIAS)) {
      bias = metadata[HDF5_METADATA_KEY_BIAS].as_double_ptr();
    }

    // perform the normalization
    if (node[path].dtype().is_float32()) {
      float* data = reinterpret_cast<float*>(vals);
      do_normalize(data, scale, bias, n_elements, n_channels);
    }
    else if (node[path].dtype().is_float64()) {
      double* data = reinterpret_cast<double*>(vals);
      do_normalize(data, scale, bias, n_elements, n_channels);
    }
    else {
      LBANN_ERROR(
        "Only float and double are currently supported for normalization");
    }
  }

  // 1D case
  else {
    double scale = metadata[HDF5_METADATA_KEY_SCALE].value();
    double bias = 0;
    if (metadata.has_child(HDF5_METADATA_KEY_BIAS)) {
      bias = metadata[HDF5_METADATA_KEY_BIAS].value();
    }

    if (node[path].dtype().is_float32()) {
      float* data = reinterpret_cast<float*>(vals);
      do_normalize(data, scale, bias, n_elements);
    }
    else if (node[path].dtype().is_float64()) {
      double* data = reinterpret_cast<double*>(vals);
      do_normalize(data, scale, bias, n_elements);
    }
    else {
      LBANN_ERROR(
        "Only float and double are currently supported for normalization");
    }
  }
}

// recursive
void hdf5_data_reader::test_that_all_nodes_contain_metadata(conduit::Node& node)
{
  if (!node.has_child(s_metadata_node_name)) {
    LBANN_ERROR("missing metadata node for: ", node.path());
  }

  // recurse for each child
  int n_children = node.number_of_children();
  for (int j = 0; j < n_children; j++) {
    if (node.child_ptr(j)->name() != s_metadata_node_name) {
      test_that_all_nodes_contain_metadata(node.child(j));
    }
  }
}

void hdf5_data_reader::parse_schemas()
{
  adjust_metadata(&m_data_schema);
  adjust_metadata(&m_experiment_schema);
  test_that_all_nodes_contain_metadata(m_data_schema);
  test_that_all_nodes_contain_metadata(m_experiment_schema);

  // get pointers to all Nodes in the data schema (this is the user-supplied
  // schema for the data as it resides on disk). On return, m_data_map maps:
  //       node_pathname -> Node*
  m_data_map.clear();
  get_schema_ptrs(&m_data_schema, m_data_map);
  get_leaves_multi(&m_experiment_schema, m_useme_node_map_ptrs);

  // At this point, each object in "m_useme_node_map_ptrs:"
  //   1. is a leaf node whose values will be used in the experiment
  //   2. has a "metatdata" child node that contains instructions for
  //      munging the data, i.e: scale, bias, ordering, coersion, packing, etc.

  // ptrs were a problem when carving out a validation set ...
  for (const auto& t : m_useme_node_map_ptrs) {
    m_useme_node_map[t.first] = *t.second;
  }

  // sanity checks
  for (const auto& [key, value] : m_useme_node_map) {
    if (!value.has_child(s_metadata_node_name)) {
      LBANN_ERROR("missing metadata child node for: ", key);
    }
  }

  construct_linearized_size_lookup_tables();
}

// recursive
void hdf5_data_reader::get_schema_ptrs(
  conduit::Node* input,
  std::unordered_map<std::string, conduit::Node*>& schema_name_map)
{

  // add the input node to the output map
  const std::string& path_name = input->path();
  if (path_name == "") {
    if (!input->is_root()) {
      LBANN_ERROR("node.path == '', but node is not root");
    }
  }
  else {
    if (schema_name_map.find(path_name) != schema_name_map.end()) {
      LBANN_ERROR("duplicate pathname: ", path_name);
    }
    schema_name_map[path_name] = input;
  }

  // recurse for each child
  int n_children = input->number_of_children();
  for (int j = 0; j < n_children; j++) {
    if (input->child_ptr(j)->name() != s_metadata_node_name) {
      get_schema_ptrs(&input->child(j), schema_name_map);
    }
  }
}

void hdf5_data_reader::get_leaves_multi(
  conduit::Node* node_in,
  std::unordered_map<std::string, conduit::Node*>& leaves_out)
{

  std::unordered_map<std::string, conduit::Node*> experiment_leaves;
  get_leaves(node_in, experiment_leaves);
  // "experiment_leaves" contains pointers to leaf nodes from
  // "m_experiment_schema;" We next use these as starting nodes for searchs in
  // m_data_schema. (recall, m_experiment_schema is a trimmed version of
  // m_data_schema)
  for (const auto& leaf : experiment_leaves) {
    std::string path_name = leaf.first;
    if (m_data_map.find(path_name) == m_data_map.end()) {
      LBANN_ERROR("pathname: ",
                  path_name,
                  " was not found in m_data_map; num map entries: ",
                  m_data_map.size(),
                  "; role: ",
                  get_role());
    }
    conduit::Node* node_for_recursion = m_data_map[path_name];

    if (!node_for_recursion->has_child(s_metadata_node_name)) {
      LBANN_ERROR("Node with path: ",
                  node_for_recursion->path(),
                  " is missing metadata node");
    }

    const conduit::Node& metadata = leaf.second->child(s_metadata_node_name);
    std::unordered_map<std::string, conduit::Node*> final_leaves;
    get_leaves(node_for_recursion, final_leaves);
    for (auto t : final_leaves) {
      conduit::Node& end_metadata = (*t.second)[s_metadata_node_name];
      for (int j = 0; j < metadata.number_of_children(); j++) {
        const std::string& field_name = metadata.child(j).name();
        end_metadata[field_name] = metadata[field_name];
      }
      leaves_out[t.first] = t.second;
    }
  }
}

// recursive
void hdf5_data_reader::get_leaves(
  conduit::Node* node,
  std::unordered_map<std::string, conduit::Node*>& leaves_out)
{
  // end of recusion conditions: no children, or only child is "metadata"
  int n = node->number_of_children();
  if (n == 0) {
    leaves_out[node->path()] = node;
    return;
  }
  if (n == 1 && node->child(0).name() == s_metadata_node_name) {
    leaves_out[node->path()] = node;
    return;
  }

  // recursion loop
  for (int j = 0; j < node->number_of_children(); j++) {
    conduit::Node* child = node->child_ptr(j);
    if (child->name() != s_metadata_node_name) {
      get_leaves(child, leaves_out);
    }
  }
}

void hdf5_data_reader::pack(conduit::Node& node, size_t index)
{
  if (m_packing_groups.size() == 0) {
    build_packing_map(node.child(0));
  }
  for (const auto& t : m_packing_groups) {
    const std::string& group_name = t.first;
    const PackingGroup& g = t.second;
    std::string group_type = conduit::DataType::id_to_name(g.data_type);
    if (group_type == "float32") {
      pack<float>(group_name, node, index);
    }
    else if (group_type == "float64") {
      pack<double>(group_name, node, index);
    }
    else {
      LBANN_ERROR("packing is currently only implemented for float32 and "
                  "float64; your data type was: ",
                  group_type,
                  " for group_name: ",
                  group_name);
    }
  }
}

struct PackingData
{
  PackingData(std::string s, int n_elts, size_t dt, int order)
    : field_name(s), num_elts(n_elts), dtype(dt), ordering(order)
  {}
  PackingData() {}
  std::string field_name;
  int num_elts;
  size_t dtype;
  conduit::index_t ordering;
};

struct
{
  bool operator()(const PackingData& a, const PackingData& b) const
  {
    return a.ordering < b.ordering;
  }
} less_oper;

void hdf5_data_reader::build_packing_map(conduit::Node& node)
{
  std::unordered_map<std::string, std::vector<PackingData>> packing_data;
  for (const auto& nd : m_useme_node_map_ptrs) {
    const conduit::Node& metadata = (*nd.second)[s_metadata_node_name];
    if (metadata.has_child(HDF5_METADATA_KEY_PACK)) {
      const std::string& group_name =
        metadata[HDF5_METADATA_KEY_PACK].as_string();
      if (!metadata.has_child(HDF5_METADATA_KEY_ORDERING)) {
        LBANN_ERROR("metadata has 'pack' but is missing 'ordering' for: ",
                    nd.first);
      }
      conduit::int64 ordering = metadata[HDF5_METADATA_KEY_ORDERING].value();
      const std::string& field_name = nd.first;
      int n_elts = node[field_name].dtype().number_of_elements();
      size_t data_type = node[field_name].dtype().id();
      packing_data[group_name].push_back(
        PackingData(field_name, n_elts, data_type, ordering));
    }
  }

  // sort the vectors by ordering numbers
  for (auto& t : packing_data) {
    std::sort(t.second.begin(), t.second.end(), less_oper);
  }

  for (const auto& t : packing_data) {
    const std::string& group_name = t.first;
    m_packing_groups[group_name].group_name = group_name; // ACH!
    for (const auto& t2 : t.second) {
      m_packing_groups[group_name].names.push_back(t2.field_name);
      m_packing_groups[group_name].sizes.push_back(t2.num_elts);
      m_packing_groups[group_name].data_types.push_back(t2.dtype);
    }
    size_t n_elts = 0;
    conduit::index_t id_sanity = 0;
    for (size_t k = 0; k < m_packing_groups[group_name].names.size(); k++) {
      n_elts += m_packing_groups[group_name].sizes[k];
      if (id_sanity == 0) {
        id_sanity = m_packing_groups[group_name].data_types[k];
      }
      else {
        if (m_packing_groups[group_name].data_types[k] != id_sanity) {
          LBANN_ERROR(
            "m_packing_groups[group_name].data_types[k] != id_sanity; you may "
            "need to coerce a data type in your schema");
        }
      }
    }
    m_packing_groups[group_name].n_elts = n_elts;
    m_packing_groups[group_name].data_type = id_sanity;
  }
}

// Union your parent's metadata with yours; in case of a key conflict,
// the child's values prevail.
// At the conclusion of recursion, every node in the tree will
// contain a metadata child node
// (recursive)
void hdf5_data_reader::adjust_metadata(conduit::Node* node)
{
  // note: next call creates the node if it doesn't exist
  conduit::Node* metadata = node->fetch_ptr(s_metadata_node_name);
  for (int j = 0; j < metadata->number_of_children(); j++) {
    const std::string& field_name = metadata->child(j).name();
    // Check that all of the field name is a valid key
    if (!is_hdf5_metadata_key_valid(field_name)) {
      LBANN_WARNING("In conduit node ",
                    node->name(),
                    " invalid schema key detected: ",
                    field_name);
    }
  }

  if (!node->is_root()) {
    const conduit::Node& parents_metadata =
      *(node->parent()->fetch_ptr(s_metadata_node_name));
    for (int j = 0; j < parents_metadata.number_of_children(); j++) {
      const std::string& field_name = parents_metadata.child(j).name();
      if (!metadata->has_child(field_name)) {
        (*metadata)[field_name] = parents_metadata[field_name];
      }
    }
  }

  // recursion loop
  for (int j = 0; j < node->number_of_children(); j++) {
    if (node->child_ptr(j)->name() != s_metadata_node_name) {
      adjust_metadata(node->child_ptr(j));
    }
  }
}

void hdf5_data_reader::coerce(const conduit::Node& metadata,
                              hid_t file_handle,
                              const std::string& original_path,
                              const std::string& new_pathname,
                              conduit::Node& node)
{
  conduit::Node tmp;
  conduit::relay::io::hdf5_read(file_handle, original_path, tmp);

  // https://github.com/LLNL/conduit/blob/develop/src/libs/conduit/conduit_data_type.hpp
  if (!tmp.dtype().is_number()) {
    LBANN_ERROR(
      "source data in field ",
      original_path,
      " is not integer or real format; ",
      "please update the data set or experiment schema to exclude the field: ",
      tmp.dtype().to_string());
  }

  const std::string& coerce_to =
    conduit_to_string(metadata[HDF5_METADATA_KEY_COERCE]);

  // Example of data coercion taken from
  // https://github.com/LLNL/conduit/blob/develop/src/tests/conduit/t_conduit_node_to_array.cpp
  conduit::DataType::TypeID coerced_tid;
  if (coerce_to == "float") {
    coerced_tid = conduit::DataType::FLOAT32_ID;
  }
  else if (coerce_to == "double") {
    coerced_tid = conduit::DataType::FLOAT64_ID;
  }
  else if (coerce_to == "int") {
    coerced_tid = conduit::DataType::INT32_ID;
  }
  else if (coerce_to == "long" || coerce_to == "int64") {
    coerced_tid = conduit::DataType::INT64_ID;
  }
  else {
    LBANN_ERROR("Un-implemented type requested for coercion: ",
                coerce_to,
                "; you need to update the data reader to support this");
  }
  // https://github.com/LLNL/conduit/blob/develop/src/libs/conduit/conduit_node.hpp#L3263
  tmp.to_data_type(coerced_tid, node[new_pathname]);
}

void hdf5_data_reader::repack_image(conduit::Node& node,
                                    const std::string& path,
                                    const conduit::Node& metadata)
{

  // ==== start: sanity checking
  if (!metadata.has_child(HDF5_METADATA_KEY_CHANNELS)) {
    LBANN_WARNING("repack_image called, but metadata is missing the 'channels' "
                  "field; please check your schemas");
    return;
  }
  if (!is_hdf5_field_channels_last(metadata)) {
    LBANN_ERROR(
      "repack_image is only designed to deal with channels last input images");
  }
  if (!metadata.has_child(HDF5_METADATA_KEY_DIMS)) {
    LBANN_ERROR("your metadata is missing 'dims' for an image");
  }
  // ==== end: sanity checking

  void* vals = node[path].element_ptr(0);
  size_t n_elts = node[path].dtype().number_of_elements();
  int64_t n_channels = metadata[HDF5_METADATA_KEY_CHANNELS].value();
  const conduit::int64* dims = metadata[HDF5_METADATA_KEY_DIMS].as_int64_ptr();
  const int num_dims =
    metadata[HDF5_METADATA_KEY_DIMS].dtype().number_of_elements();

  if (num_dims == 2) {
    const int row_dim = dims[0];
    const int col_dim = dims[1];

    if (node[path].dtype().is_float32()) {
      float* data = reinterpret_cast<float*>(vals);
      do_repack_tensor_HWC_to_CHW(data, n_elts, row_dim, col_dim, n_channels);
    }
    else if (node[path].dtype().is_float64()) {
      double* data = reinterpret_cast<double*>(vals);
      do_repack_tensor_HWC_to_CHW(data, n_elts, row_dim, col_dim, n_channels);
    }
    else {
      LBANN_ERROR(
        "Only float and double are currently supported for repacking");
    }
  }
  else if (num_dims == 3) {
    const int depth_dim = dims[0];
    const int height_dim = dims[1];
    const int width_dim = dims[2];

    if (node[path].dtype().is_float32()) {
      float* data = reinterpret_cast<float*>(vals);
      do_repack_tensor_DHWC_to_CDHW(data,
                                    n_elts,
                                    depth_dim,
                                    height_dim,
                                    width_dim,
                                    n_channels);
    }
    else if (node[path].dtype().is_float64()) {
      double* data = reinterpret_cast<double*>(vals);
      do_repack_tensor_DHWC_to_CDHW(data,
                                    n_elts,
                                    depth_dim,
                                    height_dim,
                                    width_dim,
                                    n_channels);
    }
    else {
      LBANN_ERROR(
        "Only float and double are currently supported for repacking");
    }
  }
  else {
    LBANN_ERROR("Only 2D and 3D tensors are supported for repacking");
  }
}

const std::vector<El::Int>
hdf5_data_reader::get_data_dims(std::string name) const
{
  std::unordered_map<std::string, std::vector<El::Int>>::const_iterator iter =
    m_data_dims_lookup_table.find(name);
  if (iter == m_data_dims_lookup_table.end()) {
    LBANN_ERROR(
      "get_data_dims_size was asked for info about an unknown field name: ",
      name);
  }
  return iter->second;
}

int hdf5_data_reader::get_linearized_size(
  data_field_type const& data_field) const
{
  if (m_linearized_size_lookup_table.size() == 0) {
    LBANN_ERROR("get_linearized_size was called with an empty lookup table");
  }
  std::unordered_map<std::string, int>::const_iterator iter =
    m_linearized_size_lookup_table.find(data_field);
  if (iter == m_linearized_size_lookup_table.end()) {
    LBANN_ERROR("get_linearized_size was asked for info about an unknown "
                "data field: ",
                data_field,
                "; table size: ",
                m_linearized_size_lookup_table.size(),
                " for role: ",
                get_role());
  }
  return iter->second;
}

// fills in: m_data_dims_lookup_table and m_linearized_size_lookup_table;
void hdf5_data_reader::construct_linearized_size_lookup_tables()
{
  // If there are no loaded samples bail out
  if (m_shuffled_indices.size() == 0) {
    return;
  }
  m_linearized_size_lookup_table.clear();
  m_data_dims_lookup_table.clear();

  conduit::Node node;
  size_t index = random() % m_shuffled_indices.size();

  // must load a sample to get data sizes. Alternatively, this metadata
  // could be included in the schemas
  load_sample_from_sample_list(node, index);

  return construct_linearized_size_lookup_tables(node);
}

void hdf5_data_reader::construct_linearized_size_lookup_tables(
  conduit::Node& node)
{
  std::unordered_map<std::string, conduit::Node*> leaves;
  get_leaves(&node, leaves);

  std::ostringstream ss;
  ss << '/' << LBANN_DATA_ID_STR(0);
  size_t const sz = ss.str().size();

  // loop over the data-fields (aka, leaves)
  for (const auto& t : leaves) {
    const std::string field_name = t.first.substr(sz);
    // we only care about leaves that are being used in the current experiment
    if (m_useme_node_map_ptrs.find(field_name) != m_useme_node_map_ptrs.end()) {

      // add entry to linearized size lookup table
      size_t n_elts = t.second->dtype().number_of_elements();
      m_linearized_size_lookup_table[field_name] = n_elts;

      // remainder of this block is filling in the m_data_dims_lookup_table
      conduit::Node* nd = m_useme_node_map_ptrs[field_name];
      conduit::Node* metadata = nd->fetch_ptr(s_metadata_node_name);

      // easy case: scalars, vectors
      if (!metadata->has_child(HDF5_METADATA_KEY_CHANNELS)) {
        m_data_dims_lookup_table[field_name].push_back(n_elts);
      }

      // error prone case; depends on user correctly writing schema
      // data dims for JAG images are: {4, 64, 64}; they may have previously
      // been {64, 64}; this could be a problem
      else {
        int channels = metadata->child(HDF5_METADATA_KEY_CHANNELS).to_int32();
        m_data_dims_lookup_table[field_name].push_back(channels);
        int nn_elts =
          metadata->child(HDF5_METADATA_KEY_DIMS).dtype().number_of_elements();
        const conduit::int64* tmp =
          metadata->child(HDF5_METADATA_KEY_DIMS).as_int64_ptr();
        for (int k = 0; k < nn_elts; k++) {
          m_data_dims_lookup_table[field_name].push_back(tmp[k]);
        }
      }
    }
  }
}

bool hdf5_data_reader::fetch_conduit_node(conduit::Node& sample,
                                          uint64_t data_id)
{
  // get the pathname to the data, and verify it exists in the conduit::Node
  const conduit::Node& node = get_data_store().get_conduit_node(data_id);
  sample = node;
  return true;
}

void hdf5_data_reader::print_metadata(std::ostream& os)
{
  os << std::endl
     << "Metadata and data-type information for all loaded fields follows, for "
        "role: "
     << get_role() << std::endl;

  std::unordered_map<std::string, conduit::Node*> leaves;
  std::unordered_map<std::string, conduit::Node*> mp;
  // load a sample from file, applying all transformations along the way;
  // need to do this so we can get the correct dtypes
  conduit::Node populated_node;
  if (m_shuffled_indices.size() != 0) {
    size_t index = random() % m_shuffled_indices.size();
    bool ignore_failure = true;
    load_sample_from_sample_list(populated_node, index, ignore_failure);

    // get all leaves (data fields)
    get_leaves(&populated_node, leaves);

    // build map: field_name -> Node
    for (const auto& t : leaves) {
      size_t j = t.first.find('/');
      mp[t.first.substr(j + 1)] = t.second;
    }
  }
  // print metadata and data types for all other nodes
  for (const auto& t : m_useme_node_map_ptrs) {
    const std::string& name = t.first;
    conduit::Node* node = t.second;
    os << "==================================================================\n"
       << "field: " << t.first << std::endl;
    if (!node->has_child(s_metadata_node_name)) {
      LBANN_ERROR("missing metadata node for ", name);
    }
    conduit::Node& metadata = node->fetch_existing(s_metadata_node_name);
    std::string yaml_metadata = metadata.to_yaml();
    os << "\nMetadata Info:\n--------------" << yaml_metadata;

    if (mp.find(name) != mp.end()) {
      const conduit::Node* nd = mp[name];
      std::string yaml_dtype = nd->dtype().to_yaml();
      os << "\nDataType Info:\n--------------\n" << yaml_dtype;
    }
    else {
      os << "\nThis field has been packed, per 'pack:' entry (above)\n";
    }
    os << std::endl;
  }
  os
    << "==================================================================\n\n";
}

bool hdf5_data_reader::is_composite_node(const conduit::Node& node) const
{
  if (!node.has_child(s_metadata_node_name)) {
    LBANN_ERROR("node with path: ",
                node.path(),
                " is missing metadata child node");
  }
  const conduit::Node& metadata = node.child(s_metadata_node_name);
  return metadata.has_child(s_composite_node);
}

void hdf5_data_reader::set_data_schema(const conduit::Node& s)
{
  m_data_schema = s;
  parse_schemas();
}

void hdf5_data_reader::set_experiment_schema(const conduit::Node& s)
{
  m_experiment_schema = s;
  parse_schemas();
}

bool is_hdf5_metadata_key_valid(std::string const& key)
{
  return hdf5_metadata_valid_keys.count(key);
}

bool is_hdf5_field_channels_last(conduit::Node const& metadata)
{
  // If the field has a single channel then there is no diference between
  // channels last and first
  if ((metadata.has_child(HDF5_METADATA_KEY_CHANNELS) &&
       metadata[HDF5_METADATA_KEY_CHANNELS].as_int64() == 1)) {
    return false;
  }
  if (metadata.has_child(HDF5_METADATA_KEY_LAYOUT)) {
    if (conduit_to_string(metadata[HDF5_METADATA_KEY_LAYOUT]) ==
        HDF5_METADATA_VALUE_LAYOUT_HWC) {
      return true;
    }
    if (conduit_to_string(metadata[HDF5_METADATA_KEY_LAYOUT]) ==
        HDF5_METADATA_VALUE_LAYOUT_DHWC) {
      return true;
    }
  }
  return false;
}

bool does_hdf5_field_require_repack_to_channels_first(
  conduit::Node const& metadata)
{
  if (is_hdf5_field_channels_last(metadata)) {
    if (metadata.has_child(HDF5_METADATA_KEY_TRANSPOSE)) {
      if (conduit_to_string(metadata[HDF5_METADATA_KEY_TRANSPOSE]) ==
          HDF5_METADATA_VALUE_LAYOUT_CHW) {
        return true;
      }
      if (conduit_to_string(metadata[HDF5_METADATA_KEY_TRANSPOSE]) ==
          HDF5_METADATA_VALUE_LAYOUT_CDHW) {
        return true;
      }
    }
  }
  return false;
}

std::string conduit_to_string(conduit::Node const& field)
{
  std::string str = field.to_string();
  str.erase(std::remove(str.begin(), str.end(), '\"'), str.end());
  return str;
}

} // namespace lbann
