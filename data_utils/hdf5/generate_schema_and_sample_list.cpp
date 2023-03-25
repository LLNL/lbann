////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

// Author: David Hysom (hysom1@llnl.gov)
// Issues: https://github.com/LLNL/lbann/issues/new

#include "helpers.hpp"

#include <clara.hpp>

#include <conduit/conduit.hpp>
#include <conduit/conduit_relay.hpp>
#include <conduit/conduit_relay_io.hpp>
#include <conduit/conduit_relay_io_hdf5.hpp>
#include <conduit/conduit_relay_io_identify_protocol_api.hpp>
#include <conduit/conduit_schema.hpp>
#include <conduit/conduit_utils.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using conduit::Node;
using namespace std;

string const sample_list_inclusive_fn("inclusive.sample_list");
string const sample_list_exclusive_fn("exclusive.sample_list");
string const yaml_fn("data_schema.yaml");

Node build_node_from_file(string const& filename, string const& protocol = "")
{
  conduit::Node node;
  if (protocol.empty())
    conduit::relay::io::load(filename, node);
  else
    conduit::relay::io::load(filename, protocol, node);
  return node;
}

map<string, vector<string>> get_all_sample_ids(conduit::Schema const& schema,
                                               vector<string> const& filelist,
                                               std::string const& protocol)
{
  map<string, vector<string>> sample_ids;
  for (auto const& filename : filelist) {
    if (filename.size() < 2) {
      continue;
    }
    cout << "  loading: " << filename << endl;
    Node nd = build_node_from_file(filename, protocol);
    sample_ids[filename] = data_utils::get_matching_node_paths(nd, schema);
  }
  return sample_ids;
}

void print_leading_spaces(ostream& out, size_t n) { out << string(n, ' '); }

bool is_leaf(Node const& nd) noexcept { return nd.number_of_children() == 0L; }

void write_yaml_header(ostream& out)
{
  out << R"YAML(#You must edit this file prior to use as follows.
#
#  Uncomment the 'pack' entries for whichever data fields you wish to use.
#  pack's value should be one of the following: datum, label, response.
#  If you do not wish to use a data field, simply leave 'pack' commented out
#
#  For normalization, uncomment and add appropriate values to the 'scale'
#  and 'bias' entries.  For images, the 'scale' and 'bias' entries should be
#  lists with one entry per channel. You should also add the following
#  entries for images: dims, channels. Here is an example of the metadata
#  entries for images:
#    dims: [300,300]
#    channels: 2
#    scale: [1.23, 4.56]
#    bias:  [1.0, 2.0]
#
#  The 'ordering' entries determine how data is packed into a vector
#  e.g, when you have multiple 'pack: response' entries. The values in the
#  ordering entries need not be unique and are relative
#
)YAML";
}

void write_yaml_file(ostream& out,
                     Node const& nd,
                     int indent = 0,
                     int ordering = 10)
{
  if (indent == 0) {
    write_yaml_header(out);
  }
  conduit::NodeConstIterator iter = nd.children();
  while (iter.has_next()) {
    Node const& child = iter.next();
    int n = indent;
    print_leading_spaces(out, n);
    out << child.name() << ":\n";
    print_leading_spaces(out, n + 2);
    out << "metadata:\n";
    if (child.name() == "1685.0") {
      cout << child.path() << " n children: " << child.number_of_children()
           << " is leaf? " << is_leaf(child) << endl;
    }
    if (is_leaf(child)) {
      print_leading_spaces(out, n + 4);
      out << "ordering: " << ordering << endl;
      print_leading_spaces(out, n + 4);
      out << "#pack: datum\n";
      print_leading_spaces(out, n + 4);
      out << "#scale:\n";
      print_leading_spaces(out, n + 4);
      out << "#bias:\n";
      ordering += 10;
    }
    write_yaml_file(out, child, indent + 2, ordering);
  }
}

void write_exclusive_sample_list(ostream& out,
                                 map<string, vector<string>> const& sample_ids,
                                 string const& base_dir)
{
  out << "CONDUIT_HDF5_EXCLUSION\n";
  size_t good = 0;
  for (const auto& t : sample_ids) {
    good += t.second.size();
  }
  out << good << " 0 " << sample_ids.size() << endl;
  out << base_dir << endl;
  int idx = base_dir.size();
  if (base_dir == ".") { // edge case
    idx = 0;
  }
  for (const auto& [filename, samples] : sample_ids) {
    out << filename.substr(idx) << " " << samples.size() << " 0\n";
  }
}

void write_inclusive_sample_list(ostream& out,
                                 map<string, vector<string>> const& sample_ids,
                                 string const& base_dir)
{
  out << "CONDUIT_HDF5_INCLUSION\n";
  size_t good = 0;
  for (const auto& t : sample_ids) {
    good += t.second.size();
  }
  out << good << " 0 " << sample_ids.size() << endl;
  out << base_dir << endl;
  int idx = base_dir.size();
  if (base_dir == ".") { // edge case
    idx = 0;
  }
  for (const auto& [filename, samples] : sample_ids) {
    out << filename.substr(idx) << " " << samples.size() << " 0";
    for (const auto& sample_id : samples) {
      out << " " << sample_id;
    }
    out << endl;
  }
}

// get vector containing hdf5 filenames
vector<string> get_file_names(std::string const& file_list_file_name)
{
  vector<string> file_list;
  ifstream in(file_list_file_name);
  string file_name;
  file_name.reserve(1024);
  while (in >> file_name) {
    if (file_name.size() > 2) {
      file_list.emplace_back(data_utils::normalize_path(file_name));
    }
  }
  return file_list;
}

int main(int argc, char** argv)
{
  string filelist;
  string sample_id;
  string protocol;
  bool print_help = false;

  auto cli =
    clara::Help(print_help).optional() |
    clara::Arg(filelist, "data file list")(
      "The name of a file containing the names of one or more HDF5 files")
      .required() |
    clara::Arg(sample_id, "sample id")(
      "The path to a prototypical sample in the first HDF5 file")
      .required() |
    clara::Opt(protocol, "data protocol")["-p"]["--protocol"](
      "The conduit-compatible protocol string for these files");

  auto result = cli.parse({argc, argv});
  if (!result) {
    cerr << "Error: Parsing arguments failed with message: "
         << result.errorMessage() << endl;
    return EXIT_FAILURE;
  }

  if (print_help || sample_id.empty() || filelist.empty()) {
    auto const& exe_name = cli.m_exeName.name();
    cout << cli << endl
         << "example invocations:\n"
         << "  " << exe_name << " filelist_PROBIES.txt RUN_ID/000000000\n"
         << "  " << exe_name << " filelist_carbon.txt e1/s100\n"
         << "  " << exe_name << " filelist_jag.txt 0.0.96.7.0:1\n"
         << endl;
    return EXIT_FAILURE;
  }

  // Get the list of sample files
  auto const file_names = get_file_names(filelist);

  // Since every file is expected to be the same format, and the list
  // of files may be long, we do this once up front.
  if (protocol.empty())
    conduit::relay::io::identify_protocol(file_names.front(), protocol);

  // Get base directory, which is the longest common prefix
  string const base_dir = data_utils::get_longest_common_prefix(file_names);
  cout << "Common base dir: " << base_dir << endl;

  // open output files

  // load the first hdf5 file from the filelist
  try {
    string const& filename = file_names.front();

    cout << "Searching for sample ID \"" << sample_id
         << "\" in file: " << filename << endl;

    Node node = build_node_from_file(filename, protocol);
    auto const& prototype_sample =
      data_utils::get_prototype_sample(node, sample_id);

    {
      cout << "Writing yaml file (\"" << yaml_fn << "\")... ";
      ofstream out_yaml(yaml_fn);
      write_yaml_file(out_yaml, prototype_sample);
      cout << "done." << endl;
    }

    // get pathnames for all sample IDs
    auto sample_ids =
      get_all_sample_ids(prototype_sample.schema(), file_names, protocol);

    // write the sample lists
    {
      cout << "Writing exclusive sample list (\"" << sample_list_exclusive_fn
           << "\")... ";
      ofstream out_sample_list_exclusive(sample_list_exclusive_fn);

      write_exclusive_sample_list(out_sample_list_exclusive,
                                  sample_ids,
                                  base_dir);
      cout << "done." << endl;
    }
    {
      cout << "Writing inclusive sample list (\"" << sample_list_inclusive_fn
           << "\")... ";
      ofstream out_sample_list_inclusive(sample_list_inclusive_fn);
      write_inclusive_sample_list(out_sample_list_inclusive,
                                  sample_ids,
                                  base_dir);
      cout << "done." << endl;
    }
  }
  catch (conduit::Error const& e) {
    cerr << "Error detected by Conduit:\n" << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
