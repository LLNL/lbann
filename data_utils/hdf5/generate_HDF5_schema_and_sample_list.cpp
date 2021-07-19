#include "conduit/conduit.hpp"
#include "conduit/conduit_schema.hpp"
#include "conduit/conduit_relay.hpp"

#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>

using namespace std;

const std::string sample_list_inclusive_fn("inclusive.sample_list");
const std::string sample_list_exclusive_fn("exclusive.sample_list");
const std::string yaml_fn("data_schema.yaml");

const conduit::Node*  get_sample(const vector<string>& tokens, const conduit::Node& nd) {
  const conduit::Node* node = &nd;
  size_t count = 0;
  for (size_t j=0; j<tokens.size(); j++) {
    if (node->has_child(tokens[j])) {
      ++count;
      node = &(node->child(tokens[j]));
    }
  }
  if (count == tokens.size()) {
    return node;
  }

  //recursion
  conduit::NodeConstIterator iter = nd.children();
  while (iter.has_next()) {
    const conduit::Node& child = iter.next();
    return get_sample(tokens, child);
  }
  return nullptr;
}
void get_sample_ids(const string& schema_str, const conduit::Node& nd, vector<string>& sample_ids) {
  const string candidate = nd.schema().to_string();
  if (candidate == schema_str) {
    sample_ids.push_back(nd.path());
  }
  //recursion
  conduit::NodeConstIterator iter = nd.children();
  while (iter.has_next()) {
    const conduit::Node& child = iter.next();
    get_sample_ids(schema_str, child, sample_ids);
  }
}

void get_all_sample_ids(const string& filelist, const string& schema_str, map<string, vector<string>>& sample_ids) {
  cout << "getting list of sample IDs for all files\n";
  ifstream in(filelist);
  string filename;
  while (in >> filename) {
    if (filename.size() < 2) {
      continue;
    }
    conduit::Node nd;
    cout << "  loading: " << filename << endl;
    conduit::relay::io::load(filename.c_str(), "hdf5", nd);
    get_sample_ids(schema_str, nd, sample_ids[filename]);
  }
}

void print_leading_spaces(ofstream& out, int n) {
  for (int k=0; k<n; k++) {
    out << " ";
  }
}

bool is_leaf(const conduit::Node& nd) {
  return !nd.number_of_children();
}

void write_yaml_header(ofstream& out) {
  out <<
  "#You must edit this file prior to use as follows.\n"
  "#\n"
  "#  Uncomment the 'pack' entries for whichever data fields you wish to use.\n"
  "#  pack's value should be one of the following: datum, label, response.\n"
  "#  If you do not wish to use a data field, simply leave 'pack' commented out\n"
  "#\n"
  "#  For normalization, uncomment and add appropriate values to the 'scale' \n"
  "#  and 'bias' entries.  For images, the 'scale' and 'bias' entries should be\n"
  "#  lists with one entry per channel. You should also add the following \n"
  "#  entries for images: dims, channels. Here is an example of the metadata\n"
  "#  entries for images:\n"
  "#    dims: [300,300]\n"
  "#    channels: 2\n"
  "#    scale: [1.23, 4.56]\n"
  "#    bias:  [1.0, 2.0]\n"
  "#\n"
  "#  The 'ordering' entries determine how data is packed into a vector \n"
  "#  e.g, when you have multiple 'pack: response' entries. The values in the \n"
  "#  ordering entries need not be unique and are relative\n"
  "#\n";
}

void write_yaml_file(ofstream& out, const conduit::Node* nd, int indent=0, int ordering=10) {
  if (indent == 0) {
    cout << ">>>>>>>>>> WRITING " << yaml_fn << endl;
    write_yaml_header(out);
  }
  conduit::NodeConstIterator iter = nd->children();
  while (iter.has_next()) {
    const conduit::Node& child = iter.next();
    int n = indent;
    print_leading_spaces(out, n);
    out << child.name() << ":\n";
    print_leading_spaces(out, n+2);
    out << "metadata:\n";
if (child.name() == "1685.0") {
  cout << child.path()<< " n children: " << child.number_of_children() << " is leaf? " << is_leaf(child) << endl;
}
    if (is_leaf(child)) {
      print_leading_spaces(out, n+4);
      out << "ordering: " << ordering << endl;
      print_leading_spaces(out, n+4);
      out << "#pack: datum\n";
      print_leading_spaces(out, n+4);
      out << "#scale:\n";
      print_leading_spaces(out, n+4);
      out << "#bias:\n";
      ordering += 10;
    }
    write_yaml_file(out, &child, indent+2, ordering);
  }
}

void write_exclusive_sample_list(ofstream& out, const map<string,vector<string>>& sample_ids, const string& base_dir) {
  cout << ">>>>>>>>>> WRITING exclustive sample list\n";
  out << "CONDUIT_HDF5_EXCLUSION\n";
  size_t good = 0;
  for (const auto &t : sample_ids) {
    good += t.second.size();
  }
  out << good << " 0 " << sample_ids.size() << endl;
  out << base_dir << endl;
  int idx = base_dir.size();
  if (base_dir ==  ".") { //edge case
    idx = 0;
  }
  for (const auto& t : sample_ids) {
    out << t.first.substr(idx) << " " << t.second.size() << " 0\n";
  }
}

void write_inclusive_sample_list(ofstream& out, const map<string,vector<string>>& sample_ids, const string& base_dir) {
  cout << ">>>>>>>>>> WRITING inclustive sample list\n";
  out << "CONDUIT_HDF5_INCLUSION\n";
  size_t good = 0;
  for (const auto &t : sample_ids) {
    good += t.second.size();
  }
  out << good << " 0 " << sample_ids.size() << endl;
  out << base_dir << endl;
  int idx = base_dir.size();
  if (base_dir ==  ".") { //edge case
    idx = 0;
  }
  for (const auto& t : sample_ids) {
    out << t.first.substr(idx) << " " << t.second.size() << " 0";
    for (const auto& sample_id : t.second) {
      out << " " << sample_id;
    }
    out << endl;
  }
}

void open_file(string filename, ofstream& out) {
  out.open(filename);
  if (!out) {
    cerr << "failed to open " << filename << " for WRITING\n";
    exit(9);
  }
}

string get_base_dir(const string& filelist) {
  //this is very inefficient; there are many better approaches
  //for finding longest common prefix; but this is easy to code.

  //get vector containing hdf5 filenames
  std::vector<string> f;
  ifstream in(filelist);
  string filename;
  while (in >> filename) {
    if (filename.size() > 2) {
      f.push_back(filename);
    }
  }
  if (f.size() < 1) {
    cerr << "get_base_dir; possibly empty filelist\n";
    exit(0);
  }

  //edge case
  if (f.size() == 1) {
    size_t jj = f[0].rfind('/');
    if (jj == string::npos) {
cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. string::npos\n";
      return ".";
    } else {
cout << "NOT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. string::npos\n";
      return f[0].substr(0, jj);
    }
  }

  //find longest common prefix, which is the base directory
  string b;
  int index = 0;
  while (true) {
    size_t j1 = f[0].find('/', index+1);
    if (j1 == string::npos) {
      break;
    }
    for (size_t i=1UL; i<f.size(); i++) {
      size_t j2 = f[i].find('/', index+1);
      if (j2 != j1) {
        break;
      }
    }
    index = j1;
  }
  b = f[0].substr(0, index);

  return b;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: " << argv[0] << " filelist sample_id\n"
          "\n"
          "where: 'filelist' contains the names of one or more hdf5 files;\n"
          "\n"
          "example invocations:\n"
          "  generate_yaml filelist_PROBIES.txt RUN_ID/000000000\n"
          "  generate_yaml filelist_carbon.txt e1/s100\n"
          "  generate_yaml filelist_jag.txt 0.0.96.7.0:1\n"
          "\n";
    exit(9);
  }
  string filelist(argv[1]);
  string sample_id(argv[2]);

  //get base directory, which is the longest common prefix
  string base_dir = get_base_dir(filelist);

  //open output files
  ofstream out_sample_list_inclusive;
  ofstream out_sample_list_exclusive;
  ofstream out_yaml;
  open_file(sample_list_inclusive_fn, out_sample_list_inclusive);
  open_file(sample_list_exclusive_fn, out_sample_list_exclusive);
  open_file(yaml_fn, out_yaml);

  //load the first hdf5 file from the filelist
  ifstream in(filelist.c_str());
  if (!in) {
    cerr << "failed to open " << filelist << " for reading\n";
    exit(9);
  }
  string filename;
  in >> filename;
  in.close();
  conduit::Node nd;
  cout << "loading: " << filename << endl;
  conduit::relay::io::load(filename, "hdf5", nd);

  //tokenize the sample id: split on the '/' character
  std::replace(sample_id.begin(), sample_id.end(), '/', ' ');
  istringstream iss(sample_id);
  string w;
  vector<string> tokens;
  while (iss >> w) {
    tokens.push_back(w);
  }

  //get a node whose children compose a sample, then write the yaml file
  const conduit::Node* sample_node = get_sample(tokens, nd);
  if (sample_node == nullptr) {
    cerr << "failed to find your specified sample_id: " << sample_id << endl;
    for (long long j=0L; j<nd.number_of_children(); j++) {
      cout << nd.child(j).path() << endl; //XX
    }
    exit(9);
  }
  write_yaml_file(out_yaml, sample_node);

  //get pathnames for all sample IDs
  map<string, vector<string>> sample_ids; //maps: filename -> list of sample IDs
  const string schema_str = sample_node->schema().to_string();
  get_all_sample_ids(filelist, schema_str, sample_ids);

  //write the sample lists
  cout << "base dir: " << base_dir << endl;
  write_exclusive_sample_list(out_sample_list_exclusive, sample_ids, base_dir);
  write_inclusive_sample_list(out_sample_list_inclusive, sample_ids, base_dir);

  //close output files
  out_yaml.close();
  out_sample_list_exclusive.close();
  out_sample_list_inclusive.close();
}
