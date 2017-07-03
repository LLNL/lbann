#include "mpi.h"
#include "lbann/utils/options.hpp"
#include <ctime>
#include <dirent.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <string.h>
#include <cstdlib>
#include <stdexcept>

options * options::s_instance = new options;

//============================================================================

void parse_opt(std::string tmp, std::string &key, std::string &val)
{
  key = "";
  val = "";
  if (tmp.size() > 4 and tmp.substr(0,4) == "# --") {
    tmp = tmp.substr(2);
  }
  if (tmp.size() > 2 and tmp[0] == '-' and tmp[1] == '-') {
    size_t n = tmp.find('=');
    if (n == std::string::npos) {
      key = tmp.substr(2);
      val = "1";
    } else {
      key = tmp.substr(2, n-2);
      val = tmp.substr(n+1, tmp.size()-n+1);
      if (!val.size()) {
        val = "1";
      }
    }
  }
}

void options::init(int argc, char **argv)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

  //default fileBaseName for saving
  m_opts["saveme"] = "data.prototext";

  //save cmd line
  std::string key;
  std::string value;
  std::string loadme;
  for (int j=0; j<argc; j++) {
    m_cmd_line.push_back(argv[j]);
    parse_opt(argv[j], key, value);
    if (key == "loadme") {
      loadme = value;
    }
  }

  //optionally init from file
  if (loadme != "") {
    parse_file(loadme.c_str());
  }

  //over-ride from cmd line
  parse_cmd_line(argc, argv);

  //write output file, which contains all settings,
  //along with the prototext files
  write();

  if (!m_rank) {
    std::cout << std::endl
              << "running with the following options (if any):\n";
    for (std::map<std::string, std::string>::const_iterator t = m_opts.begin(); t != m_opts.end(); t++) {
      std::cout << "  --" << t->first << "=" << t->second << std::endl;
    }
    std::cout << std::endl;
  }
}

//============================================================================

void lower(std::string &s)
{
  for (size_t j=0; j<s.size(); j++) {
    if (isalpha(s[j])) {
      s[j] = tolower(s[j]);
    }
  }
}

void lower(char *s)
{
  size_t len = strlen(s);
  for (size_t j=0; j<len; j++) {
    if (isalpha(s[j])) {
      char a = tolower(s[j]);
      s[j] = a;
    }
  }
}

//============================================================================

bool options::has_opt(std::string option)
{
  if (m_opts.find(option) == m_opts.end()) {
    return false;
  }
  return true;
}
bool options::get_bool(const char *option)
{
  int result;
  std::string opt(option);
  if (not get_int(opt, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::options::get_int() - failed to find option: " << option
        << ", or to convert to int";
    throw std::runtime_error(err.str());
  }
  if (result == 0) return false;
  return true;
}

int options::get_int(const char *option)
{
  int result;
  std::string opt(option);
  if (not get_int(opt, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_int() - failed to find option: " << option
        << ", or to convert to int";
    throw std::runtime_error(err.str());
  }
  return result;
}

double options::get_float(const char *option)
{
  std::string opt(option);
  double result;
  if (not get_float(opt, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_float() - failed to find option: " << option
        << ", or to convert the value to double";
    throw std::runtime_error(err.str());
  }
  return result;
}

std::string options::get_string(const char *option)
{
  std::string opt(option);
  std::string result;
  if (not get_string(opt, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_string() - failed to find option: " << option;
    throw std::runtime_error(err.str());
  }
  return result;
}

//============================================================================

bool options::get_int(std::string option, int &out)
{
  if (not has_opt(option)) {
    return false;
  }
  bool is_good = true;
  std::string val = m_opts[option];
  for (size_t j=0; j<val.size(); j++) {
    if (not isdigit(val[j])) {
      is_good = false;
      break;
    }
  }
  if (not is_good) {
    return false;
  }

  std::stringstream s(val);
  if (! (s>>out)) {
    return false;
  }
  return true;
}

bool options::get_float(std::string option, double &out)
{
  if (not has_opt(option)) return false;
  std::string val(m_opts[option]);
  lower(val);
  for (size_t j=0; j<val.size(); j++) {
    if (not (isdigit(val[j]) or val[j] == '-'
             or tolower(val[j] == 'e') or val[j] == '.')) {
      return false;
    }
  }
  std::stringstream s(val);
  if (! (s>>out)) {
    return false;
  }
  return true;
}

bool options::get_string(std::string option, std::string &out)
{
  if (not has_opt(option)) return false;
  out = m_opts[option];
  return true;
}

bool options::has_int(const char *option)
{
  std::string opt(option);
  int test;
  if (get_int(opt, test)) return true;
  return false;
}

bool options::has_bool(const char *option)
{
  std::string opt(option);
  int test;
  if (get_int(opt, test)) return true;
  return false;
}

bool options::has_string(const char *option)
{
  std::string opt(option);
  std::string test;
  if (get_string(opt, test)) return true;
  return false;
}

bool options::has_float(const char *option)
{
  std::string opt(option);
  double test;
  if (get_float(opt, test)) return true;
  return false;
}

//====================================================================

void options::parse_file(const char *fn)
{
  std::ifstream in(fn);
  if (not in.is_open()) {
    if (!m_rank) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: failed to open file for reading: " << fn;
      throw std::runtime_error(err.str());
    }
  }

  std::string key, line, val;
  while (in.good() and not in.eof()) {
    getline(in, line);

    //skip of commented out portion of the file
    if (line.size() > 0 and line[0] == '>') {
      while (in.good() and not in.eof()) {
        getline(in, line);
        if (line.size() > 0 and line[0] == '<') {
          break;
        }
      }
    }

    parse_opt(line, key, val);
    if (key != "") {
      m_opts[key] = val;
    }
  }
  in.close();
}


void options::parse_cmd_line(int argc, char **argv)
{
  std::string key, val;
  for (int j=1; j<argc; j++) {
    parse_opt(argv[j], key, val);
    if (key != "") {
      m_opts[key] = val;
    }
  }
}

void copy_file(std::string fn, std::ofstream &out) {
  std::ifstream in(fn.c_str());
  if (not in.is_open()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for reading: " << fn;
    throw std::runtime_error(err.str());
  }
  std::stringstream s;
  s << in.rdbuf();
  out << s.str();
}


void options::write()
{
  if (m_rank) {
    return;
  }

  //sanity: ensure we do not over-write a previously saved data file
  std::string base = ".";
  std::string name = get_string("saveme");
  size_t i = name.find("/");
  if (i != std::string::npos) {
    base = name.substr(0, i);
  }
  base += "/";

  DIR *dir;
  struct dirent *ent;
  while (true) {
  if ((dir = opendir(base.c_str())) != NULL) {
    bool name_exists = false;
    while ((ent = readdir(dir))) {
      std::string testme(ent->d_name);
      if (testme == name) {
        name_exists = true;
      }
    }
    if (not name_exists) {
      break;
    } else {
      //@todo perhaps this could be done better
      name += "_1";
    }
  }
  }
  m_opts["saveme"] = name;


    if (not has_string("do_not_save")) {
      //open output file
      char b2[1024];
      sprintf(b2, "%s", get_string("saveme").c_str());
      std::ofstream out(b2);
      if (not out.is_open()) {
        if (!m_rank) {
          std::stringstream err;
          err << __FILE__ << " " << __LINE__
              << " :: failed to open file for writing: " << b2;
          throw std::runtime_error(err.str());
        }
      }
      std::cout << std::endl << "writing options and prototext to file: " << name << "\n\n";

      //output all data
      out << "# cmd line for original experiment:\n#  $ ";
      for (size_t h=0; h<m_cmd_line.size(); h++) {
        out << m_cmd_line[h] << " ";
      }

      int size;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      std::time_t r = std::time(nullptr);
      char *tm = std::ctime(&r);
      size_t fixme = strlen(tm);
      tm[fixme-1] = 0;
      out << "\n#\n# Experiment conducted at: " 
          <<  tm
          << "\n#\n#\n# Experiment was run with lbann version: "
          << m_lbann_version << "\n#\n#\n# To rerun the experiment: \n"
           "#  $ srun -n" << size << " " << m_cmd_line[0]
          << " --loadme=" << get_string("saveme") << "\n#\n#\n"
          << "# If you rerun the experiment, the following options will be parsed\n"
          << "# from this file"
          << " (you do not need to specify them on the cmd line).\n"
          << "# Note, however, that lines that begin with \"### --\" will be ignored\n#\n"
          << "\n#";
      for (std::map<std::string, std::string>::iterator t = m_opts.begin(); t != m_opts.end(); t++) {
        std::string k = t->first;
        std::string v = t->second;
        if (k == "proto" or k == "proto_reader" or k == "proto_optimizer"
            or k == "saveme" or k == "loadme") {
          out << "\n### --" << k << "=" << v;
        } else {
          out << "\n# --" << k << "=" << v;
        }
      }
      out << "n#\n#\n#\n#\n";

      if (has_opt("loadme")) {
        std::ifstream in(get_string("loadme").c_str());
        if (in.is_open()) {
          std::string line;
          bool do_output = false;
          while (not in.eof()) {
            getline(in, line);
            if (line.find("model {") != std::string::npos) {
              do_output = true;
            }
            if (do_output) {
              out << line << std::endl;
            }
          }
        }
      } else {

      if (has_opt("proto")) {
        copy_file(get_string("proto"), out);
      }
      if (has_opt("proto_reader")) {
        copy_file(get_string("opt_reader"), out);
      }
      if (has_opt("proto_optimizer")) {
        copy_file(get_string("opt_optimizer"), out);
      }
      }
    }
}
