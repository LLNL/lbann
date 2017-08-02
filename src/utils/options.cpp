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

  if (!m_rank) {
    std::cout << std::endl
              << "running with the following options:\n";
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

double options::get_double(const char *option)
{
  std::string opt(option);
  double result;
  if (not get_double(opt, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_double() - failed to find option: " << option
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

bool options::get_double(std::string option, double &out)
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

bool options::has_double(const char *option)
{
  std::string opt(option);
  double test;
  if (get_double(opt, test)) return true;
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



