#include "mpi.h"
#include "lbann/utils/options.hpp"
#include <ctime>
#include <dirent.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

options * options::s_instance = new options;

//============================================================================

void m_parse_opt(std::string tmp, std::string &key, std::string &val)
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
  m_argc = argc;
  m_argv = argv;

  //the_default fileBaseName for saving
  m_opts["saveme"] = "data.prototext";

  //save cmd line
  std::string key;
  std::string value;
  std::string loadme;
  for (int j=0; j<argc; j++) {
    m_cmd_line.emplace_back(argv[j]);
    m_parse_opt(argv[j], key, value);
    if (key == "loadme") {
      loadme = value;
    }
  }

  //optionally init from file
  if (loadme != "") {
    m_parse_file(loadme.c_str());
  }

  //over-ride from cmd line
  m_parse_cmd_line(argc, argv);
}

//============================================================================

void lower(std::string &s)
{
  for (char & j : s) {
    if (isalpha(j)) {
      j = tolower(j);
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

bool options::m_has_opt(std::string option)
{
  if (m_opts.find(option) == m_opts.end()) {
    return false;
  }
  return true;
}

bool options::get_bool(std::string option, bool the_default)
{
  int result;
  if (!m_test_int(option, result)) {
    set_option(option, the_default);
    return the_default;
  }
  return result;
}

bool options::get_bool(std::string option) 
{
  int result;
  if (!m_test_int(option, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " ::options::get_int() - failed to find option: " << option
        << ", or to convert to int";
    throw std::runtime_error(err.str());
  }
  if (result == 0) return false;
  return true;
}

int options::get_int(std::string option, int the_default)
{
  int result;
  if (!m_test_int(option, result)) {
    set_option(option, the_default);
    return the_default;
  }
  return result;
}

int options::get_int(std::string option)
{
  int result;
  if (!m_test_int(option, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_int() - failed to find option: " << option
        << ", or to convert to int";
    throw std::runtime_error(err.str());
  }
  return result;
}

double options::get_double(std::string option, double the_default) {
  double result;
  if (!m_test_double(option, result)) {
    set_option(option, the_default);
    return the_default;
  }
  return result;
}

double options::get_double(std::string option)
{
  double result;
  if (!m_test_double(option, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_double() - failed to find option: " << option
        << ", or to convert the value to double";
    throw std::runtime_error(err.str());
  }
  return result;
}

std::string options::get_string(std::string option, std::string the_default)
{
  std::string result;
  if (!m_test_string(option, result)) {
    return the_default;
  }  
  return the_default;
}

std::string options::get_string(std::string option)
{
  std::string result;
  if (!m_test_string(option, result)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: options::get_string() - failed to find option: " << option;
    throw std::runtime_error(err.str());
  }
  return result;
}

//============================================================================

bool options::m_test_int(std::string option, int &out)
{
  if (!m_has_opt(option)) {
    return false;
  }
  bool is_good = true;
  std::string val = m_opts[option];
  for (char j : val) {
    if (!isdigit(j)) {
      is_good = false;
      break;
    }
  }
  if (!is_good) {
    return false;
  }

  std::stringstream s(val);
  if (! (s>>out)) {
    return false;
  }
  return true;
}

bool options::m_test_double(std::string option, double &out)
{
  if (!m_has_opt(option)) return false;
  std::string val(m_opts[option]);
  lower(val);
  for (char j : val) {
    if (!(isdigit(j) or j == '-'
             or tolower(j == 'e') or j == '.')) {
      return false;
    }
  }
  std::stringstream s(val);
  if (! (s>>out)) {
    return false;
  }
  return true;
}

bool options::m_test_string(std::string option, std::string &out)
{
  if (!m_has_opt(option)) return false;
  out = m_opts[option];
  return true;
}

bool options::has_int(std::string option)
{
  int test;
  if (m_test_int(option, test)) return true;
  return false;
}

bool options::has_bool(std::string option)
{
  int test;
  if (m_test_int(option, test)) return true;
  return false;
}

bool options::has_string(std::string option) {
  std::string test;
  if (m_test_string(option, test)) return true;
  return false;
}

bool options::has_double(std::string option) {

  double test;
  if (m_test_double(option, test)) return true;
  return false;
}

//====================================================================
void options::set_option(std::string name, int value) {
  m_opts[name] = std::to_string(value);
}

void options::set_option(std::string name, bool value) {
  m_opts[name] = std::to_string(value);
}

void options::set_option(std::string name, std::string value) {
  m_opts[name] = value;
}

void options::set_option(std::string name, float value) {
  m_opts[name] = std::to_string(value);
}

void options::set_option(std::string name, double value) {
  m_opts[name] = std::to_string(value);
}

//====================================================================

void options::m_parse_file(std::string fn) 
{
  std::ifstream in(fn.c_str());
  if (!in.is_open()) {
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

    m_parse_opt(line, key, val);
    if (key != "") {
      m_opts[key] = val;
    }
  }
  in.close();
}


void options::m_parse_cmd_line(int argc, char **argv)
{
  std::string key, val;
  for (int j=1; j<argc; j++) {
    m_parse_opt(argv[j], key, val);
    if (key != "") {
      m_opts[key] = val;
    }
  }
}

void options::print(std::ostream &out) {
  out << "# The following Options were used in the code\n";
  for (auto t : m_opts) {
    out << "--" << t.first << "=" << t.second << " ";
  }
  out << std::endl;
}

