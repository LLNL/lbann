#ifndef __OPTIONS_HPP__
#define __OPTIONS_HPP__

#include <cassert>
#include <map>
#include <vector>
#include <string>


/*
 * This is a singleton, globally accessible class for setting
 * and retrieving options
 */

class options
{
public :

  /// returns a pointer to the Options singleton
  static options * get() {
    return s_instance;
  }

  
  /** if cmd line contains "--loadme=<string>" then initialize
   *  options database from that file. Next, intialize
   *  from all other options on cmd line. */
  void init(int argc, char **argv);

  /// prints information on available options and the_defaults to cout
  void help();

  /// Returns true if the database contains the option
  bool has_int(const char *option);
  /// Returns true if the database contains the option
  bool has_bool(const char *option);
  /// Returns true if the database contains the option
  bool has_string(const char *option);
  /// Returns true if the database contains the option
  bool has_float(const char *option) {
    return has_double(option);
  }
  /// Returns true if the database contains the option
  bool has_double(const char *option);

  /// insert option in database; if option already exists it's value will changed
  void set_option(const char *name, const char *value) {
    m_opts[name] = value;
  }  

  /// returns the value of the option; throws exception if option doesn't exist
  int get_int(const char *option);
  /// returns the value of the option; if option isn't found, returns the the_default
  int get_int(const char *option, int the_default);

  /// returns the value of the option; throws exception if option doesn't exist
  bool get_bool(const char *option);

  /// returns the value of the option; throws exception if option doesn't exist
  float get_float(const char *option) {
    return (float) get_double(option);
  }

  /// returns the value of the option; if option isn't found, returns the the_default
  float get_float(const char *option, float the_default) {
    return (float) get_double(option);
  }

  /// returns the value of the option; throws exception if option doesn't exist
  double get_double(const char *option);

  /// returns the value of the option; if option isn't found, returns the the_default
  double get_double(const char *option, double the_default);

  /// returns the value of the option; throws exception if option doesn't exist
  std::string get_string(const char *option);

  /// returns the value of the option; if option isn't found, returns the the_default
  std::string get_string(const char *option, std::string the_default);

private:
  int m_rank;

  ///private constructor to ensure this class is a singleton
  options() {}
  options(options &) {}
  options& operator=(options const&) { return *this; }

  //the singleton instance
  static options *s_instance;

  /// the "database;" all options are stored as strings, and may be cast to int, etc.
  std::map<std::string, std::string> m_opts;

  void parse_file(const char *fn);

  bool has_opt(std::string option);
  bool get_int(std::string option, int &out);
  bool get_double(std::string option, double &out);
  bool get_string(std::string option, std::string &out);

  void parse_cmd_line(int arg, char **argv);

  std::vector<std::string> m_cmd_line;
};

#endif

