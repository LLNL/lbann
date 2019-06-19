#ifndef LBANN_UTILS_OPTIONS_HPP_INCLUDED
#define LBANN_UTILS_OPTIONS_HPP_INCLUDED

#include <iostream>
#include <map>
#include <vector>
#include <string>

namespace lbann {

/** Singleton, globally accessible class, for setting and retrieving
 *  options (key/value pairs).
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

  /// prints all registered options to 'out'
  void print(std::ostream &out = std::cout);

  //@{
  /** Returns true if the database contains the option */
  bool has_int(std::string option);
  bool has_string(std::string option);
  bool has_float(std::string option) { return has_double(option); }
  bool has_double(std::string option);
  //@}

  //@{
  /** insert option in database; if option already exists it's value will changed */
  void set_option(std::string name, int value);
  void set_option(std::string name, bool value);
  void set_option(std::string name, std::string value);
  void set_option(std::string name, float value);
  void set_option(std::string name, double value);
  //@}

  //@{ hack to pass around data structures
  void set_ptr(void *p) { m_ptrs.push_back(p); }
  std::vector<void*> & get_ptrs() { return m_ptrs; }
  void clear_ptrs() { m_ptrs.clear(); }
  //@}

  //@{
  /** returns the value of the option; throws exception if option doesn't exist,
   *  or if it's associated value (which is stored internally as a string)
   *  cannot be cast to the specified return type
   */
  int get_int(std::string option);
  bool get_bool(std::string option);
  std::string get_string(std::string option);
  float get_float(std::string option) { return (float) get_double(option); }
  double get_double(std::string option);
  //@}

  //@{
  /** returns the value of the option; if option isn't found,
   * inserts the option in the internal db, with the specified
   * default value, and returns the the_default
   */
  int get_int(std::string option, int the_default);
  bool get_bool(std::string option, bool the_default);
  std::string get_string(std::string option, std::string the_default);
  float get_float(std::string option, float the_default) { return (float) get_double(option, the_default); }
  double get_double(std::string option, double the_default);
  //@}

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

  void m_parse_file(std::string fn);

  bool m_has_opt(std::string option);
  bool m_test_int(std::string option, int &out);
  bool m_test_double(std::string option, double &out);
  bool m_test_string(std::string option, std::string &out);

  void m_parse_cmd_line(int arg, char **argv);

  std::vector<std::string> m_cmd_line;

  int m_argc;
  char **m_argv;

  std::vector<void*> m_ptrs;
};

} // namespace lbann

#endif // LBANN_UTILS_OPTIONS_HPP_INCLUDED
