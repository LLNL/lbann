#ifndef __PROTOBUF_UTILS_HPP__
#define __PROTOBUF_UTILS_HPP__

#include <vector>
#include "lbann/lbann.hpp"
#include <lbann.pb.h>

namespace lbann {

/**
 * static methods for parsing command line for prototext filenames,
 * reading in prototext files, etc.
 */

struct prototext_fn_triple {
  std::string model;
  std::string reader;
  std::string optimizer;
};


class protobuf_utils
{
public :

  /* parses the command line for --model=<string> --reader=<string>
   * optimizer=<string> and their multi counterparts: 
   * --model={<string_1>,<string_2>,...}
   * --reader={<string_1>,<string_2>,...}
   * --optimizer={<string_1>,<string_2>,...}
   * If the multi-model option is given, the reader and optimzier
   * can either be single, or contain the same number of filenames
   * as does the --model={...} specification
   */
  static void parse_prototext_filenames_from_command_line(
               int argc, 
               const char **argv, 
               std::vector<prototext_fn_triple> &names);

  static void load_prototext(
                const std::vector<prototext_fn_triple> &names,
                std::vector<lbann_data::LbannPB> &models_out);
};

  static

} //namespace lbann 
#endif

