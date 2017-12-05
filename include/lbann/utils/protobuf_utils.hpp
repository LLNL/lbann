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

  /** convience wrapper: calls parse_prototext_filenames_from_command_line(),
   *  then load_prototext(), then verify_prototext(). This is the only function
   *  that needs to be called from, e.g, model_zoo/lbann.cpp; the three called
   *  functions are made public for testing.
   */
  static void load_prototext(
                const bool master,
                const int argc,
                char **argv,
                std::vector<lbann_data::LbannPB *> &models_out); 


  /** parses the command line for --model=<string> --reader=<string>
   *  optimizer=<string> and their multi counterparts: 
   *  --model={<string_1>,<string_2>,...}
   *  --reader={<string_1>,<string_2>,...}
   *  --optimizer={<string_1>,<string_2>,...}
   *  If the multi-model option is given, the reader and optimzier
   *  can either be single, or contain the same number of filenames
   *  as does the --model={...} specification
   */
  static void parse_prototext_filenames_from_command_line(
               bool master,
               int argc, 
               char **argv, 
               std::vector<prototext_fn_triple> &names);

  static void read_in_prototext_files(
                bool master,
                std::vector<prototext_fn_triple> &names,
                std::vector<lbann_data::LbannPB*> &models_out);

  /** attempts to verify the all models are valid, and contain an
   *  optimizer and reader
   */
  static void verify_prototext(
               bool master, 
               const std::vector<lbann_data::LbannPB *> &models);

};

} //namespace lbann 
#endif

