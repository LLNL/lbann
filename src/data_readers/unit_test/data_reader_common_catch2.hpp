#ifndef __DATA_READER_TEST_COMMON_HPP__
#define __DATA_READER_TEST_COMMON_HPP__

#include <sys/stat.h>  //for mkdir
#include <sys/types.h> //for getpid
#include <sys/types.h> //for mkdir
#include <unistd.h>    //for getpid

#include "lbann/data_readers/data_reader.hpp"
#include <lbann/base.hpp>

#include <google/protobuf/text_format.h>
#include <lbann.pb.h>
namespace pb = ::google::protobuf;

/** create a directory in /tmp; returns the pathname to the directory */
std::string create_test_directory(std::string base_name)
{
  char b[2048];
  std::stringstream s;
  s << "/tmp/" << base_name << "_" << getpid();
  const std::string dir = s.str();
  lbann::file::make_directory(dir);
  // test that we can write files
  sprintf(b, "%s/test", dir.c_str());
  std::ofstream out(b);
  REQUIRE(out.good());
  out.close();
  return dir;
}

/** Instantiates one or more data readers from the input 'prototext' string.
 *  Users should ensure that the appropriate options (if any) are set prior
 *  to calling this function, i.e:
 *    lbann::options *opts = lbann::options::get();
 *    opts->set_option("preload_data_store", true);
 */
std::map<lbann::execution_mode, lbann::generic_data_reader*>
instantiate_data_readers(std::string prototext_in,
                         lbann::lbann_comm& comm_in,
                         lbann::generic_data_reader*& train_ptr,
                         lbann::generic_data_reader*& validate_ptr,
                         lbann::generic_data_reader*& test_ptr,
                         lbann::generic_data_reader*& tournament_ptr)
{
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(prototext_in, &my_proto)) {
    throw "Parsing protobuf failed.";
  }

  std::map<lbann::execution_mode, lbann::generic_data_reader*> data_readers;
  lbann::init_data_readers(&comm_in, my_proto, data_readers);

  // get pointers the the various readers
  train_ptr = nullptr;
  validate_ptr = nullptr;
  test_ptr = nullptr;
  for (auto t : data_readers) {
    if (t.second->get_role() == "train") {
      train_ptr = t.second;
    }
    if (t.second->get_role() == "validate") {
      validate_ptr = t.second;
    }
    if (t.second->get_role() == "tournament") {
      tournament_ptr = t.second;
    }
    if (t.second->get_role() == "test") {
      test_ptr = t.second;
    }
  }

  return data_readers;
}

void write_file(std::string data, std::string dir, std::string fn)
{
  std::stringstream s;
  s << dir << "/" << fn;
  std::ofstream out(s.str().c_str());
  REQUIRE(out);
  out << data;
  out.close();
}

#endif //__DATA_READER_TEST_COMMON_HPP__
