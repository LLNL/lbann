#include <iostream>
#include <string>
#include <fstream>
#include <unordered_set>
#include <set>
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/commify.hpp"

using namespace lbann;

int main(int argc, char **argv) {
  int random_seed = 42;
  lbann::world_comm_ptr comm = lbann::initialize(argc, argv, random_seed);
  int np = comm->get_procs_in_world();

  std::cerr << "STARTED!\n";

  try {

    if (np != 1) {
      LBANN_ERROR("please run with a single processor");
    }
    if (argc < 3) {
      std::cerr 
        << "usage: " << argv[0] 
        << " --input_fn=<string> --output_fn=<string> --delimiter=<char>\n"
        << "where: input_fn is csv file containing SMILES strings\n"
        << "function: computes vocabulary\n";
      exit(9);
    }

    options *opts = options::get();
    opts->init(argc, argv);
    double tm1 = get_time();

    const std::string input_fn = opts->get_string("input_fn");
    std::ifstream in(input_fn.c_str());
    if (!in) {
      LBANN_ERROR("failed to open ", input_fn , " for reading");
    }

    const std::string output_fn = opts->get_string("output_fn");
    std::ofstream out(output_fn.c_str(), std::ios::binary);
    if (!out) {
      LBANN_ERROR("failed to open ", output_fn, " for writing");
    }

    const std::string w = opts->get_string("delimiter");
    const char d = '\t';
    //const char d = w[0];

    //std::set<char> s;
    std::unordered_set<char> s;

    std::string line;
    getline(in, line); //discard header
    size_t j = 1;
    while (!in.eof()) {
      ++j;
      if (j % 1000 == 0) std::cout << j/1000 << "K lines processed" << std::endl;
      getline(in, line);
      if (line.size() < 5) continue;
      size_t h = line.find(d);
      if (h == std::string::npos) {
        LBANN_ERROR("failed to find delimiter: ", d, " on line ", j);
      }
      const std::string smiles = line.substr(0, h);
      for (const auto &t : smiles) {
        s.insert(t);
      }
    }

    int idx = 0;
    for (const auto &t : s) {
      out << t << " " << idx++ << std::endl;
    }
    out << "<bos> " << idx++ << std::endl;
    out << "<eos> " << idx++ << std::endl;
    out << "<pad> " << idx++ << std::endl;
    out << "<unk> " << idx++ << std::endl;

    in.close();
    out.close();

    std::cout << "\nprocessing time: " << get_time() - tm1 << std::endl;

  } catch (lbann::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

