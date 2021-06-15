#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/commify.hpp"

using namespace lbann;
using namespace std;

char get_delimiter(); 

int main(int argc, char **argv) {
  lbann::world_comm_ptr comm = lbann::initialize(argc, argv);
  int nprocs;// = comm->get_procs_in_world();
  int me; // = comm->get_rank_in_world();
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  bool master = me == 0 ? true : false;

  try {

    if (argc < 3) {
      if (master) {
        std::cerr
          << "usage: " << argv[0]
          << " --input_fn=<string> --output_fn=<string> --delimiter=<char>\n"
          << "where: input_fn contains a listing of SMILES data files.\n"
          << "       --delimiter is c (comma), t (tab), s (space), ' ' (space) or 0 (none)\n" 
          << "function: computes vocabulary\n\n"
          << "CAUTION: does not discard the first line!\n\n";
        exit(9);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    options *opts = options::get();
    opts->init(argc, argv);
    double tm1 = get_time();

    const std::string input_fn = opts->get_string("input_fn");
    std::ifstream in(input_fn.c_str());
    if (!in) {
      LBANN_ERROR("failed to open ", input_fn , " for reading");
    }

    // P_0 opens output file
    std::ofstream out;
    if (me == 0) {
      const std::string output_fn = opts->get_string("output_fn");
      out.open(output_fn.c_str(), std::ios::binary);
      if (!out) {
        LBANN_ERROR("failed to open ", output_fn, " for writing");
      }
    }

    char delimiter = get_delimiter();

    static int err_count = 0;

    //this will contain all characters over all input SMILES strings
    std::set<char> vocab_set;
    std::string filename;
    int file_count = -1;
    while (in >> filename) {
      ++file_count;
      if (master) {
        std::cout << "num files processed: " << file_count 
                  << " :: " << filename << std::endl;
      }
      if (file_count % nprocs != me) {
        continue;
      }
      std::ifstream in2(filename.c_str());
      if (!in2) {
        LBANN_ERROR("failed to open ", filename , " for reading");
      }

      std::string line;
      int line_count = 0;
      while (getline(in2, line)) {
        ++line_count;

        // user feedback
        if (line_count % 100000000 == 0) {
        //if (line_count % 100000000 == 0 && master) {
            std::cout << me << " :: " << line_count/100000000 << "* 100M lines processed by P_0" << std::endl;
        }

        if (line.size() < 5) {
          continue;
        }  
        size_t h = line.find(delimiter);
        if (h == std::string::npos && delimiter != 0) {
          // assume there is no delimiter
          h = line.size();
          ++err_count;
          if (err_count < 2) {
            ++err_count;
            LBANN_WARNING("failed to find delimiter: ", delimiter, " on line ",
                              line_count, " file: ", filename, " line: ", line);
            LBANN_WARNING("treating the entire lines as a SMILES string");
            if (err_count == 2) {
              LBANN_WARNING("suppressing additional delimiter warnings");
            }
          }
        }
        const std::string smiles = line.substr(0, h);
        for (const auto &t : smiles) {
          vocab_set.insert(t);
        }
      }
    }
    in.close();

    // all ranks send their portions of the vocab to P_0; P_0 combines
    std::stringstream ss;
    for (auto t : vocab_set) {
      ss << t;
    }
    const std::string buf(ss.str());

    if (me > 0) {
      int sz = buf.size();
      MPI_Send((void*)&sz, 1, MPI_INT, 0, me, MPI_COMM_WORLD);
      MPI_Send((void*)buf.data(), sz, MPI_CHAR, 0, me, MPI_COMM_WORLD);
      //comm->send(&sz, 1, 0, me);
      //comm->send(ss.str().data(), ss.str().size(), 0, me);
    }
    else {
      for (int jj=0; jj<nprocs; jj++) {
        if (jj != me) {
          std::vector<char> w;
          int sz;
          MPI_Status status;

          MPI_Recv(&sz, 1, MPI_INT, jj, jj, MPI_COMM_WORLD, &status);
          w.resize(sz);
          MPI_Recv(w.data(), sz, MPI_CHAR, jj, jj, MPI_COMM_WORLD, &status);
          std::cout << "rcvd from " << jj << "   ";
          for (auto t : w) {
            vocab_set.insert(t);
            std::cout << t;
          }
          std::cout << "   vocab size is now: " << vocab_set.size() << std::endl;
          //comm->recv(&sz, 1, 0, jj);
          //comm->recv(w.data(), sz, 0, jj);
        }
      }
    }

    // P_0 outputs the vocabulary
    if (!me) {
      int idx = 0;
      for (const auto &t : vocab_set) {
        out << t << " " << idx++ << std::endl;
      }
      out << "<bos> " << idx++ << std::endl;
      out << "<eos> " << idx++ << std::endl;
      out << "<pad> " << idx++ << std::endl;
      out << "<unk> " << idx++ << std::endl;
      out.close();
    }

    if (master) std::cout << "\nprocessing time: " << get_time() - tm1 << std::endl;

  } catch (lbann::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

char get_delimiter() {
    const std::string w = options::get()->get_string("delimiter");
    const char ww = w[0];
    char d = 0;
    switch (ww) {
      case 's' :
        d = ' ';
        break;
      case ' ' :
        d = ' ';
        break;
      case 'c' :
        d = ',';
        break;
      case 't' :
        d = '\t';
        break;
      case '0' :
        d = '\0';
        break;
      default :
        LBANN_ERROR("Invalid delimiter character; should be 's' (space), ' ' (space), 'c' (comma), 't' (tab) , or '0' (none). You passed: ", ww, "; if you pass a string for --delimiter, we use the first character as the delimit character");
    }
    return d;
}
