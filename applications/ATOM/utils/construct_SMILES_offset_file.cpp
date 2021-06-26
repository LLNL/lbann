#include <iostream>
#include <string>
#include <fstream>
#include "lbann/comm_impl.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"

using namespace lbann;

bool Has_header = false;

// returns usage string
std::string usage(int argc, char **argv);

// parse the cmd line for filenames and delimiter
void parse_inputs(char &delimiter, std::string &input_filename, std::string &output_filename);

// returns the number of lines in the file (minus one for header)
size_t get_num_samples(std::string input_filename, size_t max);

//===========================================================================

int main(int argc, char **argv) {
  lbann::world_comm_ptr comm = lbann::initialize(argc, argv);

  int np = comm->get_procs_in_world();
  if (np != 1) {
    LBANN_ERROR("please run with a single processor");
  }

  if (argc < 2) {
    std::cerr << usage(argc, argv);
    return EXIT_FAILURE;
  }

  options *opts = options::get();
  opts->init(argc, argv);
  if (opts->get_bool("has_header")) {
    Has_header = true;
  }

  char delimiter = '\0';
  std::string input_filename;
  std::string output_filename;
  parse_inputs(delimiter, input_filename, output_filename);

  try {

    //for testing and debugging during development
    long long max = INT_MAX;
    if (opts->has_int("max")) {
       max = opts->get_int("max");
    }
    std::cout << "\nRunning with max samples = " << max << std::endl;

    size_t n_samples = get_num_samples(input_filename, max);
    std::cout << "get_num_samples() = " << n_samples << std::endl;

    // open input file; optionally discard header
    std::ifstream in(input_filename.c_str());
    if (!in) {
      LBANN_ERROR("failed to open ", input_filename, " for reading");
    }
    long long offset = 0;
    std::string line;
    if (Has_header) {
      getline(in, line); //discard header
      offset = line.size() +1;
    }
    std::cout << "\nfile offset of start of 1st sample: " << offset << std::endl;

    // open output file
    char bb[1024];
    sprintf(bb, "%s.offsets", output_filename.c_str());
    std::ofstream outb(bb, std::ios::binary);
    if (!outb) {
      LBANN_ERROR("failed to open ", bb, " for writing");
    }

    // loop over the samples (lines), writing output as we iterate
    long long line_no = 0;
    const unsigned int Max_length  = std::numeric_limits<unsigned short>::max();
    while (getline(in, line)) {
      ++line_no;
      if (!line.size()) {
        LBANN_ERROR("line.size() == 0 for line number ", line_no);
      }

      // Get the length of the SMILES string. This may be shortened to USHRT_MAX
      size_t length = line.size();
      if (delimiter != '\0') {
        size_t d = line.find(delimiter);
        if (d == std::string::npos) {
          LBANN_ERROR("failed to find delimiter in line number ", line_no);
        }
        length = d;
      }
      unsigned short len = length > Max_length ? Max_length : length;

      outb.write((char*)&offset, sizeof(long long));
      outb.write((char*)&len, sizeof(unsigned short));
      offset += line.size()+1;

      if (line_no % 100000000 == 0) {
        std::cout << line_no/100000000 << "*100M offsets written\n";
      }

      if (line_no >= max) {
        break;
      }
    }

    std::cout << "parseme " << input_filename << " "
              << bb << " " << line_no << std::endl;

    in.close();
    outb.close();

    std::cout << std::endl

              << "wrote binary output to " << bb << ";\n"
              << "offsets contain " << sizeof(long long) << " bytes, and the\n"
              << "lengths contain " << sizeof(unsigned short) << " bytes\n";

  } catch (lbann::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

std::string usage(int argc, char **argv) {
  std::stringstream s;
  s <<
    "usage: " << argv[0] << " --input_fn=<string> [--delimiter=<char>] [--has_header]\n"
    "where: input_fn is a file containing a listing of SMILES strings;\n"
    "       --delimiter; use: --delimiter=' ' or --delimiter=',';\n"
    "         you need to modify the code to handle tabs ...\n"
    "         if --delimiter is not given, assumes each line contains \n"
    "         a single, newline delimited SMILES string\n"

    "\nfunction: constructs a binary file with offsets and lengths of the samples;\n"
    "offsets are long long, and lengths are short; SMILES strings that are\n"
    "too long will have their lengths recorded as std::numeric_limits<unsigned short>::max()\n\n"
    "Note: the output_fn is same as input fn, with the '.offsets' suffix added\n";
  return s.str();
}

void parse_inputs(char &delimiter, std::string &input_filename, std::string &output_filename) {
  options *opts = options::get();
  if (!opts->has_string("input_fn")) {
    LBANN_ERROR("opts->has_string('input_fn') failed");
  }
  input_filename = opts->get_string("input_fn");

  /*
  if (!opts->has_string("output_fn")) {
    LBANN_ERROR("opts->has_string('output_fn') failed");
  }
  output_filename = opts->get_string("output_fn");
  */
  output_filename = input_filename;
  delimiter = '\0';
  if (opts->has_string("delimiter")) {
    delimiter = opts->get_string("delimiter")[0];
    std::cout << "Setting delimiter to >>" << delimiter << "<<\n";
  }
}

size_t get_num_samples(std::string fn, size_t max) {
  std::cout << "Counting file lines; please wait\n";
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", fn, " for reading");
  }
  std::string line;
  if (Has_header) {
    getline(in, line); // discard header
  }
  size_t n = 0;
  while (getline(in, line)) {
    ++n;
    if (n >= max) {
      return max;
    }
    if (n % 100000000 == 0) {
      std::cout << n/100000000 << "*100M lines processed\n";
    }
  }
  in.close();
  std::cout << "Finished. Num Lines: " << n << std::endl;
  return n;
}
