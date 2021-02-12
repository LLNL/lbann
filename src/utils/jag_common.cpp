#include "lbann/comm_impl.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void read_filelist(lbann_comm *comm, const std::string &fn, std::vector<std::string> &filelist_out) {
    const int rank = comm->get_rank_in_world();
    std::string f; // concatenated, space separated filelist
    int f_size;

    // P_0 reads filelist
    if (!rank) {

      std::ifstream in(fn.c_str());
      if (!in) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + fn + " for reading");
      }

      std::stringstream s;
      std::string line;
      while (getline(in, line)) {
        if (line.size()) {
          s << line << " ";
        }
      }
      in.close();

      f = s.str();
      f_size = s.str().size();
    }

    // bcast concatenated filelist
    comm->world_broadcast<int>(0, &f_size, 1);
    f.resize(f_size);
    comm->world_broadcast<char>(0, &f[0], f_size);

    // unpack filelist into vector
    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        filelist_out.push_back(filename);
      }
    }
    if (!rank) std::cerr << "num files: " << filelist_out.size() << "\n";
}

} // namespace lbann {
