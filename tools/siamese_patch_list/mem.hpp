
#include <sys/sysinfo.h>
#include <limits>
#include <fstream>

unsigned long long getTotalSystemMemory() {
    std::string token;
    std::ifstream file("/proc/meminfo", std::ifstream::in);
    while(file >> token) {
        if(token == "MemFree:") {
            unsigned long mem;
            if(file >> mem) {
                return mem;
            } else {
                return 0;       
            }
        }
        // ignore rest of the line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0; // nothing found
}

void print_mem(std::string tag) {
  std::cout << tag << ' ' << getTotalSystemMemory() << std::endl;
}
