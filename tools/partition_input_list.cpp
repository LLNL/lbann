#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int main( int argc, char** argv)
{
  if(argc < 4) { 
    cout << "Usage .... exec input_file output_file_basename num_partitions" << endl;
    exit(-1);
  }
    
  std::string input_file = argv[1];
  std::string output_file = argv[2];
  int num_partition = atoi(argv[3]);

  std::ifstream infile(input_file);
  std::vector<std::pair<std::string, int> > lines;
  std::string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }
  
  std::cout << "Input file " << input_file << std::endl;
  std::cout << "Shuffling data" << std::endl;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(lines.begin(), lines.end(),std::default_random_engine(seed));

  int num = lines.size();
  int batch = int(num/num_partition);

  std::cout << "A total of " << num  << " rows";
  std::cout << "  Num of partition " << num_partition;
  std::cout << "  Data per partition " << batch << std::endl;

  std::cout << "Original file shuffled and save as: " << output_file << std::endl;
  ofstream base_ofs(output_file.c_str());
  if (!base_ofs) { std::cout << "\n In write_file can't open file : " << output_file;  exit(1); }
  for(auto& l0 : lines) base_ofs << l0.first << " " << l0.second << std::endl;

  std::cout << " Start partitioning "  << std::endl;
  for(int p = 0; p < num_partition; p++) {
    int start = p * batch;
    int end = start + batch;
    std::string partition_file = output_file + ".p" + to_string(p);
    std::cout << "Partitioned file name " << partition_file << std::endl;
    ofstream ofs(partition_file.c_str());
    if (!ofs) { std::cout << "\n In write_file: can't open file : " << partition_file;  exit(1); }
    for(int l = start; l < end; l++)
        ofs << lines[l].first << " " << lines[l].second << endl;
  }

  std::cout << "DONE!" << std::endl;
  return 0;

}
