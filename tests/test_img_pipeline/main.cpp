#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include "lbann/data_readers/lbann_image_utils.hpp"


using namespace lbann::patchworks;

bool test_image_io(const std::string filename);

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: > " << argv[0] << " image_filename" << std::endl;
        return 0;
    }

    std::string filename = argv[1];

    // read write test with converting to/from a serialized buffer
    bool ok = test_image_io(filename);
    if (!ok) {
        std::cout << "Test failed" << std::endl;
        return 0;
    }
    std::cout << "Complete!" << std::endl;

    return 0;
}

void show_image_size(const int width, const int height, const int type)
{
    const int depth = CV_MAT_DEPTH(type);
    const int NCh = CV_MAT_CN(type);
    const int esz = CV_ELEM_SIZE(depth);
    std::cout << "Image size                     : " << width << " x " << height << std::endl;
    std::cout << "Number of channels             : " << NCh << std::endl;
    std::cout << "Size of the channel value type : " << esz << std::endl;
    std::cout << "Total bytes                    : " << width*height*NCh*esz << std::endl;
}

std::string get_file_extention(const std::string filename)
{
    size_t pos = filename.find_last_of('.');
    if (pos == 0u) return "";
    return filename.substr(pos+1, filename.size());
}

bool read_file(const std::string filename, std::vector<unsigned char>& buf)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.good()) return false;

    file.unsetf(std::ios::skipws);

    file.seekg(0, std::ios::end);
    const std::streampos file_size = file.tellg();

    file.seekg(0, std::ios::beg);

    buf.reserve(file_size);

    buf.insert(buf.begin(),
               std::istream_iterator<unsigned char>(file),
               std::istream_iterator<unsigned char>());

    return true;
}

void write_file(const std::string filename, const std::vector<unsigned char>& buf)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file.write((const char*) buf.data(), buf.size() * sizeof(unsigned char));
    file.close();
}

bool test_image_io(const std::string filename)
{
    // define a data matrix
    ::Mat Images;
    // normally the minibatch data size is known in advance.
    // However, in this example it is not, so we resize on-demand after reading image
    // Here, we just set some number
    const int minimal_data_len = 1;
    const int minibatch_size = 2;
    Images.Resize(minimal_data_len, minibatch_size); // minibatch

    lbann::cv_process pp;

    std::vector<unsigned char> buf;
    bool ok = read_file(filename, buf);
    if (!ok) {
        std::cout << "failed to load" << std::endl;
        return false;
    }
    const unsigned int fsz = buf.size();
    std::cout << "file size " << fsz << std::endl;

    int width = 0;
    int height = 0;
    int type = 0;
    ok = lbann::image_utils::import_image(buf, width, height, type, pp, Images);
    if (!ok) {
        std::cout << "failed to import" << std::endl;
        return false;
    }
    show_image_size(width, height, type);

    ::Mat ImageC0_v;
    ::Mat ImageC1_v;
    const int sz = Images.Height();
    View(ImageC0_v, Images, 0, 0, sz, 1);
    View(ImageC1_v, Images, 0, 1, sz, 1);
    ImageC1_v = ImageC0_v;

    buf.clear();

    // Write an image
    const std::string ext = get_file_extention(filename);
    ok = lbann::image_utils::export_image(ext, buf, width, height, type, pp, ImageC1_v);
    write_file("copy4." + ext, buf);
    return ok;
}
