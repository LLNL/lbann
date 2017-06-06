#include "patchworks_utils.hpp"
#include <cstdio> // popen
#include <string>
#include <iostream>
#include <fstream>
#include <sstream> // std::ostringstream

unsigned int get_screen_resolution(std::vector<std::pair<int, int> >& res)
{
    std::string command = "xrandr | grep '*'";
    FILE *fpipe = (FILE*) popen(command.c_str(),"r");
    char line[256];
    unsigned int cnt = 0u;
    res.clear();

    while ( fgets( line, sizeof(line), fpipe) ) {
        //printf("%s", line);
        std::string rstr(line);
        unsigned int posS=0u;
        unsigned int posE=0u;
        unsigned int posX=0u;
        posX = rstr.find_first_of("xX", posX);
        posS = rstr.find_first_of(" \t\r", posS);
        posE = rstr.find_first_of(" \t\r", posS);
        std::string widthStr = rstr.substr(posS+1, (posX-posS));
        std::string heightStr = rstr.substr(posX+1, (posE-posX));
        std::stringstream wss(widthStr);
        std::stringstream hss(heightStr);
        int w=0, h=0;
        wss >> w;
        hss >> h;
        res.push_back(std::make_pair(w,h));
        cnt ++;
    }
    pclose(fpipe);

    return cnt;
}

struct path_delimiter
{
    bool operator()( char ch ) const
    {
        return ch == '/';
    }
};

bool split_path(const std::string &path, std::string &dir, std::string &name)
{
    std::string::const_iterator nb
        = std::find_if( path.rbegin(), path.rend(), path_delimiter()).base();
    dir =  std::string(path.begin(), nb);
    name = std::string(nb, path.end());
    if (name.empty()) return false;

    return true;
}

std::string name_with_no_extention(const std::string filename)
{
    size_t pos = filename.find_last_of('.');
    if (pos == 0u) return filename;
    return filename.substr(0, pos);
}

std::string get_file_extention(const std::string filename)
{
    size_t pos = filename.find_last_of('.');
    if (pos == 0u) return "";
    return filename.substr(pos+1, filename.size());
}

std::string basename_with_no_extention(const std::string filename)
{
    std::string imgdir;
    std::string imgfile;
    split_path(filename, imgdir, imgfile);
    return name_with_no_extention(imgfile);
}
