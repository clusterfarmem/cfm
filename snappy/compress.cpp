#include <fstream>
#include <sstream>
#include <iomanip>
#include <openssl/md5.h>
#include <snappy.h>
#include <iostream>
#include <vector>

std::string md5(const std::string &data) {
    unsigned char result[MD5_DIGEST_LENGTH];
    MD5((unsigned char*)data.data(), data.size(), result);

    std::ostringstream sout;
    sout<<std::hex<<std::setfill('0');
    for(auto c : result) {
        sout<<std::setw(2)<<(int)c;
    }

    return sout.str();
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    std::ifstream input(argv[1], std::ios::binary);

    input.seekg(0, std::ios::end);
    std::size_t size = input.tellg();
    input.seekg(0, std::ios::beg);


    std::string data;
    data.resize(size);

    std::size_t chunk_size = 1024;
    std::cout << "filesize: " << size << std::endl;

    int i = 0;
    for(; i < size - chunk_size; i += chunk_size) {
        input.read(&data[i], chunk_size);
    }

    //std::size_t chunk_size_tmp = size - i;
    //input.read(&data[i], chunk_size_tmp);
    
    //readFileIntoString(filename, data, 4096);
    //input.read(&data[0], size);

    std::string compressed;
    snappy::Compress(data.data(), data.size(), &compressed);

    std::string md5sum = md5(compressed);
    std::cout << "MD5: " << md5sum << std::endl;

    std::ofstream output("merged.snappy", std::ios::binary);
    output.write(compressed.data(), compressed.size());

    std::remove("merged.snappy");

    return 0;
}