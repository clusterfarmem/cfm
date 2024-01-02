#include <fstream>
#include <sstream>
#include <iomanip>
#include <openssl/md5.h>
#include <snappy.h>

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

int main() {
    std::ifstream input("enwiki-20240101-pages-articles-multistream.xml", std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    std::string compressed;
    snappy::Compress(data.data(), data.size(), &compressed);

    std::string md5sum = md5(compressed);
    std::cout << "MD5: " << md5sum << std::endl;

    std::ofstream output("enwiki-20240101-pages-articles-multistream.snappy", std::ios::binary);
    output.write(compressed.data(), compressed.size());

    std::remove("enwiki-20240101-pages-articles-multistream.snappy");

    return 0;
}