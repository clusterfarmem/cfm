all: compress

compress: compress.cpp
    g++ -o compress compress.cpp -lsnappy -lcrypto

clean:
    rm -f compress