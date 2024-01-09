all: compress

compress: compress.cpp
    g++ compress.cpp -lsnappy -lcrypto -o compress

clean:
    rm -f compress