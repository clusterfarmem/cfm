import os
import sys
import hashlib
import snappy

def compress_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    compressor = snappy.StreamCompressor()
    compressed_data = compressor.compress(data)
    compressor.flush()
    #compressed_filename = filename + '.snappy'
    #with open(compressed_filename, 'wb') as f:
        #f.write(compressed_data)
    return compressed_data

def calculate_md5(data):
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()

def main():
    filename = sys.argv[1]
    with open(filename, 'rb') as f:
        data = f.read()
    compressor = snappy.StreamCompressor()
    compressed_data = compressor.compress(data)
    compressor.flush()
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    
    print('MD5 of compressed file:', hash_md5.hexdigest())


if __name__ == "__main__":
    main()