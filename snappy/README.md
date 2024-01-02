# Snappy SETUP

## Download & unzip enwiki dump 

using the following commands:

```shell
wget https://dumps.wikimedia.org/enwiki/20240101/enwiki-20240101-pages-articles-multistream.xml.bz2
bunzip2 enwiki-20240101-pages-articles-multistream.xml.bz2
```

## Install OpenSSL

using the following command

```shell
sudo apt-get install libssl-dev
```

## Install Snappy

using the following commands:

```shell
cd snappy
./configure
make
sudo make install
```

## Compile

using the following command:

```shell
make
```

