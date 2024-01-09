#!/bin/bash

# Base URL for the files
base_url="https://dumps.wikimedia.org/enwiki/20240101/"

# Loop over the numbers 1 to 12
for i in $(seq 1 12); do
  # Generate the file name
  file="enwiki-20240101-pages-articles${i}.xml-*.bz2"

  
  wget -r -nd -np -A ${file} ${base_url}

  # Download the file
  # wget "${base_url}${file}"

  # Uncompress the file
  # bzip2 -d "${file}"
done

bunzip2 *.bz2

