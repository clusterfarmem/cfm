#!/bin/bash

echo '<?xml version="1.0" encoding="UTF-8"?>' > merged.xml
echo '<root>' >> merged.xml


for file in enwiki-*
do
    if [ -f "$file" ]
    then
        sed '1d;$d' $file >> merged.xml
    fi
done

echo '</root>' >> merged.xml