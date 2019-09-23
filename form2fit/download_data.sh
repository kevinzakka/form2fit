#!/usr/bin/env bash

echo "Downloading benchmark data..."
cd benchmark
fileid="1HDzOc45qrwMjmjxg4FUa-X2td0Rf-Rx7"
filename="data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
unzip -q data.zip
mv form2fit/data/ .
rm -rf form2fit
rm data.zip
echo "Done."