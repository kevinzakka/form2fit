echo "Downloading benchmark data..."
cd benchmark
fileid="1V8mW64uz-kmK2WJpNMeCTAlp8BYeAV_r"
filename="data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
unzip -q data.zip
rm -rf __MACOSX
rm -rf data.zip
echo "Done."