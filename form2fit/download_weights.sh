echo "Downloading model weights..."
cd code/ml/models
fileid="1nAAs7Wbfe9wywi8LtYSg1jLaIe3zxVJJ"
filename="weights.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm cookie
unzip -q weights.zip
rm -rf __MACOSX
rm weights.zip
echo "Done."