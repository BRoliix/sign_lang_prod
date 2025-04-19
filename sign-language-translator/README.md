python src/main.py --scrape --dataset how2sign
cd how2sign
wget https://raw.githubusercontent.com/how2sign/how2sign.github.io/main/download_how2sign.sh
chmod +x download_how2sign.sh

python src/main.py --process
python src/main.py --extract
python src/main.py --train
python src/main.py --translate "Hello, how are you?" --visualize
