python src/main.py --scrape --dataset how2sign
cd how2sign
wget https://raw.githubusercontent.com/how2sign/how2sign.github.io/main/download_how2sign.sh
chmod +x download_how2sign.sh


Process OpenASL Dataset:
bash
python process_openasl.py --tsv_path data/raw/openasl/tsv/openasl-v1.0.tsv --bbox_path data/raw/openasl/bbox/bbox-v1.0.json --videos_dir data/raw/openasl/videos --output_dir data/processed/keypoints
Extract Features:
bash
python feature_extraction.py --tsv_path data/raw/openasl/tsv/openasl-v1.0.tsv --videos_dir data/raw/openasl/videos --keypoints_dir data/processed/keypoints --output_dir data/processed/features

# For setup
python src/run_pipeline.py --mode emergency

# For generating sign language
python run_pipeline.py --mode generate --text "Hello, how are you?" --output_file data/output/hello.npy

# For visualizing
python run_pipeline.py --mode visualize --pose_file data/output/hello.npy --animation_file data/output/hello.gif
