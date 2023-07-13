export WANDB_BASE_URL="######:8080"
export WANDB_API_KEY="local-######"
# export WANDB_MODE="offline"
export PYTHONPATH=${PWD}
export CUDA_VISIBLE_DEVICES=1

# python3 scripts/train.py --config configs/local_nusc_mini.yaml
python3 scripts/train.py --config configs/local_nusc.yaml
# python3 scripts/train.py --config configs/local_kitti.yaml