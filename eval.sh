export WANDB_BASE_URL="######:8080"
export WANDB_API_KEY="local-######"
export WANDB_MODE="offline"
export PYTHONPATH=${PWD}
export CUDA_VISIBLE_DEVICES=0

python3 scripts/eval.py --config configs/nusc_eval.yaml