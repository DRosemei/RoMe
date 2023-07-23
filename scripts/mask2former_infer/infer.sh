export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="~/code/Mask2Former"
python inference.py  --config-file ../configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml \
 --opts MODEL.WEIGHTS ../mask2former_mapillary_vistas_swin_L.pkl
