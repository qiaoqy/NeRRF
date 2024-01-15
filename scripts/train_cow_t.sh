CUDA_LAUNCH_BLOCKING=5 \
python train/train.py \
-n cow_transparent \
-c NeRRF.conf \
-D data/blender/transparent/cow \
--gamma 0.95 \
--gpu_id=0 \
--visual_path tet_visual \
--stage 2 \
--tet_scale 4.2 \
--sphere_radius 2.30 \
--ior 1.2 \
--enable_refr
# --resume \
# --ior 1.2 \
# --use_cone
# --use_sdf
# --use_progressive_encoder