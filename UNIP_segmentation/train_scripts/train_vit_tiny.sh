model_path="mae_large_layer18_distill_unip_tiny_100ep_infmix.pth"
model_name="unip_tiny_by_mae_large"

log_dir="your_log_dir"
output_dir="your_output_dir"


# fine-tuning
CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/fine_tuning/upernet_vit_tiny_12_512_slide_14k_soda.py 2 --seed 0 --options \
    model.pretrained=$model_path \
    optimizer.paramwise_cfg.layer_decay_rate=0.85 model.backbone.use_rel_pos_bias=False \
    model.backbone.out_indices=[11,11,11,11] \
    log_config.hooks.1.log_dir=$log_dir/ft/SODA/$model_name \
    --work-dir $output_dir/ft/SODA/$model_name

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/fine_tuning/upernet_vit_tiny_12_512_slide_10k_mfnet.py 2 --seed 0 --options \
    model.pretrained=$model_path \
    optimizer.paramwise_cfg.layer_decay_rate=0.85 model.backbone.use_rel_pos_bias=False \
    model.backbone.out_indices=[11,11,11,11] \
    log_config.hooks.1.log_dir=$log_dir/ft/MFNet/$model_name \
    --work-dir $output_dir/ft/MFNet/$model_name

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/fine_tuning/upernet_vit_tiny_12_512_slide_17k_scut.py 2 --seed 0 --options \
    model.pretrained=$model_path \
    optimizer.paramwise_cfg.layer_decay_rate=0.85 model.backbone.use_rel_pos_bias=False \
    model.backbone.out_indices=[11,11,11,11] \
    log_config.hooks.1.log_dir=$log_dir/ft/SCUT/$model_name \
    --work-dir $output_dir/ft/SCUT/$model_name


# linear probing
CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/linear_probing/linear_vit_tiny_12_512_slide_14k_soda.py 2 --seed 0 --options \
    model.pretrained=$model_path \
    log_config.hooks.1.log_dir=$log_dir/lp/SODA/$model_name \
    --work-dir $output_dir/lp/SODA/$model_name

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/linear_probing/linear_vit_tiny_12_512_slide_10k_mfnet.py 2 --seed 0 --options \
    model.pretrained=$model_path \
    log_config.hooks.1.log_dir=$log_dir/lp/MFNet/$model_name \
    --work-dir $output_dir/lp/MFNet/$model_name

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/linear_probing/linear_vit_tiny_12_512_slide_17k_scut.py 2 --seed 0 --options \
    model.pretrained=$model_path \
    log_config.hooks.1.log_dir=$log_dir/lp/SCUT/$model_name \
    --work-dir $output_dir/lp/SCUT/$model_name