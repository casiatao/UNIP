_base_ = [
    '../_base_/models/upernet.py', '../_base_/datasets/mfnet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
find_unused_parameters = True
model = dict(
    pretrained=None,
    backbone=dict(
        type='MAE',
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        in_chans=3, 
        init_values=1.0,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23],
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        num_classes=9,
        channels=1024,
    ),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=9,
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
    
)

optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.65))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


data=dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=9800)
checkpoint_config = dict(by_epoch=False, interval=4900, max_keep_ckpts=1)
evaluation = dict(interval=700, metric='mIoU')

