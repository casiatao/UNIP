_base_ = [
    '../_base_/datasets/mfnet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        img_size=[512],
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_fpn=False,
        frozen_stages=12,
        out_indices=[3, 5, 7, 11]
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=[192, 192, 192, 192],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=768,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)


optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1e-6, 
                 power=1.0, min_lr=0.0, by_epoch=False)


data=dict(samples_per_gpu=4)

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

