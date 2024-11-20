_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-1x_coco.py'
# model settings
model = dict(
    type='PointRend',
backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        type='PointRendRoIHead',
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(
                _delete_=True, type='SimpleRoIAlign', output_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            _delete_=True,
            type='CoarseMaskHead',
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='MaskPointHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=80,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75)),
    test_cfg=dict(
        rcnn=dict(
            subdivision_steps=5,
            subdivision_num_points=28 * 28,
            scale_factor=2)))
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (1024, 1024)
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations_SketchPoints', with_bbox=True, with_mask=True),
    # dict(
    #     type='RandomCrop_SketchPoints',
    #     crop_size=(640,540),
    #     #crop_size=image_size,
    #     num_sketchPoints=16,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='RandomFlip_SketchPoints', prob=0.5),
    # large scale jittering
    dict(
        #RandomResize_localFeature
        type='RandomResize_SketchPoints',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop_SketchPoints',
        crop_size=image_size,
        num_sketchPoints=16,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations_SketchPoints', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs_SketchPoints')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize_SketchPoints', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations_SketchPoints', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs_SketchPoints',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_type = 'CocoDataset_SketchPoints'
# data_root = '/disk1/tenghui/Data/HAE/HAE_4×4_crop/'
data_root='/disk2/tan/Data/csb-1_2x2_crop/'
train_dataloader = dict(
    #batch_size=4,
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_train.json',
        ann_file=data_root + 'annotations/addsketch_instances_train.json',
        # data_prefix=dict(img=data_root+'init_image/train/'),
        data_prefix=dict(img=data_root+'images/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_val.json',
        ann_file=data_root + 'annotations/addsketch_instances_val.json',
        # data_prefix=dict(img=data_root+'init_image/val/'),
        data_prefix=dict(img=data_root+'images/'),
        pipeline=test_pipeline))

test_dataloader  = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_test.json',
        ann_file=data_root + 'annotations/addsketch_instances_test.json',
        # data_prefix=dict(img=data_root+'init_image/test/'),
        data_prefix=dict(img=data_root+'images/'),
        pipeline=test_pipeline))

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    # ann_file=data_root + 'annotations/new_instances_val.json',
    ann_file=data_root + 'annotations/addsketch_instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    # ann_file=data_root + 'annotations/new_instances_test.json',
    ann_file=data_root + 'annotations/addsketch_instances_test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0001,
#         weight_decay=0.05,
#         eps=1e-8,
#         betas=(0.9, 0.999)),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#             'query_embed': embed_multi,
#             'query_feat': embed_multi,
#             'level_embed': embed_multi,
#         },
#         norm_decay_mult=0.0),
#     clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters =80000#368750
# param_scheduler = [
#     # dict(
#     #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#     type='MultiStepLR',
#     begin=0,
#     end=max_iters,
#     by_epoch=False,
#     milestones=[int(0.8889*max_iters), int(0.9630*max_iters)],#[327778, 355092],
#     gamma=0.1)]

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval=1000##5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',#保存最优分割精度pth
        interval=interval),
    visualization=dict(type='DetVisualizationHook', draw=True)
    # logger=dict(
    #     type='LoggerHook',
    #     interval=50)
    )
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)#window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)

work_dir = '/disk2/tan/work_dirs/2024/csb-1_2x2_crop/point-rend_r101_3gpus'