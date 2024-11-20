_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='SOLOv2',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=80,
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01), clip_grad=dict(max_norm=35, norm_type=2))

# val_evaluator = dict(metric='segm')
# test_evaluator = val_evaluator


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
#
# data_root = '/disk1/tenghui/Data/csb-1_dataset/'
# train_dataloader = dict(
#     #batch_size=4,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_annotations/train_sketchPoints.json',
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_annotations/val_sketchPoints.json',
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=test_pipeline))
#
# test_dataloader  = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_annotations/test_sketchPoints.json',
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=test_pipeline))
#
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'coco_annotations/val_sketchPoints.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'coco_annotations/test_sketchPoints.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})


# data_root = '/disk1/liye/mmdetection-main/data/bbbc/'
# train_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/sketchPoints_instances_train_16points.json',
#         data_prefix=dict(img=data_root+'train/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/sketchPoints_instances_val_16points.json',
#         data_prefix=dict(img=data_root+'val/'),
#         pipeline=test_pipeline))
#
# test_dataloader  = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/sketchPoints_instances_test_16points.json',
#         data_prefix=dict(img=data_root+'test/'),
#         pipeline=test_pipeline))
#
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/sketchPoints_instances_val_16points.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/sketchPoints_instances_test_16points.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})

# optimizer
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)


# learning policy
max_iters =80000#368750
# param_scheduler = dict(
#     type='MultiStepLR',
#     begin=0,
#     end=max_iters,
#     by_epoch=False,
#     milestones=[int(0.8889*max_iters), int(0.9630*max_iters)],#[327778, 355092],
#     gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval=2000##5000
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
auto_scale_lr = dict(enable=False, base_batch_size=16)


# vis_backends = [dict(type='LocalVisBackend'),
#                 dict(type='TensorboardVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
#resume=True#从断点继续训练
#work_dir = '/disk1/tenghui/work_dirs/2024/csb-1/16points/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_4gpus'
#work_dir = '/disk1/tenghui/work_dirs/2024/BBBC/16points/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_4gpus_shuffle'
# work_dir = '/disk1/tenghui/work_dirs/2024/HAEcrop/16points/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_4gpus'
# work_dir = '/disk1/tenghui/work_dirs/2024/csb-1/32pointsdisorder/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_3gpus'
work_dir = '/disk2/tan/work_dirs/2024/csb-1_2x2_crop/solov2_r50_3gpus'