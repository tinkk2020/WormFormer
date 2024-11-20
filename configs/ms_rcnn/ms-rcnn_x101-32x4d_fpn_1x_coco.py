_base_ = './ms-rcnn_r50_fpn_1x_coco.py'

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

model = dict(
data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))
# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
dataset_type = 'CocoDataset'
data_root='/disk2/tan/Data/csb-1_2x2_crop/'
train_dataloader = dict(
    #batch_size=4,
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_train.json',
        ann_file=data_root + 'annotations/instances_train.json',
        # data_prefix=dict(img=data_root+'init_image/train/'),
        data_prefix=dict(img=data_root+'images/'),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_val.json',
        ann_file=data_root + 'annotations/instances_val.json',
        # data_prefix=dict(img=data_root+'init_image/val/'),
        data_prefix=dict(img=data_root+'images/'),
        pipeline=test_pipeline))

test_dataloader  = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_test.json',
        ann_file=data_root + 'annotations/instances_test.json',
        # data_prefix=dict(img=data_root+'init_image/test/'),
        data_prefix=dict(img=data_root+'images/'),
        pipeline=test_pipeline))

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    # ann_file=data_root + 'annotations/new_instances_val.json',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    # ann_file=data_root + 'annotations/new_instances_test.json',
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})




auto_scale_lr = dict(enable=False, base_batch_size=16)
# auto_scale_lr = dict(enable=True, base_batch_size=16)
#resume=True#从断点继续训练
work_dir = '/disk2/tan/work_dirs/2024/csb-1_2x2_crop/ms-rcnn_r101_3gpus'