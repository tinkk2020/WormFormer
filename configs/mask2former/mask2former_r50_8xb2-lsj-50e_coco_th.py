_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco-panoptic.py']

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
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

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
# dataset_type = 'CocoDataset_SketchPoints'
# data_root = '/disk1/tenghui/Data/HAE/HAE_4×4_crop/'
# train_dataloader = dict(
#     #batch_size=4,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/new_instances_train.json',
#         data_prefix=dict(img=data_root+'init_image/train/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/new_instances_val.json',
#         data_prefix=dict(img=data_root+'init_image/val/'),
#         pipeline=test_pipeline))
#
# test_dataloader  = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/new_instances_test.json',
#         data_prefix=dict(img=data_root+'init_image/test/'),
#         pipeline=test_pipeline))
#
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/new_instances_val.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/new_instances_test.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})

# data_root = '/disk1/liye/mmdetection-main/data/bbbc/'
# train_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/instances_train.json',
#         data_prefix=dict(img=data_root+'train/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/instances_val.json',
#         data_prefix=dict(img=data_root+'val/'),
#         pipeline=test_pipeline))
#
# test_dataloader  = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'annotations/instances_test.json',
#         data_prefix=dict(img=data_root+'test/'),
#         pipeline=test_pipeline))
#
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_val.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_test.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})

# data_root = '/disk1/tenghui/Data/csb-1_dataset/'
# train_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_annotations/train.json',
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_annotations/val.json',
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=test_pipeline))
#
# test_dataloader  = dict(
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root+'coco_annotations/test.json',
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=test_pipeline))
#
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'coco_annotations/val.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     ann_file=data_root + 'coco_annotations/test.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# data_root='/disk2/tan/Data/csb-1_2x2_crop/'
# train_dataloader = dict(
#     #batch_size=4,
#     dataset=dict(
#         type=dataset_type,
#         # ann_file=data_root+'annotations/new_instances_train.json',
#         ann_file=data_root + 'annotations/instances_train.json',
#         # data_prefix=dict(img=data_root+'init_image/train/'),
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         # ann_file=data_root+'annotations/new_instances_val.json',
#         ann_file=data_root + 'annotations/instances_val.json',
#         # data_prefix=dict(img=data_root+'init_image/val/'),
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=test_pipeline))
#
# test_dataloader  = dict(
#     dataset=dict(
#         type=dataset_type,
#         # ann_file=data_root+'annotations/new_instances_test.json',
#         ann_file=data_root + 'annotations/instances_test.json',
#         # data_prefix=dict(img=data_root+'init_image/test/'),
#         data_prefix=dict(img=data_root+'images/'),
#         pipeline=test_pipeline))
#
# val_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     # ann_file=data_root + 'annotations/new_instances_val.json',
#     ann_file=data_root + 'annotations/instances_val.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
# test_evaluator = dict(
#     _delete_=True,
#     type='CocoMetric',
#     # ann_file=data_root + 'annotations/new_instances_test.json',
#     ann_file=data_root + 'annotations/instances_test.json',
#     metric=['bbox', 'segm'],
#     format_only=False,
#     backend_args={{_base_.backend_args}})
dataset_type = 'CocoDataset_SketchPoints'
# data_root = '/disk1/tenghui/Data/HAE/HAE_4×4_crop/'
# data_root='/disk2/tan/Data/csb-1_2x2_crop/'
# data_root='/disk2/tan/Data/OrbitWorm/images1/'
data_root='/disk2/tan/Data/OrbitWorm/2nematodes_images/'
train_dataloader = dict(
    #batch_size=4,
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_train.json',
        # ann_file=data_root + 'annotations/addsketch_instances_train.json',
        # data_prefix=dict(img=data_root+'init_image/train/'),
        ann_file=data_root + 'annotations/train_2nematodes_4points.json',
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_val.json',
        # ann_file=data_root + 'annotations/addsketch_instances_val.json',
        # data_prefix=dict(img=data_root+'init_image/val/'),
        ann_file=data_root + 'annotations/val_2nematodes_4points.json',
        data_prefix=dict(img=data_root),
        pipeline=test_pipeline))

test_dataloader  = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_test.json',
        # ann_file=data_root + 'annotations/addsketch_instances_test.json',
        ann_file=data_root + 'annotations/test_2nematodes_4points.json',
        # data_prefix=dict(img=data_root+'init_image/test/'),
        data_prefix=dict(img=data_root),
        pipeline=test_pipeline))

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    # ann_file=data_root + 'annotations/new_instances_val.json',
    # ann_file=data_root + 'annotations/addsketch_instances_val.json',
ann_file=data_root + 'annotations/val_2nematodes_4points.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args={{_base_.backend_args}})
test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    # ann_file=data_root + 'annotations/new_instances_test.json',
    # ann_file=data_root + 'annotations/addsketch_instances_test.json',
    metric=['bbox', 'segm'],
ann_file=data_root + 'annotations/test_2nematodes_4points.json',
    format_only=False,
    backend_args={{_base_.backend_args}})


# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters =8000#368750
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[int(0.8889*max_iters), int(0.9630*max_iters)],#[327778, 355092],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 200##5000
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
# auto_scale_lr = dict(enable=True, base_batch_size=16)
#resume=True#从断点继续训练
#work_dir = '/disk1/tenghui/work_dirs/2024/csb-1/mask2former_r50_8xb2-lsj-50e_coco_4gpus_16batch_twice'
# work_dir = '/disk1/tenghui/work_dirs/2024/HAE_4×4_crop/mask2former_r50_8xb2-lsj-50e_coco'
# work_dir = '/disk1/tenghui/work_dirs/2024/HAEcrop/16points/mask2former_r50_8xb2-lsj-50e_coco_4gpus'
# work_dir = '/disk2/tan/work_dirs/2024/csb-1_2x2_crop/mask2former_r50_8xb2-lsj-50e_coco_3gpus_th'
work_dir = '/disk2/tan/work_dirs/orbitworm/th/mask2former_2nematodes_4gpus_changeinterval'