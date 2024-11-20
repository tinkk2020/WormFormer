_base_ = './htc-without-semantic_r50_fpn_1x_coco.py'
model = dict(
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            seg_scale_factor=1 / 8,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img='train2017/', seg='stuffthingmaps/train2017/'),
        pipeline=train_pipeline))
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (1024, 1024)


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
        ))

test_dataloader  = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=data_root+'annotations/new_instances_test.json',
        ann_file=data_root + 'annotations/addsketch_instances_test.json',
        # data_prefix=dict(img=data_root+'init_image/test/'),
        data_prefix=dict(img=data_root+'images/'),
        ))

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
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)


# vis_backends = [dict(type='LocalVisBackend'),
#                 dict(type='TensorboardVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
#resume=True#从断点继续训练
#work_dir = '/disk1/tenghui/work_dirs/2024/csb-1/16points/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_4gpus'
#work_dir = '/disk1/tenghui/work_dirs/2024/BBBC/16points/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_4gpus_shuffle'
# work_dir = '/disk1/tenghui/work_dirs/2024/HAEcrop/16points/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_4gpus'
# work_dir = '/disk1/tenghui/work_dirs/2024/csb-1/32pointsdisorder/mask2former_r50_8xb2-lsj-50e_coco_SketchPoints_3gpus'
work_dir = '/disk2/tan/work_dirs/2024/csb-1_2x2_crop/htc_r50_3gpus'