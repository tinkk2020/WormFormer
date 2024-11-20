_base_ = './cascade-mask-rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
work_dir = '/disk2/tan/work_dirs/2024/csb-1_2x2_crop/cascade_r101_3gpus'