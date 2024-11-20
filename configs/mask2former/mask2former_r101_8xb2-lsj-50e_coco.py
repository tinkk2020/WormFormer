#_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco.py']
_base_ = ['./mask2former_r50_8xb2-lsj-50e_coco_th.py']
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
work_dir = '/disk1/tenghui/work_dirs/2024/csb-1/mask2former_r101_8xb2-lsj-50e_coco_4gpus_120000'
