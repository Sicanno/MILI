# model settings
from mmdetection.mmdet.core.utils.smpl_tensorboard import SMPLBoard
import os.path as osp
from mmdetection.mmdet.core.utils.radam import RAdam
from mmdetection.mmdet.core.utils.lr_hooks import SequenceLrUpdaterHook, PowerLrUpdaterHook
import math

WITH_NR = False
FOCAL_LENGTH = 1000
model = dict(
    type='RSC_RCNN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        # Maybe I should change it to 1 to boost training, but its' better to leave it unchanged now.
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    smpl_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    smpl_head=dict(
        type='SMPLHead',
        in_size=14,
        in_channels=256,
        loss_cfg=dict(type='SMPLLoss', normalize_kpts=True,
                      adversarial_cfg=False,
                      robust_sdf=20., use_sdf=False,
                      FOCAL_LENGTH=FOCAL_LENGTH,
                      img_size=(832, 512),
                      ),
    ),
    smpl_weight=1,
)

re_weight = {'loss_disc': 1 / 60., 'adv_loss_fake': 1 / 60., 'adv_loss_real': 1 / 60.}
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
square_bbox = False
common_train_cfg = dict(
    #img_scale=[(832, 512), (416, 256), (208, 128), (104, 64)],
    img_scale=[(832, 512), (650, 400), (286, 176)],
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0.5,
    # noise_factor=1e-3,  # To avoid color jitter.
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    # max_samples=1024
    square_bbox=square_bbox,
    mosh_path='data/h36m/extras/h36m_single_train_openpose.npz',
    with_nr=WITH_NR,
    use_poly=True,
    # rot_factor=30,
)
common_val_cfg = dict(
    #img_scale=[(832, 512), (416, 256), (208, 128), (104, 64)],
    img_scale=[(832, 512), (650, 400), (286, 176)],
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0,
    noise_factor=1e-3,  # To avoid color jitter.
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    max_samples=64,
    square_bbox=square_bbox,
    mosh_path='data/h36m/extras/h36m_single_train_openpose.npz',
    with_nr=WITH_NR,
    use_poly=True,
)

h36m_dataset_type = 'H36MDataset'
h36m_data_root = '/mnt/2020/2020/multi/data/human3.6m/'
coco_dataset_type = 'COCOKeypoints'
coco_data_train_root = '/mnt/2020/2020/multi/data/coco_train2014/'
coco_data_val_root = '/mnt/2020/2020/multi/data/coco_val2014/'

common_dataset = 'CommonDataset'

panoptic_ann_root = '/mnt/2020/2020/multi/co/data_test/Panoptic/panoptic_annot/'

rcnn_pretrain_root = '/mnt/2020/2020/multi/train_anno/'
image_root = '/mnt/2020/2020/multi/data/'
ann_root = '/mnt/2020/2020/multi/train_anno/'

datasets = [
    dict(
        train=dict(
            type=h36m_dataset_type,
            #type=common_dataset,
            ann_file=ann_root + 'h36m/'+'rcnn/h3.6m_smpl.pkl',
            img_prefix=h36m_data_root,
            sample_weight=0.6,
            **common_train_cfg
        ),
        val=dict(
            type=h36m_dataset_type,
            #type=common_dataset,
            ann_file=ann_root + 'h36m/'+'rcnn/val.pkl',
            img_prefix=h36m_data_root,
            sample_weight=0.6,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=ann_root + 'coco/temp/'+'train_densepose_2014_depth_nocrowd.pkl',
            #ann_file=ann_root + 'coco/temp/'+ 'single.pkl',
            img_prefix= coco_data_train_root,
            sample_weight=0.3,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=ann_root + 'coco/temp/'+'val_densepose_2014_depth_nocrowd.pkl',
            img_prefix= coco_data_val_root,
            sample_weight=0.3,
            **common_val_cfg
        ),
    ),
    # dict(
    #     train=dict(
    #         type=common_dataset,
    #         ann_file=pose_track_root + 'rcnn/train.pkl',
    #         img_prefix=pose_track_root,
    #         sample_weight=0.3,
    #         **common_train_cfg
    #     ),
    #     val=dict(
    #         type=common_dataset,
    #         ann_file=pose_track_root + 'rcnn/val.pkl',
    #         img_prefix=pose_track_root,
    #         sample_weight=0.3,
    #         **common_val_cfg
    #     ),
    # ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file= ann_root + 'mpii/'+ 'rcnn/train.pkl',
            img_prefix=image_root + 'mpii_v1/'+'images/',
            sample_weight=0.3,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file= ann_root + 'mpii/' + 'rcnn/val.pkl',
            img_prefix=image_root + 'mpii_v1/'+'images/',
            sample_weight=0.3,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=h36m_dataset_type,
            #ann_file=image_root + 'mpi_inf_3dhp/'+'rcnn/single.pkl',
            ann_file = ann_root + 'mpi_inf_3dhp/' + 'mpi_smpl.pkl',
            img_prefix=image_root+'mpi_inf_3dhp',
            sample_weight=0.1,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=image_root + 'mpi_inf_3dhp/'+'rcnn/val.pkl',
            img_prefix=image_root+'mpi_inf_3dhp',
            sample_weight=0.1,
            ignore_3d=True,
            **common_val_cfg
        ),
    ),
    # dict(
    #     val=dict(
    #         type=common_dataset,
    #         ann_file=panoptic_root + 'processed/annotations/160422_ultimatum1.pkl',
    #         img_prefix=panoptic_root,
    #         sample_weight=0.6,
    #         ignore_3d=True,
    #         **common_val_cfg
    #     ),
    # ),
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=common_train_cfg,
    val=common_val_cfg,
)
# optimizer
optimizer = dict(type=RAdam, lr=1e-5, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

adv_optimizer = dict(type=RAdam, lr=1e-5, weight_decay=0.0001)
adv_optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = SequenceLrUpdaterHook(
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    seq=[1e-5]
)
checkpoint_config = dict(interval=1)
# yapf:disable
# runtime settings
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/mnt/8T/coco/projects/work_dirs/4_contras/baseline'
load_from = None
resume_from = osp.join(work_dir, 'latest.pth')
workflow = [('train', 1), ('val', 1)]
#workflow = [('train', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type=SMPLBoard, log_dir=work_dir, bboxes_only=False, K_SMALLEST=1,
             detail_mode=False, FOCAL_LENGTH=FOCAL_LENGTH, )
    ])
evaluation = dict(interval=1)
# yapf:enable
fuse = True
time_limit = 1 * 3000  # In sceonds
log_grad = True
