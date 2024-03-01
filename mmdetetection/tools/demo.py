"""
Demo code

Example usage:

python3 tools/demo.py configs/smpl/tune.py ./demo/raw_teaser.png --ckpt /path/to/model
"""
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)

from mmcv import Config
from mmcv.runner import Runner

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import build_optimizer
from mmdet.models.utils.smpl.renderer import Renderer
from mmdet import __version__
from mmdet.models import build_detector
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets.utils import to_tensor
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
                        np.array([0.485, 0.456, 0.406])[None, None,]

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    # 
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points,
    points = torch.einsum('bij,bkj->bki', rotation, points)
    # 
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion,
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def renderer_bv(img_t, verts_t, trans_t, bboxes_t, focal_length, render):
    R_bv = torch.zeros(3, 3)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1
    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
    area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]
    verts_t = verts_t + trans_t.unsqueeze(1)
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    # trans_tr = torch.einsum('bj,kj->bk', trans_t, R_bv)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    h, w = img_t.shape[-2:]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    img_right = render([torch.ones_like(img_t)], [verts_right],
                       translation=[torch.zeros_like(trans_t)])
    return img_right[0]


def prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH):
    verts = pred_results['pred_vertices'] + pred_results['pred_translation'][:, None]
    # 'pred_rotmat', 'pred_betas', 'pred_camera', 'pred_vertices', 'pred_joints', 'pred_translation', 'bboxes'
    pred_trans = pred_results['pred_translation'].cpu()
    pred_camera = pred_results['pred_camera'].cpu()
    pred_betas = pred_results['pred_betas'].cpu()
    pred_rotmat = pred_results['pred_rotmat'].cpu()
    pred_verts = pred_results['pred_vertices'].cpu()
    bboxes = pred_results['bboxes']
    img_bbox = img.copy()
    for bbox in bboxes:
        img_bbox = cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    img_th = torch.tensor(img_bbox.transpose([2, 0, 1]))
    _, H, W = img_th.shape
    try:
        fv_rendered = render([img_th.clone()], [pred_verts], translation=[pred_trans])[0]
        bv_rendered = renderer_bv(img_th, pred_verts, pred_trans, bbox_results[0], FOCAL_LENGTH, render)
    except Exception as e:
        print(e)
        return None

    # total_img = np.zeros((3 * H, W, 3))
    # total_img[:H] += img
    # total_img[H:2 * H] += fv_rendered.transpose([1, 2, 0])
    # total_img[2 * H:] += bv_rendered.transpose([1, 2, 0])
    # total_img = (total_img * 255).astype(np.uint8)

    # only for render
    render_image = np.zeros((H, W, 3))
    render_image += fv_rendered.transpose([1, 2, 0])
    # render_image += bv_rendered.transpose([1, 2, 0])
    render_image = (render_image * 255).astype(np.uint8)

    #total_img = (img * 255).astype(np.uint8)
#    total_img = img.copy()
#    point_size = 3
#    print_color = (0, 0, 255)
#    thickness = -1
#    batch_size = pred_joints.shape[0]
#
#    rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
#    FOCAL_LENGTH = 1000
#    # bboxes_size = np.maximum(np.abs(np.array(bboxes[:, 0]) - np.array(bboxes[:, 2])),
#    #                      np.abs(np.array(bboxes[:, 1]) - np.array(bboxes[:, 3])))
#    bboxes = torch.from_numpy(bboxes).cpu()
#    bboxes_size = torch.max(torch.abs(bboxes[:, 0] - bboxes[:, 2]),
#                             torch.abs(bboxes[:, 1] - bboxes[:, 3]))
#    # pre_camera = [s, r, t]
#    depth = 2 * FOCAL_LENGTH / (1e-6 + pred_camera[..., 0] * bboxes_size)
#    translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(
#        pred_joints.device)
#    center_pts = (bboxes[..., :2] + bboxes[..., 2:4]) / 2

#    translation[:, :-1] = depth[:, None] * \
#                          (center_pts + pred_camera[:, 1:] * bboxes_size.unsqueeze(-1) - H / 2) / FOCAL_LENGTH
#    translation[:, -1] = depth
#
#    focal_length = FOCAL_LENGTH * torch.ones_like(depth)
#    pred_keypoints_2d_smpl = perspective_projection(pred_joints,
#                                                    rotation_Is,
#                                                    translation,
#                                                    focal_length,
#                                                    H / 2)
#    valid_boxes = (torch.abs(bboxes[..., 0] - bboxes[..., 2]) > 5) & (torch.abs(
#        bboxes[..., 1] - bboxes[..., 3]) > 5)
#    batch_size = pred_joints.shape[0]
#    img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
#    img_size += torch.tensor(img.shape[1:], dtype=img_size.dtype).to(img_size.device)
#    #pred_keypoints_2d_smpl = pred_keypoints_2d_smpl / torch.tensor(img_size.unsqueeze(1))
#    total_img = (total_img * 255).astype(np.uint8)
#    for person in pred_keypoints_2d_smpl:
#     for joints in person:
#        #joints = joints + pred_trans[:, None, ]
#        cv2.circle(total_img, (int(joints[0]), int(joints[1])), point_size, print_color, thickness)

    return render_image

    #/mnt/8T/lyk/projects/work_dirs/fine_tune/branch2-pre3/

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='train config file path',default='./configs/smpl/tune.py')
    parser.add_argument('--image_folder', help='Path to folder with images',default='/mnt/8T/lyk/panda/tools/part_picture/10_Huaqiangbei/SEQ_10_001.jpg/images/'
    #'/mnt/8T/lyk/projects/demo_images/',  
    # SEQ_10_033.jpg/
    )
    parser.add_argument('--output_folder', default= '/mnt/8T/lyk/projects/final_result/forth/fail/', 
    help='Path to save results')
    parser.add_argument('--ckpt', type=str, default='/mnt/8T/lyk/projects/work_dirs/forth/branch2-pre3-coco/epoch_6.pth')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.ckpt:
        cfg.resume_from = args.ckpt

    cfg.test_cfg.rcnn.score_thr = 0.5

    FOCAL_LENGTH = cfg.get('FOCAL_LENGTH', 1000)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=('Human',))
    # add an attribute for visualization convenience
    model.CLASSES = ('Human',)

    model = MMDataParallel(model, device_ids=[0]).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = Runner(model, lambda x: x, optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.resume(cfg.resume_from)
    model = runner.model
    model.eval()
    render = Renderer(focal_length=FOCAL_LENGTH)
    img_transform = ImageTransform(
            size_divisor=32, **img_norm_cfg)
    img_scale0 = cfg.common_val_cfg.img_scale[0]
    img_scale1 = (286, 176)
    img_scale2 = (208, 128)
    # # img_scale=[(832, 512),(286, 176)],
    # img_scale=[(832, 512),(208, 128)],

    with torch.no_grad():
        folder_name = args.image_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        images = os.listdir(folder_name)
        for image in images:
            file_name = osp.join(folder_name, image)
            img = cv2.imread(file_name)
            ori_shape = img.shape
            img0, img_shape0, pad_shape0, scale_factor0 = img_transform(img, img_scale0)
            img, img_shape1, pad_shape1, scale_factor1 = img_transform(img, img_scale1, img_scale0)

            # Force padding for the issue of multi-GPU training
            padded_img = np.zeros((img.shape[0], img_scale0[1], img_scale0[0]), dtype=img.dtype)
            padded_img[:, :img.shape[-2], :img.shape[-1]] = img
            img = padded_img

            assert img.shape[1] == 512 and img.shape[2] == 832, "Image shape incorrect"

            data_batch = dict(
                img=DC([to_tensor(img[None, ...])], stack=True),
                img_meta=DC([{'img_shape':img_shape0, 'scale_factor':scale_factor0, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True),
                idx = 1
                )

            bbox_results, pred_results = model(**data_batch, return_loss=False)

            if pred_results is not None:
                pred_results['bboxes'] = bbox_results[0]
                img = denormalize(img)
                img_viz = prepare_dump(pred_results, img, render, bbox_results, FOCAL_LENGTH)
                cv2.imwrite(f'{file_name.replace(folder_name, output_folder)}.output.jpg', img_viz[:, :, ::-1])
                #cv2.imwrite(f'{file_name.replace(folder_name, output_folder)}.output.jpg', img)


if __name__ == '__main__':
    main()
