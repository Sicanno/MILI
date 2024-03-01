from __future__ import division

import argparse
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)

import torch

torch.multiprocessing.set_sharing_strategy('file_system')

print(sys.executable)
#import warnings
#warnings.filterwarnings('error')

from mmcv import Config
from mmdetection.mmdet import __version__
from mmdetection.mmdet.datasets import get_dataset
from mmdetection.mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed, train_smpl_detector_fuse, train_adv_smpl_detector)
from mmdetection.mmdet.models import build_detector
from mmdetection.mmdet.datasets.concat_dataset import ConcatDataset
from mmdet.datasets import build_dataloader, build_dataloader_fuse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default=PROJECT_PATH+'/mmdetection/configs/smpl/tune.py',help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--create_dummy',action='store_true',
                        help='Create a dummy checkpoint for recursive training on clusters')
    parser.add_argument('--load_pretrain', type=str, default=None,
                        help='Load parameters pretrained model and save it for recursive training')
    parser.add_argument('--imgs_per_gpu', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.imgs_per_gpu > 0:
        cfg.data.imgs_per_gpu = args.imgs_per_gpu

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    cfg.work_dir='/mnt/8T/lyk/projects/work_dirs/temp/forth/'

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
   
    train_dataset = get_dataset(cfg.datasets[0].train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    datasets = list()
    for flow in cfg.workflow:
        mode, epoches = flow
        cur_datasets = list()
        for dataset_cfg in cfg.datasets:
            if hasattr(dataset_cfg, mode):
                cur_datasets.append(get_dataset(getattr(dataset_cfg, mode)))
        datasets.append(ConcatDataset(cur_datasets))
    val_dataset = None
    if cfg.data.train.get('val_every', None):
        val_dataset = list()
        for dataset_cfg in cfg.datasets:
            if hasattr(dataset_cfg, 'val'):
                val_dataset.append(get_dataset(dataset_cfg.val))
        val_dataset = ConcatDataset(val_dataset)
    train_smpl_detector_fuse(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=args.validate,
                logger=logger,
                create_dummy=args.create_dummy,
                val_dataset=val_dataset,
                load_pretrain=args.load_pretrain
            )

# def parse_losses(losses, tag_tail=''):
#     log_vars = OrderedDict()
#     for loss_name, loss_value in losses.items():
#         if not ('loss' in loss_name):
#             losses[loss_name] = loss_value.cpu()

#     if 'img$idxs_in_batch' in losses:
#         last_idx = -1
#         split_idx = -1
#         for i, idx in enumerate(losses['img$idxs_in_batch'].squeeze(1)):
#             if last_idx > idx:
#                 split_idx = i
#                 break
#             else:
#                 last_idx = idx
#         split_idx = int(split_idx)
#         if split_idx > 0:
#             for loss_name, loss_value in losses.items():
#                 if loss_name.startswith('img$') and loss_name != 'img$raw_images':
#                     losses[loss_name] = losses[loss_name][:split_idx]

#     for loss_name, loss_value in losses.items():
#         # To avoid stats pollution for validation inside training epoch.
#         loss_name = f'{loss_name}/{tag_tail}'
#         if loss_name.startswith('img$'):
#             log_vars[loss_name] = loss_value
#             continue
#         if isinstance(loss_value, torch.Tensor):
#             log_vars[loss_name] = loss_value.mean()
#         elif isinstance(loss_value, list):
#             log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
#         else:
#             raise TypeError(
#                 '{} is not a tensor or list of tensors'.format(loss_name))

#     loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
#     # TODO: Make the code more elegant here.
#     log_vars[f'loss/{tag_tail}'] = loss
#     for name in log_vars:
#         if not name.startswith('img$'):
#             log_vars[name] = log_vars[name].item()

#     return loss, log_vars

# def batch_processor(model, data, mode, **kwargs):
#     # NOTE: The mode is str instead of boolean now.
#     losses = model(**data)
#     tag_tail = mode
#     loss, log_vars = parse_losses(losses, tag_tail)
#     #log_total = OrderedDict()
#     # loss_total = losses[-1]
#     # losses = losses[:-1].clone()
#     # if type(losses) == list:
#     #     for each_num in range(len(losses)):
#     #         loss, log_vars = parse_losses(losses[each_num], tag_tail, each_num)
#     #         loss_total += loss
#     #         log_total.update(log_vars)
#     outputs = dict(
#         loss=loss , log_vars=log_vars, num_samples=len(data['img'].data))
#     return outputs

# def my_train_smpl_detector_fuse(model, datasets, cfg, **kwargs):
#     # prepare data loaders
#     data_loaders = [
#         build_dataloader_fuse(
#             dataset,
#             cfg.data.imgs_per_gpu,
#             cfg.data.workers_per_gpu,
#             cfg.gpus,
#             dist=False) for dataset in datasets
#     ]
#     # #put model on gpus
#     # device = [i+1 for i in range(cfg.gpus)]
#     # torch.cuda.set_device(1)
#     # model = MMDataParallel(model, device_ids=device).cuda()
#     device = [i for i in range(cfg.gpus)]
#     # model = MMDataParallel(model, device_ids=device)
#     # model = model.cuda()
#     # put model on gpus
#     #torch.cuda.set_device(0)
#     model = MMDataParallel(model, device_ids=device).cuda()
#         # build runner
#     optimizer = build_optimizer(model, cfg.optimizer)
#     runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
#                     cfg.log_level)
#     # fp16 setting
#     fp16_cfg = cfg.get('fp16', None)
#     if fp16_cfg is not None:
#         optimizer_config = Fp16OptimizerHook(
#             **cfg.optimizer_config, **fp16_cfg, distributed=False)
#     else:
#         optimizer_config = cfg.optimizer_config
#     # TODO: Build up a logger here that inherit the hook class
#     runner.register_training_hooks(cfg.lr_config, optimizer_config,
#                                    cfg.checkpoint_config, cfg.log_config)

#     val_dataset_cfg = cfg.data.val
#     eval_cfg = cfg.get('evaluation', {})

#     pretrain_path = kwargs.get('load_pretrain', None)
#     if kwargs.get('load_pretrain', None):
#         print(f"Load pretrained model from {pretrain_path}")
#         runner._epoch -= 1
#         runner.load_checkpoint(pretrain_path)
#         # torch.save(model.state_dict(),'/mnt/8T/lyk/projects/work_dirs/temp/forth/epoch_43.pth',_use_new_zipfile_serialization=False)
#         now_save_checkpoint(cfg.work_dir, filename_tmpl='pretrained_{}.pth')

#     # torch.save(model.state_dict(),'/mnt/8T/lyk/projects/work_dirs/temp/epoch_43.pth',_use_new_zipfile_serialization=False)

#     # runner.save_checkpoint(cfg.work_dir, filename_tmpl='pretrained_{}.pth')

# def now_save_checkpoint(self,
#                         out_dir,
#                         filename_tmpl='epoch_{}.pth',
#                         save_optimizer=True,
#                         meta=None):
#         if meta is None:
#             meta = dict(epoch=self.epoch + 1, iter=self.iter)
#         else:
#             meta.update(epoch=self.epoch + 1, iter=self.iter)

#         filename = filename_tmpl.format(self.epoch + 1)
#         filepath = osp.join(out_dir, filename)
#         linkpath = osp.join(out_dir, 'latest.pth')
#         optimizer = self.optimizer if save_optimizer else None
#         save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
#         # use relative symlink
#         mmcv.symlink(filename, linkpath)

# def save_checkpoint(model, filename, optimizer=None, meta=None):
#     """Save checkpoint to file.

#     The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
#     ``optimizer``. By default ``meta`` will contain version and time info.

#     Args:
#         model (Module): Module whose params are to be saved.
#         filename (str): Checkpoint filename.
#         optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
#         meta (dict, optional): Metadata to be saved in checkpoint.
#     """
#     if meta is None:
#         meta = {}
#     elif not isinstance(meta, dict):
#         raise TypeError('meta must be a dict or None, but got {}'.format(
#             type(meta)))
#     meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

#     mmcv.mkdir_or_exist(osp.dirname(filename))
#     if hasattr(model, 'module'):
#         model = model.module

#     checkpoint = {
#         'meta': meta,
#         'state_dict': weights_to_cpu(model.state_dict())
#     }
#     if optimizer is not None:
#         checkpoint['optimizer'] = optimizer.state_dict()

#     torch.save(checkpoint, filename,_use_new_zipfile_serialization=False)

# def weights_to_cpu(state_dict):
#     """Copy a model state_dict to cpu.

#     Args:
#         state_dict (OrderedDict): Model weights on GPU.

#     Returns:
#         OrderedDict: Model weights on GPU.
#     """
#     state_dict_cpu = OrderedDict()
#     for key, val in state_dict.items():
#         state_dict_cpu[key] = val.cpu()
#     return state_dict_cpu

# def build_optimizer(model, optimizer_cfg):
#     """Build optimizer from configs.

#     Args:
#         model (:obj:`nn.Module`): The model with parameters to be optimized.
#         optimizer_cfg (dict): The config dict of the optimizer.
#             Positional fields are:
#                 - type: class name of the optimizer.
#                 - lr: base learning rate.
#             Optional fields are:
#                 - any arguments of the corresponding optimizer type, e.g.,
#                   weight_decay, momentum, etc.
#                 - paramwise_options: a dict with 3 accepted fileds
#                   (bias_lr_mult, bias_decay_mult, norm_decay_mult).
#                   `bias_lr_mult` and `bias_decay_mult` will be multiplied to
#                   the lr and weight decay respectively for all bias parameters
#                   (except for the normalization layers), and
#                   `norm_decay_mult` will be multiplied to the weight decay
#                   for all weight and bias parameters of normalization layers.

#     Returns:
#         torch.optim.Optimizer: The initialized optimizer.
#     """
#     if hasattr(model, 'module'):
#         model = model.module

#     optimizer_cfg = optimizer_cfg.copy()
#     paramwise_options = optimizer_cfg.pop('paramwise_options', None)
#     # if no paramwise option is specified, just use the global setting
#     if paramwise_options is None:
#         return obj_from_dict(optimizer_cfg, torch.optim,
#                              dict(params=model.parameters()))
#     else:
#         assert isinstance(paramwise_options, dict)
#         # get base lr and weight decay
#         base_lr = optimizer_cfg['lr']
#         base_wd = optimizer_cfg.get('weight_decay', None)
#         # weight_decay must be explicitly specified if mult is specified
#         if ('bias_decay_mult' in paramwise_options
#                 or 'norm_decay_mult' in paramwise_options):
#             assert base_wd is not None
#         # get param-wise options
#         bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
#         bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
#         norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
#         # set param-wise lr and weight decay
#         params = []
#         for name, param in model.named_parameters():
#             param_group = {'params': [param]}
#             if not param.requires_grad:
#                 # FP16 training needs to copy gradient/weight between master
#                 # weight copy and model weight, it is convenient to keep all
#                 # parameters here to align with model.parameters()
#                 params.append(param_group)
#                 continue

#             # for norm layers, overwrite the weight decay of weight and bias
#             # TODO: obtain the norm layer prefixes dynamically
#             if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
#                 if base_wd is not None:
#                     param_group['weight_decay'] = base_wd * norm_decay_mult
#             # for other layers, overwrite both lr and weight decay of bias
#             elif name.endswith('.bias'):
#                 param_group['lr'] = base_lr * bias_lr_mult
#                 if base_wd is not None:
#                     param_group['weight_decay'] = base_wd * bias_decay_mult
#             # otherwise use the global settings

#             params.append(param_group)

#         if issubclass(optimizer_cfg['type'], torch.optim.Optimizer):
#             optimizer_cls = optimizer_cfg.pop('type')
#         else:
#             optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
#         return optimizer_cls(params, **optimizer_cfg)

if __name__ == '__main__':
    main()
