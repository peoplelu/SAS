import os
import sys
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax
from tensorboardX import SummaryWriter

from MinkowskiEngine import SparseTensor
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, save_checkpoint, \
    export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature
from dataset.label_constants import *
from dataset.feature_loader import FusedFeatureLoader
from dataset.feature_loader import collation_fn_eval_all_offset as collation_fn
from dataset.point_loader import Point3DLoader, collation_fn_eval_all
from models.disnet import DisNet as Model
from tqdm import tqdm


best_iou = 0.0


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='SAS 3D distillation: Second Stage.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/distill_openseg.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/distill_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    
    # By default we use shared memory for training
    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    args.index_split = 0

    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda())
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as ex:
        # The model was trained in a parallel manner, so need to be loaded differently
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                # remove module
                k = k[7:]
            else:
                # add module
                k = 'module.' + k

            new_state_dict[k]=v
        model.load_state_dict(new_state_dict, strict=True)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=args.aug,
                                    memcache_init=args.use_shm, eval_all=False, loop=args.loop,
                                    input_color=args.input_color, return_sp=True, return_name=True,
                                    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=args.workers, pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last=True, collate_fn=collation_fn,
                                            worker_init_fn=worker_init_fn)
    if args.evaluate:
        val_data = Point3DLoader(datapath_prefix=args.data_root,
                                 voxel_size=args.voxel_size,
                                 split='val', aug=False,
                                 memcache_init=args.use_shm,
                                 eval_all=True,
                                 input_color=args.input_color)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                shuffle=False,
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    
        val_data_train = Point3DLoader(datapath_prefix=args.data_root,
                                 voxel_size=args.voxel_size,
                                 split='train', aug=False,
                                 memcache_init=args.use_shm,
                                 eval_all=True,
                                 input_color=args.input_color)
        val_sampler_train = torch.utils.data.distributed.DistributedSampler(
            val_data_train) if args.distributed else None
        val_loader_train = torch.utils.data.DataLoader(val_data_train, batch_size=args.batch_size_val,
                                                shuffle=False,
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler_train)

        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation

    # ####################### Distill ####################### #
    val_sampler_train.set_epoch(0)
    initialize_pred(val_loader_train, model, criterion)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        loss_train = distill(train_loader, model, optimizer, epoch)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train[0], epoch_log)
            writer.add_scalar('loss_train_1', loss_train[1], epoch_log)
            writer.add_scalar('loss_train_2', loss_train[2], epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion)
            # raise NotImplementedError

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(cfg=cfg)
    return model

def obtain_text_features_and_palette():
    '''obtain the CLIP text feature and palette.'''

    if 'scannet' in args.data_root:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other'
        palette = get_palette()
        dataset_name = 'scannet'
    elif 'matterport' in args.data_root:
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
        dataset_name = 'matterport'
    elif 'nuscenes' in args.data_root:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
        dataset_name = 'nuscenes'

    if not os.path.exists('saved_text_embeddings'):
        os.makedirs('saved_text_embeddings')

    if 'openseg' in args.feature_2d_extractor:
        model_name="ViT-L/14@336px"
        postfix = '_768' # the dimension of CLIP features is 768
    elif 'lseg' in args.feature_2d_extractor:
        model_name="ViT-B/32"
        postfix = '_512' # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    clip_file_name = 'saved_text_embeddings/clip_{}_labels{}.pt'.format(dataset_name, postfix)

    labelset = [ "a " + label + " in a scene" for label in labelset]
    labelset[-1] = 'other'
    text_features = extract_clip_feature(labelset, model_name=model_name)
    torch.save(text_features, clip_file_name)

    return text_features, palette


def distill(train_loader, model, optimizer, epoch):
    '''Distillation pipeline.'''

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()
    loss_meter_1 = AverageMeter()
    loss_meter_2 = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    text_features, palette = obtain_text_features_and_palette()
    text_features = text_features.float().detach()

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, feat_3d, mask, inds_recons, vox_ind, superpoint, scene_name, offset) = batch_data

        coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
        sinput = SparseTensor(
            feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))

        # load superpoint and corresponding index
        vox_ind, inds_recons, superpoint = vox_ind.cuda(non_blocking=True), inds_recons.cuda(non_blocking=True), superpoint.cuda(non_blocking=True)
        assert inds_recons.shape[0] == superpoint.shape[0]
        assert vox_ind.shape[0] == coords.shape[0]

        # model inference
        output_3d = model(sinput)
        output_3d_ = output_3d[inds_recons, :]

        with torch.no_grad():
            new_list = []
            for j, scene_name in enumerate(scene_name):
                if j == 0:
                    current_pred = output_3d_[0:offset[j], :]
                else:
                    current_pred = output_3d_[offset[j-1]:offset[j], :]

                path = 'pred_history'
                history_pred = torch.load(os.path.join(path, scene_name + '.pth')).cuda(non_blocking=True).float()

                new_pred = 0.9999 * history_pred + 0.0001 * current_pred
                new_list.append(new_pred)
                torch.save(new_pred.detach().cpu().half(), os.path.join(path, scene_name + '.pth'))                                           
            new_output_3d = torch.cat(new_list, dim=0)
            new_output_3d = new_output_3d[vox_ind]



        if output_3d.shape[0] <= 20000:
            superpoint = superpoint[vox_ind]
            _, superpoint = torch.unique(superpoint, sorted=True, return_inverse=True)

            new_output_3d_mean = scatter_mean(new_output_3d, superpoint, dim=0)
            new_output_3d_full = new_output_3d_mean[superpoint, :]


            ps_label_mean = (new_output_3d_mean @ text_features.t()).max(dim=-1)[1]
            ps_label_full = (new_output_3d_full @ text_features.t()).max(dim=-1)[1]

            output_logits = output_3d @ text_features.t()

            loss_1 = nn.CrossEntropyLoss()(output_logits, ps_label_full)

            output_3d = scatter_mean(output_3d, superpoint, dim=0)
            output_logits = output_3d @ text_features.t()

            loss_2 = nn.CrossEntropyLoss()(output_logits, ps_label_mean)
            

            loss = loss_1 + loss_2
        else:
            rand_ind = np.random.choice(range(output_3d.shape[0]), 20000, replace=False)

            output_3d = output_3d[rand_ind]
            new_output_3d = new_output_3d[rand_ind]

            superpoint = superpoint[vox_ind][rand_ind]
            _, superpoint = torch.unique(superpoint, sorted=True, return_inverse=True)

            new_output_3d_mean = scatter_mean(new_output_3d, superpoint, dim=0)
            new_output_3d_full = new_output_3d_mean[superpoint, :]


            ps_label_mean = (new_output_3d_mean @ text_features.t()).max(dim=-1)[1]
            ps_label_full = (new_output_3d_full @ text_features.t()).max(dim=-1)[1]

            output_logits = output_3d @ text_features.t()

            loss_1 = nn.CrossEntropyLoss()(output_logits, ps_label_full)

            output_3d = scatter_mean(output_3d, superpoint, dim=0)
            output_logits = output_3d @ text_features.t()

            loss_2 = nn.CrossEntropyLoss()(output_logits, ps_label_mean)

            loss = loss_1 + loss_2



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_1.update(loss_1.item(), args.batch_size)
        loss_meter_2.update(loss_2.item(), args.batch_size)

        batch_time.update(time.time() - end)

        # adjust learning rate
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            args.base_lr, current_iter, max_iter, power=args.power)

        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time, data_time=data_time,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

        end = time.time()

        torch.cuda.empty_cache()

   

    return [loss_meter.avg, loss_meter_1.avg, loss_meter_2.avg]


def validate(val_loader, model, criterion):
    '''Validation.'''

    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()


    model.eval()
    # obtain the CLIP feature
    text_features, _ = obtain_text_features_and_palette()

    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label, inds_reverse, superpoint, scene_name) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            superpoint = superpoint.cuda(non_blocking=True)

            output = model(sinput)
            output = output[inds_reverse, :]

            output = output.half() @ text_features.t()
            loss = criterion(output, label)
            output = torch.max(output, 1)[1]

            intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
                                                                  args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def initialize_pred(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()


    model.eval()
    # obtain the CLIP feature
    text_features, _ = obtain_text_features_and_palette()

    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label, inds_reverse, superpoint, scene_name) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            superpoint = superpoint.cuda(non_blocking=True)

            output = model(sinput)
            output = output[inds_reverse, :]

            output = scatter_mean(output, superpoint, dim=0)
            output = output[superpoint, :]

            assert len(scene_name) == 1

            if not os.path.exists('pred_history'):
                os.makedirs('pred_history')

            torch.save(output.cpu().half(), os.path.join('pred_history', scene_name[0] + '.pth'))

            output = output.half() @ text_features.t()
            loss = criterion(output, label)
            output = torch.max(output, 1)[1]

            intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
                                                                  args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
