import argparse
import logging
import os
import pprint
import copy

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
from PIL import Image

from dataset.semicd import SemiCDDatasetV2
from model.segmentor.deeplabv3plusattn import DeepLabV3PlusAttn
from model.segmentor.pspnet import PSPNet

from metrics import evaluate
from util.utils import count_params, init_log, AverageMeter, load_ckpt
from util.dist_helper import setup_distributed
from util.uncertainty import compute_uncertainty


parser = argparse.ArgumentParser(description='Learning_Remote_Sensing_Aleatoric_Uncertainty_for_Semi-Supervised_Change_Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--ckpt', default=None, type=str)

##########################################################################################################
def mixed_chess(img1, img2, patch_num=(2,2),is_tensor=True):
    assert img1.ndim == img2.ndim == 4
    h1, w1 = img1.shape[-2:]
    h2, w2 = img2.shape[-2:]
    assert h1 == h2 or w1 == w2

    patch_width = w1 // patch_num[0]
    patch_height = h1 // patch_num[1]
    if is_tensor:
        mixedImg = torch.zeros_like(img1).type_as(img1)
    else:
        mixedImg = np.zeros_like(img1)
    for i in range(patch_num[1]):
        for j in range(patch_num[0]):
            if (i+j)%2 == 0:
                mixedImg[:,:,i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width] = img1[:,:,i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
            else:
                mixedImg[:,:,i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width] = img2[:,:,i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
    return mixedImg

def two_mixed_tensor(img1, img2, patch_num=(8, 8)):
    """ 
    Args:
        img1 (tensor): (bs, channel, h, w)
        img2 (tensor): (bs, channel, h, w)
        patch_num (tuple, optional): _description_. Defaults to (8, 8).

    """
    temp1 = copy.deepcopy(img1)
    img1 = mixed_chess(temp1, img2, patch_num)
    img2 = mixed_chess(img2, img1, patch_num)

    return img1, img2

def mixed_conf(img1, img2, confs, patch_num=(8, 8),thres=0.5):
    """
        confs guided mixture
    Args:
        img1 (tensor): (bs, channel, h, w)
        img2 (tensor): (bs, channel, h, w)
        confs (tensor:bool): (bs, h, w)
        patch_num (tuple, optional): _description_. Defaults to (8, 8).

    Returns:
        _type_: _description_
    """
    def check_patch(confs, i, j, img_id, patch_height, patch_width, thres):
        positive_confs = confs[img_id, i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]

        return True if (positive_confs.sum().item() / (patch_height*patch_width)) >= thres else False
        
    assert img1.ndim == img2.ndim == 4
    bs1, _, h1, w1 = img1.shape
    bs2, _, h2, w2 = img2.shape
    assert h1 == h2 or w1 == w2 or bs1 == bs2

    patch_width = w1 // patch_num[0]
    patch_height = h1 // patch_num[1]

    mixedImg = img1.clone()
    for row in range(patch_num[1]):
        for col in range(patch_num[0]):
            for img_id in range(bs1):
                if check_patch(confs, row, col, img_id, patch_height, patch_width, thres):
                    mixedImg[img_id,:,row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width] = img2[img_id,:,row*patch_height:(row+1)*patch_height, col*patch_width:(col+1)*patch_width]
    return mixedImg

def two_mixed_confs(img1, img2, confs, patch_num=(8, 8), thres=0.5):
    """ 
    Args:
        img1 (tensor): (bs, channel, h, w)
        img2 (tensor): (bs, channel, h, w)
        patch_num (tuple, optional): _description_. Defaults to (8, 8).

    """
    temp1 = copy.deepcopy(img1)
    img1 = mixed_conf(temp1, img2, confs, patch_num, thres)
    img2 = mixed_conf(img2, img1, confs, patch_num, thres)

    return img1, img2

##########################################################################################################

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model_zoo = {'deeplabv3plusattn': DeepLabV3PlusAttn, 'pspnet': PSPNet}
    assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)
    if args.ckpt is not None:
        model = load_ckpt(model, args.ckpt)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)
    aux = nn.SmoothL1Loss().cuda(local_rank)
    # aux = nn.L1Loss().cuda(local_rank)     
    # aux = nn.MSELoss().cuda(local_rank)  

    trainset_u = SemiCDDatasetV2(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiCDDatasetV2(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDatasetV2(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)


    #################################   settings    #################################
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1
    lr_min = cfg.get('lr_min', 4e-3)
    un_pred_h = []
    un_pred_var = []
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        un_pred_h = AverageMeter()
        un_pred_var = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((imgA_x, imgB_x, mask_x),
                (imgA_u_w, imgB_u_w, imgA_u_s1, imgB_u_s1, 
                 imgA_u_s2, imgB_u_s2, ignore_mask, cutmix_box1, cutmix_box2, mask),
                (imgA_u_w_mix, imgB_u_w_mix, imgA_u_s1_mix, imgB_u_s1_mix, 
                 imgA_u_s2_mix, imgB_u_s2_mix, ignore_mask_mix, _, _, mask_mix)) in enumerate(loader):

            imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            imgA_u_s1, imgB_u_s1 = imgA_u_s1.cuda(), imgB_u_s1.cuda()
            imgA_u_s2, imgB_u_s2 = imgA_u_s2.cuda(), imgB_u_s2.cuda()
            ignore_mask = ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            imgA_u_w_mix, imgB_u_w_mix = imgA_u_w_mix.cuda(), imgB_u_w_mix.cuda()
            imgA_u_s1_mix, imgB_u_s1_mix = imgA_u_s1_mix.cuda(), imgB_u_s1_mix.cuda()
            imgA_u_s2_mix, imgB_u_s2_mix = imgA_u_s2_mix.cuda(), imgB_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(imgA_u_w_mix, imgB_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            imgA_u_s1[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1] = \
                imgA_u_s1_mix[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1]
            imgB_u_s1[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1] = \
                imgB_u_s1_mix[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1]
            imgA_u_s2[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1] = \
                imgA_u_s2_mix[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1]
            imgB_u_s2[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1] = \
                imgB_u_s2_mix[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = imgA_x.shape[0], imgA_u_w.shape[0]

            preds, preds_fp = model(torch.cat((imgA_x, imgA_u_w)), torch.cat((imgB_x, imgB_u_w)), True)
            preds = model(torch.cat((imgA_x, imgA_u_w)), torch.cat((imgB_x, imgB_u_w)))
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            pred_u_s1, pred_u_s2 = model(torch.cat((imgA_u_s1, imgA_u_s2)), torch.cat((imgB_u_s1, imgB_u_s2))).chunk(2)
            
            patch_num = cfg.get('patch_num', (4, 4))
            thrs = cfg.get('conf_thres', 0.3)

            imgA_u_s1, imgA_u_s2 = two_mixed_confs(imgA_u_s1, imgA_u_s2, conf_u_w, patch_num, thrs)
            imgB_u_s1, imgB_u_s2 = two_mixed_confs(imgB_u_s1, imgB_u_s2, conf_u_w, patch_num, thrs)
            pred_u_s1_mix, pred_u_s2_mix = model(torch.cat((imgA_u_s1, imgA_u_s2)), torch.cat((imgB_u_s1, imgB_u_s2))).chunk(2)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)


            if rank == 0:
                with torch.no_grad():
                    conf_pred = compute_uncertainty(pred_u_w, mode='entropy',axis=0)
                    conf_pred_1 = compute_uncertainty(pred_u_w_fp, mode='entropy',axis=0)
                    un_pred_h.update(conf_pred.item())
                    un_pred_var.update(conf_pred_1.item())
                

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()
            
            aux_s1s2_loss = aux(pred_u_s1_mix, pred_u_s2_mix)

            ratio = [6,2,2,2,1]
            pred_mean = (conf_pred_1.item() + conf_pred.item()) / 2
            loss = (loss_x*ratio[0] + loss_u_s1 * ratio[1]*(1-conf_pred.item()) + loss_u_s2 * ratio[2]*(1-conf_pred.item()) + loss_u_w_fp * ratio[3]*(1-conf_pred_1.item()) + aux_s1s2_loss * ratio[4]*(1-pred_mean)) / sum(ratio)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            if optimizer.param_groups[0]["lr"] > lr_min:
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_sup', loss_x.item(), iters)
                writer.add_scalar('train/loss_unsup_avg', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_unsup_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mu_w', conf_pred, iters)
                writer.add_scalar('train/mu_wf', conf_pred_1, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total-loss: {:.3f}, Loss-sup: {:.3f}, Loss-unsup-s1: {:.3f}, Loss-unsup-s2: {:.3f}, Loss-unsup-w-fp: {:.3f}'.format(i, total_loss.avg, total_loss_x.avg, loss_u_s1.item(), loss_u_s2.item(),total_loss_w_fp.avg))

        iou_class, overall_acc, p, r, f1, kappa_score = evaluate(model, valloader, cfg)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.4f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.4f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.4f}'.format(overall_acc))
            logger.info('***** Evaluation ***** >>>> precision: {:.4f}'.format(p))
            logger.info('***** Evaluation ***** >>>> recall: {:.4f}'.format(r))
            logger.info('***** Evaluation ***** >>>> F1 score: {:.4f}'.format(f1))
            logger.info('***** Evaluation ***** >>>> Kappa score: {:.4f}'.format(kappa_score))
            
            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
            writer.add_scalar('eval/precision', p, epoch)
            writer.add_scalar('eval/recall', r, epoch)
            writer.add_scalar('eval/F1', f1, epoch)
            writer.add_scalar('eval/Kappa', kappa_score, epoch)

        is_best = iou_class[1] > previous_best_iou
        previous_best_iou = max(iou_class[1], previous_best_iou)
        if is_best:
            previous_best_acc = overall_acc
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
