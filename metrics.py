import torch
import numpy as np

import torch
import numpy as np
import torch.distributed as dist
from util.utils import AverageMeter, intersectionAndUnion

import warnings
import pdb

def ConfusionMatrix(num_classes, pres, gts):
    def __get_hist(pre, gt):
        pre = pre.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        pre[pre >= 0.5] = 1
        pre[pre < 0.5] = 0
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        mask = (gt >= 0) & (gt < num_classes)
        label = num_classes * gt[mask].astype(int) + pre[mask].astype(int)
        hist = np.bincount(label, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist
    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(gts, pres):
        cm += __get_hist(lt.flatten(), lp.flatten())
    return cm


def get_score(confusionMatrix):
    precision = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=0) + np.finfo(np.float32).eps)
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=1) + np.finfo(np.float32).eps)
    f1score = 2 * precision * recall / ((precision + recall) + np.finfo(np.float32).eps)
    iou = np.diag(confusionMatrix) / (
            confusionMatrix.sum(axis=1) + confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) + np.finfo(
        np.float32).eps)
    po = np.diag(confusionMatrix).sum() / (confusionMatrix.sum() + np.finfo(np.float32).eps)
    pe = (confusionMatrix[0].sum() * confusionMatrix[0:2][0].sum() + confusionMatrix[1].sum() * confusionMatrix[0:2][
        1].sum()) / confusionMatrix.sum() ** 2 + np.finfo(np.float32).eps
    kc = (po - pe) / (1 - pe + np.finfo(np.float32).eps)
    return precision, recall, f1score, iou, kc


def get_score_sum(confusionMatrix):
    num_classes = confusionMatrix.shape[0]
    precision, recall, f1score, iou, kc = get_score(confusionMatrix)
    cls_precision = dict(zip(['precision_' + str(i) for i in range(num_classes)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(num_classes)], recall))
    cls_f1 = dict(zip(['f1_' + str(i) for i in range(num_classes)], f1score))
    cls_iou = dict(zip(['iou_' + str(i) for i in range(num_classes)], iou))
    return cls_precision, cls_recall, cls_f1, cls_iou, kc


def bs_visual(preds, imgA, imgB, gts):
    bs = len(preds) 
    pdb.set_trace()


def evaluate(model, loader, cfg, visual=False):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()
    cm_total = np.zeros((2, 2))

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            pred = model(imgA, imgB).argmax(dim=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cm = ConfusionMatrix(2, pred, mask)
                cm_total += cm
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            
            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

            ################################### external metrics #######################################
    p, r, f1, iou, kappa_score = get_score_sum(cm_total)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    return iou_class, overall_acc, p['precision_1'], r['recall_1'], f1['f1_1'], kappa_score


def uncertainty_evaluate(model, loader, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()
    cm_total = np.zeros((2, 2))

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            pred = model(imgA, imgB).argmax(dim=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cm = ConfusionMatrix(2, pred, mask)
                cm_total += cm
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            
            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

            ################################### external metrics #######################################
    p, r, f1, iou, kappa_score = get_score_sum(cm_total)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    return iou_class, overall_acc, p['precision_1'], r['recall_1'], f1['f1_1'], kappa_score