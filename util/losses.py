import numpy as np
import os

def compute_loss_l(criterion, pred, mask):
    loss = criterion(pred, mask)
    return loss

def compute_loss_u(criterion, pred, conf, mask, conf_t):
    loss = criterion(pred, mask)
    loss = loss * (conf >= conf_t)
    loss = loss.sum() / mask.numel()
    return loss

