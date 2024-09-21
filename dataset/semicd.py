from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SemiCDDatasetV2(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        
        imgA = Image.open(os.path.join(self.root, 'A', id)).convert('RGB')
        imgB = Image.open(os.path.join(self.root, 'B', id)).convert('RGB')
        
        # if '.jpg' in id:
        #     mask = np.array(Image.open(os.path.join(self.root, 'label', id)))
        # elif'.png' in id:
        #     mask = np.array(Image.open(os.path.join(self.root, 'label', id)))[:,:,0]
        mask = np.array(Image.open(os.path.join(self.root, 'label', id)))
        if mask.ndim == 3:
            mask = mask[:,:,0]

        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id
        
        imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size)
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)

        if self.mode == 'train_l':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask

        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)
        imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA)
        imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB)

        blur_prob = 0.5
        cutmix_prob = 0.5
        colorjit = (0.5, 0.5, 0.5, 0.25)

        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(*colorjit)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=blur_prob)
        cutmix_box1 = obtain_cutmix_box(imgA_s1.size[0], p=cutmix_prob)
        
        if random.random() < 0.8:
            imgB_s1 = transforms.ColorJitter(*colorjit)(imgB_s1)
        imgB_s1 = blur(imgB_s1, p=blur_prob)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(*colorjit)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=blur_prob)
        cutmix_box2 = obtain_cutmix_box(imgA_s2.size[0], p=cutmix_prob)

        if random.random() < 0.8:
            imgB_s2 = transforms.ColorJitter(*colorjit)(imgB_s2)
        imgB_s2 = blur(imgB_s2, p=blur_prob)
        
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        # mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 255] = 255

        return normalize(imgA_w), normalize(imgB_w), normalize(imgA_s1), normalize(imgB_s1), \
            normalize(imgA_s2), normalize(imgB_s2), ignore_mask, cutmix_box1, cutmix_box2, np.array(mask).astype(np.uint8)

class SemiHelperDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        
        imgA = Image.open(os.path.join(self.root, 'A', id)).convert('RGB')
        imgB = Image.open(os.path.join(self.root, 'B', id)).convert('RGB')
        
        # ipdb.set_trace()
        mask = np.array(Image.open(os.path.join(self.root, 'label', id)))
        # mask = np.array(Image.open(os.path.join(self.root, 'label', id)))[:,:,0]
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id
        
        imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size)
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)

        if self.mode == 'train_l':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id

        # 都是Image对象
        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)
        imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA)
        imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB)

        blur_prob = 0.0
        cutmix_prob = 0.0
        colorjit = (0.5, 0.5, 0.5, 0.25)

        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(*colorjit)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=blur_prob)
        cutmix_box1 = obtain_cutmix_box(imgA_s1.size[0], p=cutmix_prob)
        
        if random.random() < 0.8:
            imgB_s1 = transforms.ColorJitter(*colorjit)(imgB_s1)
        imgB_s1 = blur(imgB_s1, p=blur_prob)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(*colorjit)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=blur_prob)
        cutmix_box2 = obtain_cutmix_box(imgA_s2.size[0], p=cutmix_prob)

        if random.random() < 0.8:
            imgB_s2 = transforms.ColorJitter(*colorjit)(imgB_s2)
        imgB_s2 = blur(imgB_s2, p=blur_prob)
        
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        # mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 255] = 255
        imgA_w, mask = normalize(imgA_w, mask)

        return imgA_w, normalize(imgB_w), normalize(imgA_s1), normalize(imgB_s1), \
            normalize(imgA_s2), normalize(imgB_s2), ignore_mask, cutmix_box1, cutmix_box2, mask, id

    def __len__(self):
        return len(self.ids)


class BaseCDDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        
        imgA = Image.open(os.path.join(self.root, 'A', id)).convert('RGB')
        imgB = Image.open(os.path.join(self.root, 'B', id)).convert('RGB')
        
        # ipdb.set_trace()
        mask = np.array(Image.open(os.path.join(self.root, 'label', id)))
        # mask = np.array(Image.open(os.path.join(self.root, 'label', id)))[:,:,0]
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))


        if self.mode == 'val':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id
        
        imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size)
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)

        if self.mode == 'train_l':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask

        # 都是Image对象
        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)
        imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA)
        imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB)

        blur_prob = 0.0
        colorjit = (0.5, 0.5, 0.5, 0.25)

        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(*colorjit)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=blur_prob)
        
        if random.random() < 0.8:
            imgB_s1 = transforms.ColorJitter(*colorjit)(imgB_s1)
        imgB_s1 = blur(imgB_s1, p=blur_prob)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(*colorjit)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=blur_prob)

        if random.random() < 0.8:
            imgB_s2 = transforms.ColorJitter(*colorjit)(imgB_s2)
        imgB_s2 = blur(imgB_s2, p=blur_prob)
        
        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        # mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 255] = 255
        # import pdb
        # pdb.set_trace()
        return normalize(imgA_w), normalize(imgB_w), normalize(imgA_s1), normalize(imgB_s1), \
            normalize(imgA_s2), normalize(imgB_s2), ignore_mask

    def __len__(self):
        return len(self.ids)
