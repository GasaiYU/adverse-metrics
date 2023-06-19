import re
import numpy as np
from PIL import Image
import torch
import argparse

import random
import torch.nn.functional as F
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_image_list(path, subset_size):
    """
    Get the tensor image list from the filelist.
    
    path: Path to the images filelist.
    """
    with open(path, 'r') as f:
        ls = f.read().strip().split('\n')
        filenames = sorted([x.split(',')[0] for x in ls])[:subset_size]
        image_list = [np.asarray(Image.open(filename)).transpose((2, 0, 1)) for filename in filenames]
    return image_list

def get_image_label_list(path, subset_size):
    """
    Get the tensor image list and tensor image label list from the filelist.
    
    path: Path to the images filelist.
    """
    with open(path, 'r') as f:
        ls = f.read().strip().split('\n')
        filenames = sorted([x.split(',')[0] for x in ls])[:subset_size]
        labelnames = sorted([x.split(',')[0] for x in ls])[:subset_size]
        image_list = [np.asarray(Image.open(filename)).transpose((2, 0, 1)) for filename in filenames]
        label_list = [np.asarray(Image.open(labelname)).transpose((2, 0, 1)) for labelname in labelnames]
    return image_list, label_list


def crop_label_list(label_list, h_w_list=None, num_crops=10):
    """
    Crop the labels of the images.
    
    label_list: The label tensor list.
    num_crops: The number of crops per label
    """
    crop_index = []
    crop_labels = []
    
    h = label_list[0].shape[1] / 2
    w = label_list[0].shape[2] / 4
    
    flag =True
    if not h_w_list:
        h_w_list = []
        flag = False
    cnt = 0

    for label in label_list:
        for i in range(num_crops):
            if not flag:
                crop_h_start = random.randint(0, label.shape[1] - h - 1)
                crop_w_start = random.randint(0, label.shape[2] - w - 1)
                h_w_list.append((crop_h_start, crop_w_start))
            else:
                crop_h_start, crop_w_start = h_w_list[cnt]
                cnt += 1

            crop_h_end = crop_h_start + h
            crop_w_end = crop_w_start + w

            crop_index.append((int(crop_h_start), int(crop_h_end), int(crop_w_start), int(crop_w_end)))
            crop_label = label[:, int(crop_h_start):int(crop_h_end), int(crop_w_start):int(crop_w_end)]
            down_sample_crop_label = torch.tensor(crop_label)
            down_sample_crop_label = F.interpolate(input=down_sample_crop_label.unsqueeze(0), size=[16, 16]).squeeze(0)
            crop_labels.append(down_sample_crop_label)

    return crop_index, crop_labels, h_w_list


def find_knn(fake_crop_labels, real_crop_labels, threhold=0.5):
    """
    Find the nearest fake_crop_label and real_crop_label, their similarity must greater than 0.5
    
    fake_crop_labels: 16*16 fake label tensors
    real_crop_labels: 16*16 real label tensors
    threhold: The smallest similarity's threhold
    """
    res = []
    similarities = []
    for i, fake_crop_label in enumerate(fake_crop_labels):
        for real_crop_label in real_crop_labels:
            similarities.append(((fake_crop_label == real_crop_label).sum() / fake_crop_label.reshape(-1).shape[0]).float())
        values, indices = torch.sort(torch.tensor(similarities), descending=True)
        similarities = []
        if values[0] > threhold:
            res.append((i, indices[0]))
    return res


def get_vgg_features(crop_image, vgg):
    """
    Get the features from vgg16 network relu1-2, relu2-2, relu3-3, relu4-3, relu5-3
    
    crop_image: The fake/real image crop
    vgg: vgg network
    """
    activation = []
    def hook(model, input, output):
        activation.append(output)
    
    vgg.features[3].register_forward_hook(hook=hook)
    vgg.features[8].register_forward_hook(hook=hook)
    vgg.features[15].register_forward_hook(hook=hook)
    vgg.features[22].register_forward_hook(hook=hook)
    vgg.features[29].register_forward_hook(hook=hook)
    
    res = vgg(crop_image)
    return activation
    
def calculate_fid(fake_path, real_path, subset_size=50):
    """
    Calculate the FrechetInceptionDistance (FID) between src and tgt distribution
    
    fake_path: Path to the synthetic images filelist 
    real_path: Path to the real images filelist
    """
    fake_list = get_image_list(fake_path, subset_size)
    real_list = get_image_list(real_path, subset_size)
    fake = torch.tensor(np.stack(fake_list, axis=0))
    real = torch.tensor(np.stack(real_list, axis=0))

    fid = FrechetInceptionDistance(feature=64)
    fid.update(real, real=True)
    fid.update(fake, real=False)
    fid = fid.compute()
    
    print(f'fid_score: {fid}')
    return fid


def calculate_kid(fake_path, real_path, subset_size=50):
    """
    Calculate the KernelInceptionDistance (KID) between src and tgt distribution
    
    fake_path: Path to the synthetic images filelist 
    real_path: Path to the real images filelist
    
    """
    fake_list = get_image_list(fake_path, subset_size)
    real_list = get_image_list(real_path, subset_size)
    fake = torch.tensor(np.stack(fake_list, axis=0))
    real = torch.tensor(np.stack(real_list, axis=0))
    
    kid = KernelInceptionDistance(subset_size=subset_size)
    kid.update(fake, real=False)
    kid.update(real, real=True)
    kid_mean, kid_std = kid.compute()
    print(f'kid_mean ± kid_std: {kid_mean:.6f}±{kid_std:.6f}')
    return kid_mean, kid_std


def calculate_skvd(fake_path, real_path, subset_size=50):
    """
    Calculate the sKVD between src and tgt distribution
    More info about sKVD metrics: https://arxiv.org/abs/2105.04619
    
    fake_path: Path to the synthetic images filelist 
    real_path: Path to the real images filelist
    """
    fake_image_list, fake_label_list = get_image_label_list(fake_path, subset_size)
    real_image_list, real_label_list = get_image_label_list(real_path, subset_size)
    fake_crop_index, fake_crop_labels, h_w_list = crop_label_list(fake_label_list)
    real_crop_index, real_crop_labels, _ = crop_label_list(real_label_list, h_w_list)
    
    select_indices = find_knn(fake_crop_labels, real_crop_labels, threhold=0.5)

    vgg = models.vgg16(pretrained=True).to(device)
    
    fake_vgg_features = []
    real_vgg_features = []
    
    for indice in select_indices:
        i, j = indice
        h1_start, h1_end, w1_start, w1_end = fake_crop_index[i]
        h2_start, h2_end, w2_start, w2_end = real_crop_index[j]

        fake_crop_image = torch.tensor(fake_image_list[i][:, h1_start:h1_end, w1_start:w1_end], dtype=torch.float32).unsqueeze(0).to(device)
        real_crop_image = torch.tensor(real_image_list[j][:, h2_start:h2_end, w2_start:w2_end], dtype=torch.float32).unsqueeze(0).to(device)
        
        fake_vgg_features.append(get_vgg_features(fake_crop_image, vgg))
        real_vgg_features.append(get_vgg_features(real_crop_image, vgg))
    
    breakpoint()
    mmd_skvd_subset_size = len(fake_vgg_features)
    skvd_names = ['skvd_1_2', 'skvd_2_2', 'skvd_3_3', 'skvd_4_3', 'skvd_5_3']
    
    
    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser('Calculate the FID metrics between the synthetic and real images.')
    p.add_argument('--fake_path', type=str, help='path to the synthetic images filelist')
    p.add_argument('--real_path', type=str, help='path to the real images filelist')
    p.add_argument('--metric', type=str, choices=['fid', 'kid', 'skvd'], help='The metrics want to calculate')
    args = p.parse_args()
    if args.metric == 'fid':
        calculate_fid(args.fake_path, args.real_path)
    elif args.metric == 'kid':
        calculate_kid(args.fake_path, args.real_path)
    elif args.metric == 'skvd':
        calculate_skvd(args.fake_path, args.real_path)
    else:
        raise NotImplementedError("Not supported metric")