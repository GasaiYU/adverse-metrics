import torch
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from PIL import Image

import tqdm

import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cacl_psnr(real_filelist, fake_filelist, subset_size=20):
    real_images, fake_images = [], []
    with open(real_filelist, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            real_image = torch.from_numpy(np.asarray(Image.open(line))).permute(2,0,1) / 255.0
            real_images.append(real_image)

    with open(fake_filelist, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            fake_image = torch.from_numpy(np.asarray(Image.open(line))).permute(2,0,1) / 255.0
            fake_images.append(fake_image)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    total_iter = len(real_images) // subset_size

    psnr_value = 0
    for i in tqdm.trange(total_iter):
        psnr_fake_images = torch.stack(fake_images[subset_size*i: subset_size*(i+1)], dim=0)
        psnr_real_images = torch.stack(real_images[subset_size*i: subset_size*(i+1)], dim=0)
        psnr_value += psnr(psnr_fake_images, psnr_real_images)

    return psnr_value / total_iter

if __name__ == '__main__':
    p = argparse.ArgumentParser('Calculate the psnr metric between the generated images and real images')
    p.add_argument('--fake_filelist', help='The path to the fake filelist', type=str)
    p.add_argument('--real_filelist', help='The path to the real filelist', type=str)
    p.add_argument('--subset_size', type=int, help='The size of the subset size', default=20)

    args = p.parse_args()
    psnr_score = cacl_psnr(args.real_filelist, args.fake_filelist, args.subset_size)
    print(f'[PSNR Score] {psnr_score}')