import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from util import normalize_coords, gen_grid_np
from PIL import Image

def load_image4(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img= Image.fromarray(img)
    img = Image.merge("RGB", (img, img, img))
    img= np.array(img)
   
    
    return img
def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class RAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        #img_names = sorted(os.listdir(self.img_dir))
        img_names = sorted([i for i in os.listdir(self.img_dir) if i[0] != '.'])
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = load_image4(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        #-----------------------------------------------------------------------------------
        self.num_pairs = self.args.num_pairs
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)
     

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        
        block_len = (self.num_pts**0.5) #num_pts = 1024   #16
        h_blocks = self.h//block_len #25   #50
        w_blocks = self.w//block_len #24  #49
        num_blocks = (h_blocks)*(w_blocks) #2450
        id1 = int((idx // (num_blocks))%self.num_imgs) #0
        
        img_name1 = self.img_names[id1]
        max_interval = min(self.max_interval.value, self.num_imgs - 1)
        img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
        img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)]

        # sample more often from i-1 and i+1
        id2s = np.array([self.img_names.index(n) for n in img2_candidates])
        sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
        sample_weights /= np.sum(sample_weights)
        sample_weights[np.abs(id2s - id1) <= 1] = 0.5
        sample_weights /= np.sum(sample_weights)

        img_name2 = np.random.choice(img2_candidates, p=sample_weights)
        id2 = self.img_names.index(img_name2)
        frame_interval = abs(id1 - id2)

        # read image, flow and confidence
        img1 = load_image4(os.path.join(self.img_dir, img_name1)) / 255.
        img2 = load_image4(os.path.join(self.img_dir, img_name2)) / 255.

        flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
        flow = np.load(flow_file)
        mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
        masks = imageio.imread(mask_file) / 255.

        coord1 = self.grid
        coord2 = self.grid + flow

        cycle_consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

#-------------------------------------------------------------------------------------------------       
        block_num = idx % num_blocks  #80
        x_range = ((block_num%w_blocks)*block_len , ((block_num%w_blocks)+1)*block_len)
        y_range = ((block_num//w_blocks)*block_len , ((block_num//w_blocks)+1)*block_len)
        x, y = np.meshgrid(np.arange(self.w), np.arange(self.h))
        mask = np.logical_and.reduce((x >= x_range[0], x < x_range[1], y >= y_range[0], y < y_range[1]))
        
        if mask.sum() == 0:
            print('zero')
            invalid = True
            mask = np.ones_like(cycle_consistency_mask)
        else:
            invalid = False
        
        

        
        pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)

        pts1 = torch.from_numpy(coord1[mask]).float()
        pts2 = torch.from_numpy(coord2[mask]).float()
        pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]

        covisible_mask = torch.from_numpy(cycle_consistency_mask[mask]).float()[..., None]
        weights = torch.ones_like(covisible_mask) * pair_weight

        gt_rgb1 = torch.from_numpy(img1[mask]).float()
        gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                align_corners=True).squeeze().T

        if invalid:
            weights = torch.zeros_like(weights)


        data = {'ids1': id1,
                'ids2': id2,
                'pts1': pts1,  # [n_pts, 2]
                'pts2': pts2,  # [n_pts, 2]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'gt_rgb2': gt_rgb2,
                'weights': weights,  # [n_pts, 1]
                'covisible_mask': covisible_mask,  # [n_pts, 1]
                }
        return data