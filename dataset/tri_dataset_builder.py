import torch
import yaml
import os
import numpy as np
import pathlib
from diffusion.triplane_util import augment
from utils.parser_util import get_gen_args
import random
from PIL import Image
from diffusion.nn import decompose_featmaps, compose_featmaps



class TriplaneDataset(torch.utils.data.Dataset):
    def __init__(self, args, imageset):
        self.args = args
        self.imageset = imageset
        with open(args.yaml_path, 'r') as stream:
            data_yaml = yaml.safe_load(stream)

        # Get split configuration - supports multiple split sets
        split_config_name = getattr(args, 'split_config', 'default')

        print("split_config_name", split_config_name)

        # Handle different split configurations
        if imageset == 'train':
            split = data_yaml['split']['train']
        elif imageset == 'val':
            split = data_yaml['split']['valid']
        elif imageset == 'minival' or imageset == 'mini_val':
            split = data_yaml['split']['valid']
        elif imageset == 'mini_train':
            split = data_yaml['split'].get('mini_train', data_yaml['split']['train'])
            print("split", split)
        self.tri_size=list(args.tri_size)

        self.scans = []
        
        for i_folder in split:
            if args.dataset == 'kitti': 
                folder = str(i_folder).zfill(2)
                    
                tri_path = os.path.join(args.data_path, folder, self.args.triplane_noisy)
                condition_path=os.path.join(args.data_path, folder, self.args.triplane_cond)
                    
                files = list(pathlib.Path(tri_path).glob('*.npy'))
                 
                files_cond=[pathlib.Path(condition_path) / (p.stem + '.npy') for p in files]
                  
                for idx,filename in enumerate(files):
                    if imageset == 'val' or imageset == 'mini_val' or imageset == 'minival':
                        if (int(str(filename).split('/')[-1].split('.')[0].split("_")[0]) % 5 == 0) :
                            self.scans.append({
                                "triplane": str(filename),
                                "condition" : str(files_cond[idx]),
                            })
                        else :
                            self.scans.append({
                                "triplane": str(filename),
                                "condition" : str(files_cond[idx]),
                            })

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        scale_triplanes = getattr(self.args, 'scale_triplanes', False)
        triplane_scale_factor = getattr(self.args, 'triplane_scale_factor', 1.0)
        if scale_triplanes and triplane_scale_factor != 1.0:
            triplane = triplane / triplane_scale_factor

        if self.args.conditioning:
            if random.random() < self.args.guidance_prob:
                condition = np.load(self.scans[index]["condition"])
                path = self.scans[index]["condition"]
            else :
                condition = np.zeros_like(triplane)
                path = self.scans[index]["triplane"]
        else:
            condition = np.zeros_like(triplane)
            path = self.scans[index]["triplane"]

        if (self.imageset == 'train') and self.args.augment_data: 
            print("augment data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            bev_training = getattr(self.args, 'bev_training', False)
            
            p = torch.randint(0, 6, (1,)).item()
            triplane = augment(triplane, p, self.tri_size,bev_training)
            condition = augment(condition, p, self.tri_size,bev_training)


            
        cond = {'y': condition, 'H': self.tri_size[0], 'W': self.tri_size[1], 'D': self.tri_size[2], 'path': path}

        return triplane, cond
