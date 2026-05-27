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
                complete_path_velodyne = pathlib.Path(os.path.join("/home/fbalde/scratch/datasets/semantic_kitti/dataset/sequences", folder, "velodyne"))
                files_velodyne=[complete_path_velodyne / (p.stem + '.bin') for p in files]
                  
                for idx,filename in enumerate(files):
                    if imageset == 'val' or imageset == 'mini_val' or imageset == 'minival':
                        if (int(str(filename).split('/')[-1].split('.')[0].split("_")[0]) % 5 == 0):
                            self.scans.append({
                                "triplane": str(filename),
                                "condition": str(files_cond[idx]),
                                "velodyne_path": str(files_velodyne[idx]),
                            })
                    else:
                        self.scans.append({
                            "triplane": str(filename),
                            "condition": str(files_cond[idx]),
                            "velodyne_path": str(files_velodyne[idx]),
                        })

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        triplane = np.load(self.scans[index]["triplane"]).squeeze()

        scale_triplanes = getattr(self.args, 'scale_triplanes', False)
        triplane_scale_factor = getattr(self.args, 'triplane_scale_factor', 1.0)
        if scale_triplanes and triplane_scale_factor != 1.0:
            triplane = triplane / triplane_scale_factor

        if self.args.conditioning:
            if random.random() < self.args.guidance_prob:
                condition = np.load(self.scans[index]["condition"])
                path = self.scans[index]["condition"]
            else:
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


        load_lidar = getattr(self.args, 'load_lidar', False)
        data={}
        if load_lidar and self.scans[index].get("velodyne_path"):
            velodyne_path = self.scans[index]["velodyne_path"]
            if os.path.exists(velodyne_path):
                # Load points (x, y, z, intensity)
                velodyne = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
                points = velodyne[:, :3]

                # Generate binary occupancy grid (256, 256, 32)
                # Standard SemanticKITTI bounds: x:[0, 51.2], y:[-25.6, 25.6], z:[-2, 4.4]
                # Resolution: 0.2m
                grid_size = (256, 256, 32)
                x_min, x_max = 0.0, 51.2
                y_min, y_max = -25.6, 25.6
                z_min, z_max = -2.0, 4.4

                # Filter points outside bounds
                mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                       (points[:, 1] >= y_min) & (points[:, 1] < y_max) & \
                       (points[:, 2] >= z_min) & (points[:, 2] < z_max)
                points_filtered = points[mask]

                # Discretize coordinates
                coords = np.zeros(points_filtered.shape, dtype=np.int32)
                coords[:, 0] = np.floor((points_filtered[:, 0] - x_min) / 0.2).astype(np.int32)
                coords[:, 1] = np.floor((points_filtered[:, 1] - y_min) / 0.2).astype(np.int32)
                coords[:, 2] = np.floor((points_filtered[:, 2] - z_min) / 0.2).astype(np.int32)

                # Create binary occupancy grid
                occupancy = np.zeros(grid_size, dtype=np.float32)
                occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
                data['occupancy'] = occupancy
        cond = {'y': condition, 'H': self.tri_size[0], 'W': self.tri_size[1], 'D': self.tri_size[2], 'path': path}

        return triplane, cond, data
