import argparse
import json
from dataset_c.path_manager import *
import numpy as np
from utils_c.utils import read_semantickitti_yaml, read_nuscenes_yaml
import yaml


def add_encoding_training_options(parser):
    group = parser.add_argument_group("encoding")
    group.add_argument("--feat_channel_up", type=int, default=64, help="conv feature dimension")
    group.add_argument("--mlp_hidden_channels", type=int, default=256, help="mlp hidden dimension")
    group.add_argument("--mlp_hidden_layers", type=int, default=4, help="mlp hidden layers")
    group.add_argument("--invalid_class", type=bool, default=False)
    group.add_argument("--padding_mode", default='replicate')
    group.add_argument("--lovasz", default=True)
    group.add_argument("--geo_feat_channels", type=int, default=16, help="geometry feature dimension")
    group.add_argument("--z_down", default=False)

def add_diffusion_training_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--steps", type=int, default=100, help="diffusion step")
    group.add_argument("--is_rollout", type=bool, default=True)
    group.add_argument('--mult_channels', default=(1, 2, 4))
    group.add_argument("--diff_lr", type=float, default=5e-4, help="initial learning rate for diffusion training")
    group.add_argument("--schedule_sampler", type=str, default="uniform", help="schedule sampler")
    group.add_argument("--ema_rate", type=float, default=0.9999, help="ema rate")
    group.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    group.add_argument("--log_interval", type=int, default=500, help="log interval")
    group.add_argument("--save_interval", type=int, default=1000, help="save interval")
    group.add_argument("--use_fp16", type=bool, default=False)
    group.add_argument("--predict_xstart", type=bool, default=True)  # prédit directement l'image de départ au lieu du bruit!!!
    group.add_argument("--learn_sigma", type=bool, default=False)
    group.add_argument("--timestep_respacing", default='')
    group.add_argument("--use_ddim", type=str2bool, default=False, help="use ddim")
    group.add_argument("--conv_down", default=True)
    group.add_argument("--diff_n_iters", type=int, default=50000, help="lr ann eal steps for diffusion training")
    group.add_argument("--tri_z_down", default=False)
    group.add_argument('--tri_unet_updown', type=bool, default=True)
    group.add_argument("--model_channels", default=64, help="model channels")
    group.add_argument("--use_conv3d_aware_unet", type=str2bool, default=False, help="use Conv3DAwareUNet instead of TriplaneUNetModel")

    # Conv3DAwareUNet specific parameters (from JSON config)
    group.add_argument("--unet_layers_per_block", type=int, default=2, help="layers per block in UNet")
    group.add_argument("--unet_attention_head_dim", type=int, default=8, help="attention head dimension")
    group.add_argument("--unet_dropout", type=float, default=0.0, help="dropout rate in UNet")
    group.add_argument("--unet_act_fn", type=str, default="silu", help="activation function (silu, mish, gelu, relu)")
    group.add_argument("--unet_norm_eps", type=float, default=1e-5, help="normalization epsilon")
    group.add_argument("--unet_norm_num_groups", type=int, default=32, help="number of groups for group norm")
    group.add_argument("--unet_flip_sin_to_cos", type=str2bool, default=True, help="flip sin to cos in timestep embedding")
    group.add_argument("--unet_freq_shift", type=int, default=0, help="frequency shift in timestep embedding")
    group.add_argument("--unet_resnet_time_scale_shift", type=str, default="default", help="time scale shift (default, scale_shift)")
    group.add_argument("--unet_downsample_padding", type=int, default=1, help="padding in downsample layers")
    group.add_argument("--unet_add_attention", type=str2bool, default=True, help="add attention in mid block")
    group.add_argument("--unet_mid_block_scale_factor", type=float, default=1.0, help="scale factor for mid block")

def add_generation_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--triplane", default=True)
    group.add_argument("--pos", default=True, type=bool)
    group.add_argument("--voxel_fea", default=False)
    group.add_argument('--ssc_refine', default=False, type=bool)
    group.add_argument("--refine_dataset", default='monoscene', choices=['monoscene', 'occdepth', 'scpnet', 'ssasc', 'lmsc', 'motionsc', 'sscfull'])
    group.add_argument("--triplane_loss_type", type=str, default='l2', choices=['l1',  'l2',])
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--diff_net_type", type=str, default='unet_tri')
    group.add_argument("--repaint", default=False, type=bool)

def add_refine_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--triplane", default=True)
    group.add_argument("--pos", default=True, type=bool)
    group.add_argument("--voxel_fea", default=False)
    group.add_argument('--ssc_refine', default=True, type=bool)
    group.add_argument("--dataset",  default='kitti')
    group.add_argument("--triplane_loss_type", type=str, default='l2', choices=['l1',  'l2',])
    group.add_argument("--diff_net_type", type=str, default='unet_tri')
    group.add_argument("--repaint", default=False, type=bool)
    group.add_argument("--batch_size", type=int, default=1)

def add_in_out_sampling(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--triplane", default=True)
    group.add_argument("--pos", default=True, type=bool)
    group.add_argument("--voxel_fea", default=False)
    group.add_argument('--ssc_refine', default=False, type=bool)
    group.add_argument("--refine_dataset", default='monoscene', choices=['monoscene', 'occdepth', 'scpnet', 'ssasc', 'lmsc', 'motionsc', 'sscfull'])
    group.add_argument("--triplane_loss_type", type=str, default='l2', choices=['l1',  'l2',])
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--diff_net_type", type=str, default='unet_tri')
    group.add_argument("--repaint", default=True, type=bool)
    group.add_argument("--dataset",  default='kitti')


def get_gen_args(args):
    if args.dataset == 'kitti' :
        # if args.z_down : H, W, D = 128 ,128, 16
        # elif args.full_size : H, W, D = 256, 256, 32
        # else : H, W, D = 128, 128, 32
        learning_map, learning_map_inv = read_semantickitti_yaml()
        grid_size = tuple(args.grid_size)#(1, 256, 256, 32)
        class_name = [
                'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
                'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
                'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
            ]
        # tri_size = (128, 128, 16) if args.z_down else (128, 128, 32)
        # tri_size = (256, 256, 32) if args.full_size else tri_size
        tri_size= tuple(args.tri_size)
        H=tri_size[0]
        W=tri_size[1]
        D=tri_size[2]
        num_class = args.num_class
        max_points = 400000

    elif args.dataset == 'carla' :
        if args.z_down : H, W, D = 64 ,64, 4
        else : H, W, D = 64, 64, 8
        with open(args.yaml_path, 'r') as stream:
            data_yaml = yaml.safe_load(stream)
        label_remap = data_yaml["learning_map"]
        learning_map = np.asarray(list(label_remap.values()))
        learning_map_inv = None
        class_name = ['building', 'barrier', 'other', 'pedestrian', 'pole', 'road', 'ground', 'sidewalk', 'vegetation', 'vehicle']
        grid_size = (1, 128, 128, 8)
        tri_size = (64, 64, 4) if args.z_down else (64, 64, 8)
        num_class = 11
        max_points = 70000
    elif args.dataset == 'nuscenes' :
        learning_map, learning_map_inv = read_nuscenes_yaml()
        grid_size = tuple(args.grid_size)#(1, 256, 256, 32)
        class_name = ['empty', 'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'pedestrian', 'traffic-cone', 'trailer', 'truck', 'driveable-surface', 'other-flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
        tri_size = tuple(args.tri_size)
        H=tri_size[0]
        W=tri_size[1]
        D=tri_size[2]
        num_class = 18
        max_points = 400000
    return H, W, D, learning_map, learning_map_inv, class_name, grid_size, tri_size, num_class, max_points


def diffusion_defaults():
    return dict(
        learn_sigma=False,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def diffusion_model_defaults():
    return dict(
        in_channels=8,
        out_channels=8,
        num_res_blocks=1,
        dropout=0,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
    )


def get_args_by_group(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return group_dict
    return ValueError('group_name was not found.')


def load_and_overwrite_args(args, path, ignore_keys=[]):
    with open(path, "r") as f:
        overwrite_args = json.load(f)
    for k, v in overwrite_args.items():
        if k not in ignore_keys:
            setattr(args, k, v)
    return args


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
