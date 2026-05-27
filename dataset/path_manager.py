import os

# manual definition
PROJECT_NAMES = 'SemCity' 
SEMKITTI_DATA_PATH = '/home/fbalde/scratch/datasets/semantic_kitti/dataset/sequences' # the path to the sequences folder
CARLA_DATA_PATH = '' # the path to the sequences folder~/scratch/datasets/semantic_kitti/dataset/sequences

# auto definition
CARLA_YAML_PATH = os.getcwd() + '/dataset/carla.yaml'
SEMKITTI_YAML_PATH = '/home/fbalde/SemCity/dataset/semantic-kitti.yaml'#os.getcwd() + '/dataset/semantic-kitti.yaml'

# manual definition after training
AE_PATH = os.getcwd() + ''  # the path to the pt file 
GEN_DIFF_PATH = os.getcwd() + '' 
SSC_DIFF_PATH = os.getcwd()  + ''