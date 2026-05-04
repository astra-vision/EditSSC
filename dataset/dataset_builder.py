from dataset.kitti_dataset import SemKITTI
from dataset.carla_dataset import CarlaDataset
from dataset.nuscenes_dataset import NuSceneDataset

def dataset_builder(args):
    print("build dataset")
    if args.dataset == 'kitti':
        dataset = SemKITTI(args, 'train',augment=args.augment_data)
        val_dataset = SemKITTI(args, 'val')
        
        class_names = [
            'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
            'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
            ]
            
    return dataset, val_dataset, args.num_class, class_names