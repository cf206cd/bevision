from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
import torchvision
from PIL import Image
import numpy as np
from config import Config
from pyquaternion import Quaternion
from utils import generate_step
import cv2

class NuScenesDataset(VisionDataset):
    def __init__(self,config=Config):
        super().__init__(root=config.DATASET_DATAROOT)
        self.config = config
        self.nusc = NuScenes(version=config.DATASET_VERSION, dataroot=config.DATASET_DATAROOT, verbose=config.DATASET_VERBOSE, map_resolution=config.DATASET_MAP_RESOLUTION)
        self.scenes = create_splits_scenes()[config.DATASET_SPLIT]
        self.samples = [sample for sample in self.nusc.sample if
                   self.nusc.get('scene', sample['scene_token'])['name'] in self.scenes]
        self.samples_data = self.nusc.sample_data
        self.transform_image = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=config.MEAN,std=config.STD),
                torchvision.transforms.Resize(config.INPUT_IMAGE_SIZE)
                ))
        self.catogories = [item['name'] for item in self.nusc.category]

    def __getitem__(self, index):
        sample_record = self.samples[index]
        images,rots,trans,intrinsics = self.generate_inputs(sample_record)
        segment_gt = self.generate_targets(sample_record)
        return images,rots,trans,intrinsics,segment_gt

    def __len__(self):
        return len(self.samples)

    def generate_inputs(self,sample_record):
        widths=[]
        heights=[]
        translations=[]
        rotations=[]
        camera_intrinsics=[]
        raws=[]
        for sd_token in sample_record['data'].values():
            sd_record = self.nusc.get('sample_data', sd_token)
            if sd_record['sensor_modality'] == 'camera':
                widths.append(sd_record['width'])
                heights.append(sd_record['height'])
                calibrated_sensor = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                translations.append(calibrated_sensor['translation'])
                rotations.append(calibrated_sensor['rotation'])
                camera_intrinsics.append(calibrated_sensor['camera_intrinsic'])
                raws.append(Image.open(self.nusc.get_sample_data_path(sd_token)))
        images = np.stack([self.transform_image(raw) for raw in raws])
        rots = np.stack(Quaternion(rotation).rotation_matrix for rotation in rotations).astype(np.float32)
        trans = np.stack(translation for translation in translations).astype(np.float32)
        intrinsics =  np.stack([self.transform_intrinsic(np.array(camera_intrinsic),widths[i],heights[i]) for i,camera_intrinsic in enumerate(camera_intrinsics)]).astype(np.float32)
        return images,rots,trans,intrinsics

    def transform_intrinsic(self,intrinsic,width,height):
        intrinsic[0] *= self.config.INPUT_IMAGE_SIZE[1]/width
        intrinsic[1] *= self.config.INPUT_IMAGE_SIZE[0]/height
        return intrinsic

    def generate_targets(self,sample_record):
        ego_pose = self.nusc.get('ego_pose',self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])['ego_pose_token'])
        tran = -np.array(ego_pose['translation'])
        rot = Quaternion(ego_pose['rotation']).inverse
        seg_start,seg_interval,seg_count = generate_step([self.config.GRID_CONFIG['xbound'],
                                                        self.config.GRID_CONFIG['ybound']])
        segment_gt = np.zeros((len(self.catogories),*seg_count[:2].astype(int)))
        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            #box = Box(ann_record['translation'],ann_record['size'],Quaternion(ann_record['rotation']),label=self.catogories.index(ann_record['category_name']))
            box = Box(ann_record['translation'],ann_record['size'],Quaternion(ann_record['rotation']),label=0)
            box.translate(tran)
            box.rotate(rot)
            #for segmentation ground truth
            points = np.round((box.bottom_corners()[:2].T-seg_start)/seg_interval).astype(np.int32)[:,::-1]
            segment_gt[box.label] = self.draw_segment_map(segment_gt[box.label],points)
        return segment_gt

    def draw_segment_map(self,segment_map,points):
        cv2.fillPoly(segment_map, [points], 1.0)
        return segment_map
        
if __name__ == "__main__":
    nusc_dataset = NuScenesDataset()
    nusc_dataloader = DataLoader(nusc_dataset,batch_size=2)
    for epoch in range(2):
        for iter,data in enumerate(nusc_dataloader):
            print("iter:{}".format(iter)) 