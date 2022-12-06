import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from nuscenes.nuscenes import NuScenes
import torchvision
from PIL import Image
import numpy as np
from config import Config
import quaternion
from utils import generate_grid
class NuScenesDataset(VisionDataset):
    def __init__(self,version='v1.0-mini', dataroot='D:/dataset/nuscenes', verbose=True, map_resolution=0.1,config=Config):
        super().__init__(dataroot)
        self.config = config
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose, map_resolution=map_resolution)
        self.samples = self.nusc.sample
        self.samples_data = self.nusc.sample_data
        self.normalize_image = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=config.MEAN,
                std=config.STD),
                torchvision.transforms.Resize(config.INPUT_IMAGE_SIZE)
                ))
        self.max_instance_num = config.MAX_INSTANCE_NUM
        self.data_catogories = [item['name'] for item in self.nusc.category]

    def __getitem__(self, index):
        sample_record = self.samples[index]
        data_list = []
        for sd_token in sample_record['data'].values():
            sd_record = self.nusc.get('sample_data', sd_token)
            if sd_record['sensor_modality'] == 'camera':
                data = {}
                data['translation'] = np.array(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['translation'],dtype=np.float32)
                data['rotation'] = quaternion.as_rotation_matrix(quaternion.from_float_array(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['rotation'])).astype(np.float32)
                data['camera_intrinsic'] = np.array(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['camera_intrinsic'],dtype=np.float32)
                data['camera_intrinsic'][0] *= self.config.INPUT_IMAGE_SIZE[1]/sd_record['width']
                data['camera_intrinsic'][1] *= self.config.INPUT_IMAGE_SIZE[0]/sd_record['height']
                data['raw'] = self.normalize_image(Image.open(self.nusc.get_sample_data_path(sd_token)))
                data['ego_pose_translation'] = np.array(self.nusc.get('ego_pose',sd_record['ego_pose_token'])['translation'],dtype=np.float32)
                data['ego_pose_rotation'] = quaternion.as_rotation_matrix(quaternion.from_float_array(self.nusc.get('ego_pose',sd_record['ego_pose_token'])['rotation'])).astype(np.float32)
                data_list.append(data)
        instances = []
        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            instance = {}
            instance['translation'] = np.array(ann_record['translation'],dtype=np.float32)#x:forward,y:left,z:up
            instance['size'] = np.array(ann_record['size'],dtype=np.float32)#width, length, height
            instance['rotation'] = quaternion.as_rotation_matrix(quaternion.from_float_array(ann_record['rotation'])).astype(np.float32)
            instance['category'] = self.data_catogories.index(ann_record['category_name'])+1
            instances.append(instance)
        scene = self.nusc.get('scene',sample_record['scene_token'])
        log = self.nusc.get('log',scene['log_token'])
        map_token = log['map_token']
        sample_count = len(sample_record['anns'])
        x,rots,trans,intrins = self.generate_inputs(data_list)
        heatmap_gt,regression_gt,segment_gt = self.generate_targets(sample_count,instances,map_token)
        return x,rots,trans,intrins,heatmap_gt,regression_gt,segment_gt

    def __len__(self):
        return len(self.samples)

    def generate_inputs(self,data):
        x = np.stack([i['raw'] for i in data])
        rots =  np.stack([i['rotation'] for i in data])
        trans =  np.stack([i['translation'] for i in data])
        intrins =  np.stack([i['camera_intrinsic'] for i in data])
        return x,rots,trans,intrins

    def generate_targets(self,sample_count,instances,map_token):
        det_resolution,det_start_position,det_dimension = generate_grid([self.config.GRID_CONFIG['det']['xbound'],
                                                        self.config.GRID_CONFIG['det']['ybound']])
        heatmap_gt = np.zeros(len(self.data_catogories),det_dimension.numpy()[:2].astype(int))
        for instance in instances:
            radius = self.gassian_radius(instance['size'][:2])

        return np.zeros(3,dtype=np.float32),np.zeros(3,dtype=np.float32),np.zeros(3,dtype=np.float32)

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)


if __name__ == "__main__":
    nusc_dataset = NuScenesDataset()
    nusc_dataloader = DataLoader(nusc_dataset,batch_size=2)
    for epoch in range(2):
        for iter,data in enumerate(nusc_dataloader):
            #print("iter:{},data:{}".format(iter,data)) 
            pass