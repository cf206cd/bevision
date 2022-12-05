import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from nuscenes.nuscenes import NuScenes
import torchvision
from PIL import Image
import numpy as np
from config import Config
import quaternion

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

    def __getitem__(self, index):
        sample_record = self.samples[index]
        data_list = []
        for sd_token in sample_record['data'].values():
            sd_record = self.nusc.get('sample_data', sd_token)
            if sd_record['sensor_modality'] == 'camera':
                data = {}
                data['translation'] = torch.tensor(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['translation'])
                data['rotation'] = torch.tensor(quaternion.as_rotation_matrix(quaternion.from_float_array(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['rotation'])),dtype=torch.float32)
                data['camera_intrinsic'] = torch.tensor(self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['camera_intrinsic'])
                data['camera_intrinsic'][0] *= self.config.INPUT_IMAGE_SIZE[1]/sd_record['width']
                data['camera_intrinsic'][1] *= self.config.INPUT_IMAGE_SIZE[0]/sd_record['height']
                data['raw'] = self.normalize_image(Image.open(self.nusc.get_sample_data_path(sd_token)))
                data['ego_pose_translation'] = torch.tensor(self.nusc.get('ego_pose',sd_record['ego_pose_token'])['translation'])
                data['ego_pose_rotation'] = torch.tensor(quaternion.as_rotation_matrix(quaternion.from_float_array(self.nusc.get('ego_pose',sd_record['ego_pose_token'])['rotation'])),dtype=torch.float32)
                data_list.append(data)
        default_instance = {'token':'0','translation':[0,0,0],'size':[0,0,0],'rotation':[0,0,0,0],'category_name':'None'}
        instances = [default_instance for i in range(self.max_instance_num)]
        for i,ann_token in enumerate(sample_record['anns']):
            ann_record = self.nusc.get('sample_annotation', ann_token)
            instance = {}
            instance['token'] = ann_record['token']
            instance['translation'] = ann_record['translation']
            instance['size'] = ann_record['size']
            instance['rotation'] = ann_record['rotation']
            instance['category_name'] = ann_record['category_name']
            instances[i] = instance
        scene = self.nusc.get('scene',sample_record['scene_token'])
        log = self.nusc.get('log',scene['log_token'])
        map_token = log['map_token']
        sample_count = len(sample_record['anns'])
        return {'data':data_list,
                "sample_count":sample_count,
                'instances':instances,
                'map':map_token}

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    nusc_dataset = NuScenesDataset()
    nusc_dataloader = DataLoader(nusc_dataset,batch_size=2)
    for epoch in range(2):
        for iter,data in enumerate(nusc_dataloader):
            print("iter:{},data{}".format(iter,data))