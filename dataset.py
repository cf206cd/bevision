import torch
from torch.utils.data import Dataset,DataLoader
from nuscenes.nuscenes import NuScenes
import torchvision
from PIL import Image
import numpy as np

class NuScenesDataset(Dataset):
    def __init__(self,nusc):
        super().__init__()
        self.nusc = nusc
        self.samples = self.nusc.sample
        self.samples_data = self.nusc.sample_data
        self.normalise_image = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
                ))

    def __getitem__(self, index):
        sample_record = self.samples[index]
        data = {}
        for sd_token in sample_record['data'].values():
            sd_record = self.nusc.get('sample_data', sd_token)
            if sd_record['sensor_modality'] == 'camera':
                data['channel'] = sd_record['channel']
                data['translation'] = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['translation']
                data['rotation'] = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['rotation']
                data['camera_intrinsic'] = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['camera_intrinsic']
                data['raw'] = self.normalise_image(Image.open(self.nusc.get_sample_data_path(sd_token)))
                data['ego_pose'] = self.nusc.get('ego_pose',sd_record['ego_pose_token'])
        annotation = {}
        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            annotation['token'] = ann_record['token']
            annotation['translation'] = ann_record['translation']
            annotation['size'] = ann_record['size']
            annotation['rotation'] = ann_record['rotation']
            annotation['category_name'] = ann_record['category_name']
        scene = self.nusc.get('scene',sample_record['scene_token'])
        log = self.nusc.get('log',scene['log_token'])
        annotation["map"] = log['map_token']
        return {'data':data,
                'annotation':annotation}

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='D:/dataset/nuscenes', verbose=True)
    nusc_dataset = NuScenesDataset(nusc)
    nusc_dataset[0]
    nusc_dataloader = DataLoader(nusc_dataset,batch_size=2)
    for epoch in range(2):
        for iter,data in enumerate(nusc_dataloader):
            print("iter",iter)
            print("data",data['data'])
            print("annotation",data['annotation'])