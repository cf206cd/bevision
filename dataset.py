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
        self.transform_image = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=config.MEAN,std=config.STD),
                torchvision.transforms.Resize(config.INPUT_IMAGE_SIZE)
                ))
        self.catogories = [item['name'] for item in self.nusc.category]

    def __getitem__(self, index):
        sample_record = self.samples[index]
        data_list = []
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])['ego_pose_token'])
        for sd_token in sample_record['data'].values():
            sd_record = self.nusc.get('sample_data', sd_token)
            if sd_record['sensor_modality'] == 'camera':
                data = {}
                data['width'] = sd_record['width']
                data['height'] = sd_record['height']
                data['translation'] = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['translation']
                data['rotation'] = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['rotation']
                data['camera_intrinsic'] = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])['camera_intrinsic']
                data['raw'] = Image.open(self.nusc.get_sample_data_path(sd_token))
                data_list.append(data)
        instances = []
        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            instance = {}
            instance['translation'] = ann_record['translation']#x:forward,y:left,z:up
            instance['size'] = ann_record['size']#width, length, height
            instance['rotation'] = ann_record['rotation']
            instance['category'] = self.catogories.index(ann_record['category_name'])
            instances.append(instance)
        scene = self.nusc.get('scene',sample_record['scene_token'])
        log = self.nusc.get('log',scene['log_token'])
        map_token = log['map_token']
        x,rots,trans,intrinsics = self.generate_inputs(data_list)
        heatmap_gt,regression_gt,segment_gt = self.generate_targets(egopose,instances,map_token)
        return x,rots,trans,intrinsics,heatmap_gt,regression_gt,segment_gt

    def __len__(self):
        return len(self.samples)

    def generate_inputs(self,data):
        x = np.stack([self.transform_image(i['raw']) for i in data])
        rots = np.stack(self.rotation_matrix(i['rotation']) for i in data)
        trans = np.stack(i['translation'] for i in data)
        intrinsics =  np.stack([self.transform_intrinsic(np.array(i['camera_intrinsic'],dtype=np.float32),i['width'],i['height']) for i in data])
        return x,rots,trans,intrinsics

    def transform_intrinsic(self,intrinsic,width,height):
        intrinsic[0] *= self.config.INPUT_IMAGE_SIZE[1]/width
        intrinsic[1] *= self.config.INPUT_IMAGE_SIZE[0]/height
        return intrinsic

    def rotation_matrix(self,rotation):
        return quaternion.as_rotation_matrix(quaternion.from_float_array(rotation)).astype(np.float32)

    def euler_angles(self,rotation):
        return quaternion.as_euler_angles(quaternion.from_float_array(rotation)).astype(np.float32)

    def generate_targets(self,egopose,instances,map_token):
        ego_pose_translation = np.array(egopose['translation'],dtype=np.float32)
        ego_pose_rotation = self.rotation_matrix(egopose['rotation'])
        det_resolution,det_start_position,det_dimension = generate_grid([self.config.GRID_CONFIG['det']['xbound'],
                                                        self.config.GRID_CONFIG['det']['ybound']])
        seg_resolution,seg_start_position,seg_dimension = generate_grid([self.config.GRID_CONFIG['seg']['xbound'],
                                                        self.config.GRID_CONFIG['seg']['ybound']])
        heatmap_gt = np.zeros((len(self.catogories),*det_dimension[:2].astype(int)))
        regression_gt = np.zeros((5,*det_dimension[:2].astype(int)))
        segment_gt = np.zeros((self.config.NUM_SEG_CLASSES,*seg_dimension[:2].astype(int)))
        for instance in instances:
            det_size = (instance['size'][:2]/ det_resolution)
            radius = max(0,int(self.gaussian_radius(det_size)))
            center = np.linalg.inv(ego_pose_rotation).dot(np.array(instance['translation'],dtype=np.float32)-ego_pose_translation)
            center_loc = det_dimension[:2]-((center[:2] - (det_start_position - det_resolution / 2.)) / det_resolution)[::-1]
            category = instance['category']

            rotation_matrix = np.linalg.inv(ego_pose_rotation).dot(self.rotation_matrix(instance['rotation']))
            rotation = quaternion.from_rotation_matrix(rotation_matrix)
            euler_angles = quaternion.as_euler_angles(rotation)
            value = np.array((
                center_loc[0]-int(center_loc[0]),
                center_loc[1]-int(center_loc[1]),
                euler_angles[0],
                instance['size'][0],
                instance['size'][1]))
            # if self.catogories[category].startswith('vehicle'):
            #     print(value)
            heatmap_gt[category] = self.draw_umich_gaussian(heatmap_gt[category],center_loc,radius)
            regression_gt = self.draw_dense_regression(regression_gt,heatmap_gt[category],center_loc,radius,value)
        
        # for i in range(len(self.catogories)):
        #     self.save_heatmap(heatmap_gt,i)
        return heatmap_gt,regression_gt,segment_gt

    # def save_heatmap(self,heatmap,i):
    #     import cv2
    #     cv2.imwrite("{}.jpg".format(self.catogories[i]),heatmap[i]*255)

    def gaussian_radius(self, det_size, min_overlap=0.7):
        width, height  = det_size

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

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]
            
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: 
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def draw_dense_regression(self,regmap,heatmap,center,radius,value):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
        dim = value.shape[0]
        reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
        
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]
            
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom,
                                    radius - left:radius + right]
        masked_reg = reg[:, radius - top:radius + bottom,
                            radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
            masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
        regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
        return regmap

if __name__ == "__main__":
    nusc_dataset = NuScenesDataset()
    nusc_dataloader = DataLoader(nusc_dataset,batch_size=2)
    for epoch in range(2):
        for iter,data in enumerate(nusc_dataloader):
            #print("iter:{},data:{}".format(iter,data)) 
            pass