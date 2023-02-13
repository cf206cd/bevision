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
        images,rots,trans,intrinsics = self.generate_inputs(sample_record)
        heatmap_gt,regression_gt,segment_gt = self.generate_targets(sample_record)
        return images,rots,trans,intrinsics,heatmap_gt,regression_gt,segment_gt

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
        rots = np.stack(Quaternion(rotation).rotation_matrix for rotation in rotations)
        trans = np.stack(translation for translation in translations)
        intrinsics =  np.stack([self.transform_intrinsic(np.array(camera_intrinsic,dtype=np.float32),widths[i],heights[i]) for i,camera_intrinsic in enumerate(camera_intrinsics)])
        return images,rots,trans,intrinsics

    def transform_intrinsic(self,intrinsic,width,height):
        intrinsic[0] *= self.config.INPUT_IMAGE_SIZE[1]/width
        intrinsic[1] *= self.config.INPUT_IMAGE_SIZE[0]/height
        return intrinsic

    def generate_targets(self,sample_record):
        ego_pose = self.nusc.get('ego_pose',self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])['ego_pose_token'])
        tran = -np.array(ego_pose['translation'])
        rot = Quaternion(ego_pose['rotation']).inverse
        det_start,det_interval,det_count = generate_step([self.config.GRID_CONFIG['det']['xbound'],
                                                        self.config.GRID_CONFIG['det']['ybound']])
        seg_start,seg_interval,seg_count = generate_step([self.config.GRID_CONFIG['seg']['xbound'],
                                                        self.config.GRID_CONFIG['seg']['ybound']])
        heatmap_gt = np.zeros((len(self.catogories),*det_count[:2].astype(int)))
        regression_gt = np.zeros((5,*det_count[:2].astype(int)))
        segment_gt = np.zeros((len(self.catogories),*seg_count[:2].astype(int)))
        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            box = Box(ann_record['translation'],ann_record['size'],Quaternion(ann_record['rotation']),label=self.catogories.index(ann_record['category_name']))
            box.translate(tran)
            box.rotate(rot)

            #for detection ground truth
            radius = max(self.config.RADIUS_TAU,int(self.gaussian_radius(box.wlh[:2]/det_interval)))
            center_loc = det_count-1-((box.center[:2] - det_start) / det_interval)[::-1]
            orientation = np.arctan2(box.orientation.rotation_matrix[1,0],box.orientation.rotation_matrix[0,0])
            value = np.array((
                (center_loc[0]-np.floor(center_loc[0]))*2-1,
                (center_loc[1]-np.floor(center_loc[1]))*2-1,
                orientation*0.5/np.pi,
                box.wlh[0]/det_interval[0],
                box.wlh[1]/det_interval[1]))
            heatmap_gt[box.label],regression_gt = self.draw_detect_map(heatmap_gt[box.label],regression_gt,center_loc,radius,value)

            #for segmentation ground truth
            points = seg_count-1-np.round((box.bottom_corners()[:2].T-seg_start)/seg_interval).astype(np.int32)[:,::-1]
            segment_gt[box.label] = self.draw_segment_map(segment_gt[box.label],points)
        return heatmap_gt,regression_gt,segment_gt

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

    def draw_detect_map(self,heatmap,regression,center,radius,value, k=1):
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
        masked_regression = regression[:, y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom,
                                    radius - left:radius + right]
        masked_reg = reg[:, radius - top:radius + bottom,
                            radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            idx = (masked_gaussian >= masked_heatmap).reshape(
                1, masked_gaussian.shape[0], masked_gaussian.shape[1])
            masked_regression = (1-idx) * masked_regression + idx * masked_reg
        regression[:, y - top:y + bottom, x - left:x + right] = masked_regression
        return heatmap,regression

    def draw_segment_map(self,segment_map,points):
        cv2.fillPoly(segment_map, [points], 1.0)
        return segment_map
        
if __name__ == "__main__":
    nusc_dataset = NuScenesDataset()
    nusc_dataloader = DataLoader(nusc_dataset,batch_size=2)
    for epoch in range(2):
        for iter,data in enumerate(nusc_dataloader):
            print("iter:{}".format(iter)) 