import torch
from model import BEVision
import torch.nn as nn
from config import Config
import numpy as np
from PIL import Image
from utils import generate_grid
import os

class Predictor:
    def __init__(self,config):
        self.config = config
        self.model = BEVision(config.GRID_CONFIG,num_det_classes=config.NUM_DET_CLASSES,num_seg_classes=config.NUM_SEG_CLASSES,image_size=config.INPUT_IMAGE_SIZE).to(torch.device(config.DEVICE),)
        self.model.load_state_dict(torch.load(os.path.join(self.config.LOG_DIR,"model_last.pt"),map_location=config.DEVICE))
        self.model.to(self.config.DEVICE).eval()
        xc = torch.tensor(generate_grid(config.GRID_CONFIG['det']['xbound']),device=self.config.DEVICE)
        yc = torch.tensor(generate_grid(config.GRID_CONFIG['det']['ybound']),device=self.config.DEVICE)
        self.xyc = torch.stack(torch.meshgrid(xc, yc, indexing='ij'), dim=2)

    def predict(self,x,rots,trans,intrins):
        x = torch.tensor(x).to(self.config.DEVICE)
        rots = torch.tensor(rots).to(self.config.DEVICE)
        trans = torch.tensor(trans).to(self.config.DEVICE)
        intrins = torch.tensor(intrins).to(self.config.DEVICE)
        seg_res = self.model(x,rots,trans,intrins)
        return seg_res

if __name__ == '__main__':
    config = Config
    predictor = Predictor(config)
    channels = ['CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_FRONT_LEFT']
    image_list = [Image.open("./image/{}.jpg".format(channel)).resize(config.INPUT_IMAGE_SIZE[::-1]) for channel in channels]
    x = np.stack(image_list)
    x = x.astype(np.float32)
    x /= 255.0
    x /= np.array(config.MEAN)
    x -= np.array(config.STD)
    x = x.transpose(0,3,1,2)
    x = np.expand_dims(x,0)
    rots = np.array([[[[ 5.6847786e-03, -5.6366678e-03,  9.9996793e-01],
                    [-9.9998349e-01, -8.3711528e-04,  5.6801485e-03],
                    [ 8.0507132e-04, -9.9998379e-01, -5.6413338e-03]],
                    
                    [[-8.3292955e-01, -9.9460376e-06,  5.5337900e-01],
                    [-5.5330491e-01,  1.6378816e-02, -8.3281773e-01],
                    [-9.0554096e-03, -9.9986583e-01, -1.3647903e-02]],
                    
                    [[-9.3477553e-01,  1.5875839e-02, -3.5488400e-01],
                    [ 3.5507455e-01,  1.1370495e-02, -9.3476886e-01],
                    [-1.0805031e-02, -9.9980932e-01, -1.6265968e-02]],
                    
                    [[ 2.4217099e-03, -1.6753608e-02, -9.9985671e-01],
                    [ 9.9998909e-01, -3.9591072e-03,  2.4883694e-03],
                    [-4.0002293e-03, -9.9985182e-01,  1.6743837e-02]],
                    
                    [[ 9.4776034e-01,  8.6657219e-03, -3.1886551e-01],
                    [ 3.1896114e-01, -1.3976300e-02,  9.4766474e-01],
                    [ 3.7556388e-03, -9.9986476e-01, -1.6010212e-02]],
                    
                    [[ 8.2075834e-01, -3.4143668e-04,  5.7127541e-01],
                    [-5.7127160e-01,  3.2195018e-03,  8.2075477e-01],
                    [-2.1194580e-03, -9.9999475e-01,  2.4473800e-03]]]],dtype=np.float32)
    
    trans = np.array([[[ 1.70079119,  0.01594563,  1.51095764],
                    [ 1.55084775, -0.4934048,   1.49574801],
                    [ 1.0148781 , -0.48056822,  1.56239545],
                    [ 0.02832603,  0.00345137,  1.57910346],
                    [ 1.035691,    0.48479503,  1.59097015],
                    [ 1.52387798,  0.49463134,  1.50932822]]],dtype=np.float32)
    
    intrins = np.array([[[[506.5669,    0.,      326.5068 ],
                        [  0.,      900.5634,  349.51614],
                        [  0.,        0.,        1.     ]],

                        [[504.33896,   0.,      323.18732],
                        [  0.,      896.6026,  352.23782],
                        [  0.,        0.,        1.     ]],

                        [[503.8055,    0.,      322.90118],
                        [  0.,      895.6543,  356.4059 ],
                        [  0.,        0.,        1.     ]],

                        [[323.68842,   0.,      331.68784],
                        [  0.,      575.44604, 342.598  ],
                        [  0.,        0.,        1.     ]],

                        [[502.6966,    0.,      316.84503],
                        [  0.,      893.68286, 350.41833],
                        [  0.,        0.,        1.     ]],

                        [[509.03915,   0.,      330.6462 ],
                        [  0.,      904.95856, 341.15674],
                        [  0.,        0.,        1.     ]]]],dtype=np.float32)
    print(predictor.predict(x,rots,trans,intrins))