import torch
from config import Config
from model import BEVerseWithFixedParam
import torch.onnx

config = Config
rots = torch.randn(1,6,3,3)
trans = torch.randn(1,6,3)
intrins = torch.zeros(1,6,3,3)
for i in range(3):
    rots[:,:,i,i] = 1
    intrins[:,:,i,i] = 1
model = BEVerseWithFixedParam(rots,trans,intrins,grid_confs=config.GRID_CONFIG,num_det_classes=config.NUM_DET_CLASSES,num_seg_classes=config.NUM_SEG_CLASSES,image_size=config.INPUT_IMAGE_SIZE).to(torch.device(config.DEVICE))
model.eval()
x = torch.randn(1, 6, 3, 640, 640)
with torch.no_grad():
    jit_model = torch.jit.script(model,x)
    print(jit_model)
    torch.onnx.export(model,x,"beverse.onnx",opset_version=16)