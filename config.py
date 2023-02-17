class Config:
    DATASET_VERSION='v1.0-mini'
    DATASET_DATAROOT='D:/dataset/nuscenes/mini'
    DATASET_VERBOSE=True
    DATASET_MAP_RESOLUTION=0.1
    DATASET_SPLIT = 'train'
    #for grid config:start, end, step
    GRID_CONFIG = {
    #for 3D grid:x forward,y left,z up
    'xbound': [-50.0, 50.0, 200],
    'ybound': [-50.0, 50.0, 200],
    'zbound': [-10.0, 10.0, 1],
    'dbound': [1.0, 50.0, 49],
    }

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    INPUT_IMAGE_SIZE = (288,512) #height,width

    DEVICE = "cuda:0"
    EPOCH = 20
    
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-7
    LR_SCHE_STEP_SIZE = 1500
    LR_SCHE_GAMMA = 1
    MAX_GRAD_NORM = 5

    NUM_CAMERAS = 6
    NUM_SEG_CLASSES = 1
    LOG_DIR = "./runs"
    RANDOM_SEED = 42