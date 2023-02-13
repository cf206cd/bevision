class Config:
    DATASET_VERSION='v1.0-mini'
    DATASET_DATAROOT='D:/dataset/nuscenes/mini'
    DATASET_VERBOSE=True
    DATASET_MAP_RESOLUTION=0.1
    DATASET_SPLIT = 'train'
    RADIUS_TAU = 2
    #for grid config:start, end, step
    GRID_CONFIG = {
    #for 3D grid:x forward,y left,z up
    'base': {
        'xbound': [-50.0, 50.0, 200],
        'ybound': [-50.0, 50.0, 200],
        'zbound': [-10.0, 10.0, 1],
        'dbound': [1.0, 60.0, 59],
    },
    #for 2D gird:x down,y right
    'det': {
        'xbound': [-40.0, 40.0, 200],
        'ybound': [-40.0, 40.0, 200],
    },
    'seg': {
        'xbound': [-40.0, 40.0, 200],
        'ybound': [-40.0, 40.0, 200],
    }
    }

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    INPUT_IMAGE_SIZE = (640,640) #height,width

    DEVICE = "cpu"
    EPOCH = 10
    
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.99
    LR_SCHE_STEP_SIZE = 150
    LR_SCHE_GAMMA = 0.1

    NUM_DET_CLASSES = 23
    NUM_SEG_CLASSES = 23
    MODEL_SAVE_PATH = "./model.pt"
    DET_THRESHOLD = 0.5
    RANDOM_SEED = 42