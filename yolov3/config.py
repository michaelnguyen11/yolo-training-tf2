from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.YOLO = edict()
# Yolo common setting
__C.YOLO.CLASSES = "../data/widerface.names"
__C.YOLO.ANCHORS = "../data/yolo_anchors_train.txt"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.ORIGINAL_WEIGHT = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT = "./checkpoint/yolov3_coco_demo.ckpt"

# Train options
__C.YOLO.TRAIN = edict()

__C.YOLO.TRAIN.ANNOT_PATH = "../data/WIDER_train_1.txt"
__C.YOLO.TRAIN.BATCH_SIZE = 6
# __C.YOLO.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.YOLO.TRAIN.INPUT_SIZE = [416]
__C.YOLO.TRAIN.DATA_AUG = True
__C.YOLO.TRAIN.LEARNING_RATE_INIT = 1e-3
__C.YOLO.TRAIN.LEARNING_RATE_END = 1e-6
__C.YOLO.TRAIN.WARMUP_EPOCHS = 2
__C.YOLO.TRAIN.EPOCHS = 30
__C.YOLO.TRAIN.INITIAL_WEIGHT = "./checkpoint/yolov3_coco_demo.ckpt"

# Test options
__C.YOLO.TEST = edict()

__C.YOLO.TEST.ANNOT_PATH = "../data/WIDER_val_1.txt"
__C.YOLO.TEST.BATCH_SIZE = 2
__C.YOLO.TEST.INPUT_SIZE = 544
__C.YOLO.TEST.DATA_AUG = False
__C.YOLO.TEST.WRITE_IMAGE = True
__C.YOLO.TEST.WRITE_IMAGE_PATH = "./data/detection/"
__C.YOLO.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.YOLO.TEST.WEIGHT_FILE = "./checkpoint/yolov3_test_loss=9.2099.ckpt-5"
__C.YOLO.TEST.SHOW_LABEL = True
__C.YOLO.TEST.SCORE_THRESHOLD = 0.3
__C.YOLO.TEST.IOU_THRESHOLD = 0.45
