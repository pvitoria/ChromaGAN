import os

# DIRECTORY INFORMATION
DATASET = "imagenet"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

TRAIN_DIR = "train_full"
TEST_DIR = "histo"

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 10

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
USE_PRETRAINED = False
PRETRAINED = "model2_1.ckpt"
NUM_EPOCHS = 5
