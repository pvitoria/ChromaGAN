# ChromaGAN
Official Keras Implementation of ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution [<a href="https://arxiv.org/pdf/1907.09837.pdf"> arXiv</a>]


<p align="center">
  <img width="600" src="Figures/Results.png?raw=true">
</p>


## Requirements
```
pip install -r requirements.txt
```

## Network Architecture
[<img width="900" src="Figures/ColorizationModel.png?raw=true">](Figures/ColorizationModel.png?raw=true)

## Network Parameters
All the parameters can be modified from the config.py file.
```
import os

# DIRECTORY INFORMATION
DATASET = "imagenet"
TEST_NAME ="TEST1"
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
  ```

## Training
To train the network:
```
python ChromaGAN.py
  ```
  
  ## Testing
To test the network:
```
python ChromaGANprint.py
```

    
## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/pdf/1907.09837.pdf"> ChromaGAN: An Adversarial Approach for Picture Colorization</a>:

```
@article{vitoria2019chromagan,
  title={ChromaGAN: An Adversarial Approach for Picture Colorization},
  author={Vitoria, Patricia and Raad, Lara and Ballester, Coloma},
  journal={arXiv preprint arXiv:1907.09837},
  year={2019}
}
```
