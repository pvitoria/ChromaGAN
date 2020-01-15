# ChromaGAN
Official Keras Implementation of ChromaGAN: An Adversarial Approach for Picture Colorization [<a href="https://arxiv.org/pdf/1907.09837.pdf">arXiv</a>]


<p align="center">
  <img width="600" src="Figures/Results.png?raw=true">
</p>


## Network Architecture
[<img width="900" src="Figures/ColorizationModel.png?raw=true">](Figures/ColorizationModel.png?raw=true)



## Prerequisits 
Linux 

Python 3

NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

## Getting Started


### Clone Repository
```
git clone https://github.com/pvitoria/ChromaGAN
cd ChromaGAN/
```

### Requirements
```
pip install -r requirements.txt
```

### Download the dataset
Download dataset and place it in the `/DATASET/` folder.
We have train our model with ImageNet dataset.
You can download it from <a href="http://image-net.org/download"> here </a> 


### Network Parameters
All the parameters can be modified from the `config.py` file. 
Note: Modify the name of the dataset in the config file in `DATASET`. For each test you can modify the folder name in `TEST_NAME`. The variable PRETRAINED should be changed by the name of your pretrained colorization file.
```
import os

# DIRECTORY INFORMATION
DATASET = "_"
TEST_NAME ="_"
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/'+DATASET+'/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

TRAIN_DIR = "_"
TEST_DIR = "_"

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 10


# TRAINING INFORMATION
USE_PRETRAINED = False
PRETRAINED = "my_model_colorizationEpoch4.h5"
NUM_EPOCHS = 5
  ```

### Training
To train the network:
```
cd ChromaGAN/SOURCE/
python ChromaGAN.py
  ```
Models are saved to `./MODELS/DATASET/TEST_NAME/` 
  ### Testing
To test the network:
```
cd ChromaGAN/SOURCE/
python ChromaGANprint.py
```
Images are saved to `./RESULT/DATASET/TEST_NAME/` 

Note: Pretrained models will be provided soon.
    
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
## Aknowledgments 

The authors acknowledge partial support by MICINN/FEDER UE project, reference PGC2018-098625-B-I00 VAGS, and by H2020-MSCA-RISE-2017 project, reference 777826 NoMADS. We also thank the support of NVIDIA Corporation for the donation of GPUs used in this work.
