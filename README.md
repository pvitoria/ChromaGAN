# ChromaGAN
Official Keras Implementation of ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution [<a href="https://openaccess.thecvf.com/content_WACV_2020/html/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.html">WACV 2020</a>] [<a href="https://arxiv.org/pdf/1907.09837.pdf">arXiv</a>] 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pvitoria/ChromaGAN/blob/master/DemoChromaGAN.ipynb)

<p align="center">
  <img height="500" src="Figures/27.31452560_ILSVRC2012_val_00000186._reconstructed.jpg?raw=true">
<img height="500" src="Figures/10.23894882_ILSVRC2012_val_00000680._reconstructed.jpg?raw=true">
<img height="280" src="Figures/22.09770966_ILSVRC2012_val_00001042._reconstructed.jpg?raw=true">
<img height="280" src="Figures/23.27637291_ILSVRC2012_val_00000023._reconstructed.jpg?raw=true">
<img height="280" src="Figures/26.34756279_ILSVRC2012_val_00000796._reconstructed.jpg?raw=true">
<img height="280" src="Figures/27.84384155_ILSVRC2012_val_00000013._reconstructed.jpg?raw=true">
<img height="280" src="Figures/28.33658218_ILSVRC2012_val_00049397._reconstructed.jpg?raw=true">
<img height="280" src="Figures/28.42907333_ILSVRC2012_val_00001398._reconstructed.jpg?raw=true">
<img height="280" src="Figures/31.34938049_ILSVRC2012_val_00000211._reconstructed.jpg?raw=true">
<img height="280" src="Figures/8.27914047_ILSVRC2012_val_00000915._reconstructed.jpg?raw=true">
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
DATASET = "imagenet" # modify
TEST_NAME ="test1" # modify
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/'+DATASET+'/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/'+DATASET+'/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/'+DATASET+'/')
LOG_DIR = os.path.join(ROOT_DIR, 'LOGS/'+DATASET+'/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 10


# TRAINING INFORMATION
PRETRAINED = "my_model_colorization.h5" 
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
To test the network you can either run the code directly from Colab using our  <a href="http://gpi.upf.edu/chromagan/my_model_colorization.h5">Demo</a> or run the code as follows :
```
cd ChromaGAN/SOURCE/
python ChromaGANPrint.py
```
Images are saved to `./RESULT/DATASET/TEST_NAME/` 

## Pretrained Weights

You can donwload the pretrained weights from <a href="https://drive.google.com/drive/folders/12s4rbLmnjW4e8MmESbfRStGbrjOrahlW?usp=sharing">here</a>.
In order to test the network you should use the file called ` my_model_colorization.h5. 
    
## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/pdf/1907.09837.pdf"> ChromaGAN: An Adversarial Approach for Picture Colorization</a>:

```
@inproceedings{vitoria2020chromagan,
  title={ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution},
  author={Vitoria, Patricia and Raad, Lara and Ballester, Coloma},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={2445--2454},
  year={2020}
}
```
## Aknowledgments 

The authors acknowledge partial support by MICINN/FEDER UE project, reference PGC2018-098625-B-I00 VAGS, and by H2020-MSCA-RISE-2017 project, reference 777826 NoMADS. We also thank the support of NVIDIA Corporation for the donation of GPUs used in this work.
