# Deep Learning-based Diagnosis of Maxillary Sinusitis using Paranasal Sinus X-ray 

## Introduction
This project aims to build a deep learning-based diagnostic model of oral cancer. The convolutional neural network (EfficientNet-B3) were trained and evaluated using a total of 12,400 endoscopy of oral cavity (JPG format) acquired from six different medical centers, which will be available with restricted access. This repository includes 10-fold cross-validation, data preprocessing, and evaluation process. This study was supported by grants from the National Information Society Agency (NIA) of Korea.

## Table of Contents
1. Installation and Requirements
	- 1.1. Required Libraries
	- 1.2. Installation
2. Run
	- 2.1. Training
	- 2.2. Testing
3. Licenses

 
## 1. Installation and Requirements

### 1.1. Required Libraries
```
pandas==1.1.1

Pillow==7.2.0

pydicom==2.0.0

scikit-learn==0.23.2

torch==1.7.0.dev20200819+cu101

torchvision==0.8.0.dev20200828+cu101

efficientnet-pytorch==0.7.0

opencv-python-headless==4.4.0.46

matplotlib==3.3.1
```
### 1.2. Installation
Install the library by yourself or build docker image using Dockerfile.

For example, excute the following command in the path where Dockerfile locates.
```
docker build -t oral_cancer .
```
## 2. Data preprocessing
- Resizing to 300x300
- Normalization by statistics of ImageNet dataset : mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)

## 3. Run
### 3.1. Training
```
python train.py 
```
### 3.2. Testing
For example,
```
python train.py --fold 0 --trained_model='model/trained-model-fold0.pth' 
```
## 4. Licenses
Copyright (c) MINDs n company. All rights reserved.

Licensed under the Apache License, Version 2.0.


