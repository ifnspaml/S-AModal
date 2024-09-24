# S-AModal
![Static Badge](https://img.shields.io/badge/upload_status-upload_in_progress-pink)
> Foundation Models for Amodal Video Instance
Segmentation in Automated Driving \
> Jasmin Breitenstein, Franz Jünger, Andreas Bär, Tim Fingscheidt \
> TU Braunschweig, Institute for Communications Technology

This repository contains the code to our paper "Foundation Models for Amodal Video Instance
Segmentation in Automated Driving".

## Description
S-AModal is an amodal video instance segmentation method, that builds on a visible video instance segmentation 
providing visible instance masks per frame. S-AModal randomly selects points from this mask to prompt an amodal SAM 
network for the corresponding amodal mask. If no visible mask is predicted, we apply point tracking to track the
previous point prompts and the previous amodal masks to the current frame.

Our code can be used to finetune and evaluate SAM [Kyrillov et al., 2023] to amodal segmentation on the 
AmodalSynthDrive [Sekkat et al., 2023] and on the KINS-car dataset [Yao et al., 2022].
The repository then contains code to apply the proposed S-AModal method for amodal video instance segmentation.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation
### Create the Environment
- Clone the repository:
```bash
git clone https://github.com/ifnspaml/S-AModal
cd s-amodal
```
- Create a new Anaconda environment:
```bash
conda create --name samodal python=3.8.16
conda activate samodal
```
- Install Pytorch:
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

- Install the following packages as needed:
```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

- Please install the requirements:
```bash
pip install -r requirements.txt
```
- For saliency point augmentation, please clone the visual saliency transformer [repository](https://github.com/nnizhang/VST)
```bash
git clone https://github.com/nnizhang/VST.git
```
## Usage 
todo

## Acknowledgements
todo

## License
todo


