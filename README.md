# S-AModal

[**Foundation Models for Amodal Video Instance
Segmentation in Automated Driving**](https://arxiv.org/abs/2409.14095) \
Jasmin Breitenstein, Franz Jünger, Andreas Bär, Tim Fingscheidt \
TU Braunschweig, Institute for Communications Technology

![S-AModal Result Visualizations](figures/git_visualization.gif)

## Description
S-AModal is an amodal video instance segmentation method, that builds on a visible video instance segmentation 
providing visible instance masks per frame. S-AModal randomly selects points from this mask to prompt an amodal SAM 
network for the corresponding amodal mask. If no visible mask is predicted, we apply point tracking to track the
previous point prompts and the previous amodal masks to the current frame.

Our code can be used to finetune and evaluate [SAM](https://github.com/facebookresearch/segment-anything/tree/main) to amodal segmentation on the 
[AmodalSynthDrive](http://amodalsynthdrive.cs.uni-freiburg.de/) and on 
the [KINS-car](https://github.com/amazon-science/self-supervised-amodal-video-object-segmentation/tree/main) dataset.
The repository then contains code to apply the proposed S-AModal method for amodal video instance segmentation.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation
### Create the Environment
Clone the repository:
```
git clone https://github.com/ifnspaml/S-AModal
cd s-amodal
```
Create a new Anaconda environment:
```
conda create --name samodal python=3.8.16
conda activate samodal
```
Install Pytorch:
```
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the following packages as needed:
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Please install the requirements if needed. You can also opt for separate conda installations of the requirements.
```
pip install -r requirements.txt
```
For saliency point augmentation, please clone the visual saliency transformer [repository](https://github.com/nnizhang/VST)
```
git clone https://github.com/nnizhang/VST.git
```

For point tracking with  [CoTracker](https://github.com/facebookresearch/co-tracker) download the checkpoint:
```
mkdir pt_checkpoint
wget --output-document pt_checkpoint/cotracker_stride_4_wind_8.pth https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
```
and install:
```
pip install git+https://github.com/facebookresearch/co-tracker.git@4f297a92fe1a684b1b0980da138b706d62e45472
```
## Usage 

To train amodal SAM on AmodalSynthDrive according to our paper:
```
python sam_finetune.py -bs 1 -lr 0.00001 -optimizer aw -dataset asd -mode samadpt -if_warmup False
```
Change the dataset to kcar if you want to train on the KINS-car dataset. 
You can also change the adaptation mode if you want to.
Per default, we use sam_vit_b checkpoint as pre-trained weights due to our memory restraints. 
If you want, you can replace this to another checkpoint in the code of sam_finetune.py.
You can also use the provided shell script sam_finetune.sh to run the fine-tuning.

**Image-level evaluation:** For ASD based on the ground truth and KINS-car based of PointTrack, please run:
```
python sam_inference.py --save_path predictions_asd.json
```
Make sure that you set all correct options in sam_inference.py (dataset, model_checkpoint, mode, modeltype). 
You can also change the inference options pt_augmentation = {None, 'saliency', 'maxdis', 'maxent', 'random'} for point augmentation and
num_points = 1 or any number larger than 0 for the number of point prompts per mask for amodal SAM. 
Default are pt_augmentation = None and num_points = 1.  

If you use a different VIS method on ASD, you can perform inference using the following command:
```
python sam_inference_asdpreds.py --save_path predictions_asd_predbased.json
```
This script is based on the standard output format of [GenVIS](https://github.com/miranheo/GenVIS).

**Video-level evaluation:** For efficiency reasons, we perform the video-level evaluation separately. 
This can easily be combined for full online inference. Video-level evaluation is only possible for AmodalSynthDrive. 
To evaluate S-AModal on video level on ASD based on the ground truth run:
```
python eval_asd.coco_preds_to_ytvis.py
```
This directly provides video-level evaluation and performs point tracking with CoTracker through occlusions. 
It also saves visualizations of the results. Please set the correct paths in the script.

If you wish to use another point tracking method, we refer to [Segment Anything Meets Point Tracking](https://github.com/SysCV/sam-pt/tree/main) 
on how to install them and set them up.

For video-level evaluation on ASD based on an actual VIS method such as GenVIS, please run:
```
python eval_asd.coco_preds_to_ytvis_genvispreds.py
```
The structure of the files is slightly different in this case.

## Acknowledgements

This repository benefits from the works of [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main), 
[Segment Anything Meets Point Tracking](https://github.com/SysCV/sam-pt/tree/main), [AutoSAM](https://github.com/talshaharabany/AutoSAM/tree/main), 
[Medical SAM Adapter](https://github.com/MedicineToken/Medical-SAM-Adapter/tree/main) and [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch/tree/main).
Thanks to the authors for their amazing work and for publishing their code!

## Citing S-AModal
```bash
@inProceedings{Breitenstein2024,
title={Foundation Models for Amodal Video Instance Segmention in Automated Driving},
author = {Jasmin Breitenstein and Franz J\"unger and Andreas B\"ar and Tim Fingscheidt},
booktitle = {Proc. of ECCV-VCAD-Workshop},
year = {2024},
month = sep,
pages = {1--18},
address = {Milan, Italy}
}
```


