#Datasets are adapted from https://github.com/talshaharabany/AutoSAM/blob/main/dataset/polyp.py

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import cvbase as cvb
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
import pickle
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from utils.helper_functions import *


class KINSCarDataset(data.Dataset):
    """
    dataloader for KINS-car dataset
    KINS-car dataset: https://github.com/amazon-science/self-supervised-amodal-video-object-segmentation/tree/main
    """

    def __init__(self, image_root, gt_root, sam_model, trainsize=352, augmentations=None, train=True, sam_trans=None,test =False):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.test = test

        anns = pickle.load(
            open(gt_root, "rb"))

        #hard coded access to common files and kins json
        kins_annotions_path = '/beegfs/work/shared/kitti-kins/update_train_2020.json'
        common_files_path = '/beegfs/work/breitenstein/sam-pt/common_images.json'

        with open(kins_annotions_path, 'r') as file:
            self.kins_anns = json.load(file)

        with open(common_files_path, 'r') as file:
            self.common_files = json.load(file)

        self.videos = anns.keys()

        #now extract from common images the corresponding KINS filenames:
        cor_kins_files = []
        for vid in list(self.videos):
            for common in self.common_files:
                if vid in common[0]:
                    cor_kins_files.append(common[1])

        #now we have a list of KINS files in our split:
        # look for the correct annotations:
        kins_images = self.kins_anns['images']
        filename_to_dict = {d['file_name']: d for d in kins_images}
        kins_annotations = self.kins_anns['annotations']
        image_id_list=[]
        self.images = {}

        all_files= []
        for f in kins_images:
            all_files.append(f['file_name'])

        for kins_file in cor_kins_files:
            if kins_file.split('/')[-1] in all_files:
                dict_entry = filename_to_dict[kins_file.split('/')[-1]]
                self.images[dict_entry['id']] = kins_file.split('/')[-1]
                image_id_list.append(dict_entry['id'])

        self.anns_info = []
        for ann in kins_annotations:
            if ann['image_id'] in image_id_list:
                if ann['category_id']==4:
                    self.anns_info.append(ann)

        if self.test:

            for ann in self.anns_info:
                image_id = ann['image_id']
                cor_im = [im for im in kins_images if im['id']==image_id][0]

                pred_mask_encoded = mask_utils.merge(mask_utils.frPyObjects(ann['a_segm'], cor_im['height'], cor_im['width']))
                pred_mask_encoded['counts'] = str(pred_mask_encoded['counts'], 'utf-8')
                ann['segmentation'] = pred_mask_encoded
                ann['ignore'] = 0
                ann['area'] = ann['a_area']
                # we need an occlusion rate
                if ann['area'] > 0:
                    ann['occlude_rate'] =  1 - (ann['i_area']/ann['area'])
                else:
                    ann['occlude_rate'] = int(0)
                ann['iscrowd']= 0

            self.coco_gt_dict = {
                        "info": {},
                        "licenses": [],
                        "images": kins_images,
                        "annotations": self.anns_info,
                        "categories": self.kins_anns['categories'],
                    }
            with open('ground_truth_annotations_kcar_test.json', 'w', encoding='utf-8') as f:
                json.dump(self.coco_gt_dict, f)


        self.image_root = image_root
        self.gt_root = gt_root

        self.size = len(self.anns_info)
        self.train = train
        self.sam_trans = sam_trans
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model

    def __len__(self):
        return len(self.anns_info)

    def __getitem__(self, index):
        anns = self.anns_info[index]# annotation for index
        image_id = anns['image_id']
        img_name = self.images[image_id]#['file_name']

        img_path = os.path.join(self.image_root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        a_mask = polys_to_mask(anns["a_segm"], height, width)
        i_mask = polys_to_mask(anns["i_segm"], height, width)
        mask = a_mask + i_mask
        mask = mask.astype('uint8')

        img = torch.tensor(img)
        img = img.permute((2, 0, 1))
        mask = torch.tensor(mask)
        img = F.interpolate(img[np.newaxis],size=(375, 1242),mode='bilinear')
        mask = F.interpolate(mask[np.newaxis,np.newaxis],size=(375, 1242),mode='nearest')

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(np.array(img[0].permute((1,2,0))))
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        original_size = img.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])
        input_image_preprocess = (input_image_torch - self.sam.pixel_mean) / self.sam.pixel_std

        # Pad
        h, w = input_image_preprocess.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        input_image_preprocess = F.pad(input_image_preprocess, (0, padw, 0, padh))

        visible_mask = mask[0,0].clone()
        visible_mask[visible_mask > 0] -= 1

        amodal_mask = mask[0,0] - visible_mask

        amodal_mask[amodal_mask > 0.5] = 1
        amodal_mask[amodal_mask <= 0.5] = 0

        visible_mask[visible_mask > 0.5] = 1
        visible_mask[visible_mask <= 0.5] = 0

        points = torch.where(visible_mask==1)
        if len(points[0]) > 0: # select a random point from the visible mask
            coord = np.random.randint(0,len(points[0]))
            point = [points[1][coord].item(),points[0][coord].item()]
        else:
            points = torch.where(amodal_mask == 1)
            print('image has no visible mask:', img_name, flush=True)
            print('corresponding annotation:', anns, flush=True)
            coord = np.random.randint(0, len(points[0]))
            point = [points[1][coord].item(), points[0][coord].item()]
        point = np.array([point])
        point_labels = np.array([1])
        point_coords = self.transform.apply_coords(point, (375, 1242))
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        amodal_mask = amodal_mask[np.newaxis]
        image_size = tuple(img.shape[1:3])
        if self.test:
            return input_image_preprocess[0], amodal_mask, (375, 1242), \
                   input_size, (coords_torch[0], labels_torch[0]), img, \
                   image_id, visible_mask[np.newaxis], point, \
                   point_labels, 1  # 1 for track_id
        else:
            return input_image_preprocess[0], amodal_mask, (375, 1242), input_size, (coords_torch[0], labels_torch[0])

def get_kins_car_dataset(sam, sam_trans=None):
    image_root = "/beegfs/data/shared/kitti-kins/KINS_Video_Car/Kins/training/image_2/"
    gt_root = "/beegfs/data/shared/kitti-kins/KINS_Video_Car/car_data/train_data.pkl"
    ds_train = KINSCarDataset(image_root, gt_root, sam, sam_trans=sam_trans)
    image_root = "/beegfs/data/shared/kitti-kins/KINS_Video_Car/Kins/training/image_2/"
    gt_root = "/beegfs/data/shared/kitti-kins/KINS_Video_Car/car_data/val_data.pkl"
    ds_test = KINSCarDataset(image_root, gt_root, sam, train=False, sam_trans=sam_trans)
    return ds_train, ds_test

if __name__ == '__main__':

    sam_args = {
        'sam_checkpoint': "../cp/sam_vit_b_01ec64.pth",
        'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)

    ds_train, ds_test = get_kins_car_dataset(sam_trans=sam_trans)

