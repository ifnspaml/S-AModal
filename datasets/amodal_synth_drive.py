from PIL import Image
import torch.utils.data as data
import cv2
import cvbase as cvb

import json
import torch.nn.functional as F
from utils.helper_functions import *
tor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide




#ASD_Dataset class is adapted from https://github.com/talshaharabany/AutoSAM/blob/main/dataset/polyp.py
class ASD_Dataset(data.Dataset):
    """
    dataloader for AmodalSynthDrive dataset
    """

    def __init__(self, image_root, gt_root, sam_model, trainsize=352, augmentations=None, train=True, sam_trans=None,test =False,firstcall = False,
                 pt_augmentation=None, num_points=1):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.image_root = image_root
        self.gt_root = gt_root
        self.test = test
        videos = sorted(os.listdir(gt_root))
        self.firstcall = firstcall
        self.anns_info = []
        self.image_dict = []
        self.pt_augmentation = pt_augmentation #can be None, saliency, maxdis, maxent, random
        self.num_points = num_points
        self.image_info = {}

        if self.test:
            self.coco_gt_dict = {
                        "info": {},
                        "licenses": [],
                        "images": [],
                        "annotations": [],
                        "categories": [{"name": "person", "supercategory": "human", "id": 24},
                                       {"name": "rider", "supercategory": "human", "id": 25},
                                       {"name": "car", "supercategory": "vehicle", "id": 26},
                                       {"name": "truck", "supercategory": "vehicle", "id": 27},
                                       {"name": "bus", "supercategory": "vehicle", "id": 28},
                                       {"name": "motor", "supercategory": "vehicle", "id": 32},
                                       {"name": "bike", "supercategory": "vehicle", "id": 33}],
                    }
            self.firstcall =True

        if self.firstcall:
            counter = 0
            instance_counter = 0
            for video in videos:
                frames = [f for f in sorted(os.listdir(gt_root + video)) if f.endswith(".json")]
                for frame in frames:
                    frame_annotations = cvb.load(gt_root + video + '/' + frame)
                    for j in frame_annotations:
                        if frame_annotations[j]['amodal_mask']:
                            self.anns_info.append(frame_annotations[j])
                            if self.test:
                                #fix for false annotations:
                                if frame_annotations[j]['category_id']==21:
                                    frame_annotations[j]['category_id']= 27 #wrong annotation in /20230702032750_SoftRainSunset_Town10HD/front_full_0022 : should be a truck 27
                                if frame_annotations[j]['category_id']==11:
                                    frame_annotations[j]['category_id']= 26 #wrong annotation in /20230706030929_WetCloudyNoon_Town10HD/front_full_0071 : should be a car 26
                                frame_annotations[j]['image_id']=counter
                                frame_annotations[j]['ignore'] = int(0)
                                frame_annotations[j]['segmentation'] = frame_annotations[j]['amodal_mask']
                                frame_annotations[j]['area'] = int(mask_utils.area(frame_annotations[j]['segmentation']))

                                #we need an occlusion rate
                                if frame_annotations[j]['area'] >0:
                                    occlusion_area = int(mask_utils.area(frame_annotations[j]['occlusion_mask']))
                                    frame_annotations[j]['occlude_rate'] = occlusion_area / frame_annotations[j]['area']
                                else:
                                    frame_annotations[j]['occlude_rate'] = int(0)

                                frame_annotations[j]['id'] = instance_counter
                                self.coco_gt_dict['annotations'].append(frame_annotations[j])
                            #self.anns_info[-1]['video'] = video #add videoname
                            #self.anns_info[-1]['frame'] = frame # add frame name --> needed to load image
                            self.image_dict.append(self.image_root + video + '/' + frame.split('_aminseg')[0] + '_rgb.jpg') #--> needed to load image

                        else:
                            #load the mask because amodal_mask only exists if occlusion is present
                            frame_path2 = gt_root + video + '/' + frame.split('.json')[0] + '.png'
                            mask = np.array(Image.open(frame_path2))
                            id_mask = mask.copy()

                            id_mask[id_mask != int(j)] = 0
                            id_mask[id_mask == int(j)] = 1
                            id_mask = id_mask.astype(np.uint8)
                            id_mask = np.asfortranarray(id_mask)


                            rle_encoding = mask_utils.encode(id_mask)
                            frame_annotations[j]['amodal_mask'] = rle_encoding
                            frame_annotations[j]['segmentation'] = rle_encoding
                            frame_annotations[j]['area'] = int(mask_utils.area(frame_annotations[j]['segmentation']))
                            frame_annotations[j]['ignore'] = int(0)
                            frame_annotations[j]['occlude_rate'] = int(0)
                            self.anns_info.append(frame_annotations[j])
                            self.image_dict.append(self.image_root + video + '/' + frame.split('_aminseg')[0] + '_rgb.jpg') #--> needed to load image
                            if self.test:
                                frame_annotations[j]['image_id']=counter
                                frame_annotations[j]['id'] = instance_counter
                                self.coco_gt_dict['annotations'].append(frame_annotations[j])
                        instance_counter+=1

                    self.image_info[self.image_root + video + '/' + frame.split('_aminseg')[0] + '_rgb.jpg'] = {
                        'image_id': counter, 'video': video}
                    if self.test:
                        self.coco_gt_dict['images'].append({'file_name': self.image_root + video + '/' + frame.split('_aminseg')[0] + '_rgb.jpg',
                                                       'id': counter,  'video': video, 'height': 1080, 'width': 1920 })
                    counter +=1



            # fix: check that we exclude any empty annotations
            count = 0
            for i in range(0, len(self.anns_info)):
                if self.anns_info[count]['amodal_mask']['counts'] == 'PPYo1':
                    self.anns_info.remove(self.anns_info[count])
                    self.image_dict.remove(self.image_dict[count])
                    count -= 1
                count += 1

            # save all the dictionaries, so that we can just load them instead of creating them every time:
            coco_gt_dict_serializable_annsinfo = convert_to_json_serializable(self.anns_info)
            with open('anns_info_asd_%s.json' % image_root.split('/')[5], 'w', encoding='utf-8') as f:
                json.dump(coco_gt_dict_serializable_annsinfo, f)

            coco_gt_dict_serializable_img_dict = convert_to_json_serializable(self.image_dict)
            with open('imgs_dict_asd_%s.json' % image_root.split('/')[5], 'w', encoding='utf-8') as f:
                json.dump(coco_gt_dict_serializable_img_dict, f)

            coco_gt_dict_serializable_image_info = convert_to_json_serializable(self.image_info)
            with open('imgage_info_asd_%s.json' % image_root.split('/')[5], 'w', encoding='utf-8') as f:
                json.dump(coco_gt_dict_serializable_image_info, f)
        else:
            with open('anns_info_asd_%s.json' % image_root.split('/')[5], 'r', encoding='utf-8') as f:
                self.anns_info = json.load(f)
            with open('imgs_dict_asd_%s.json' % image_root.split('/')[5], 'r', encoding='utf-8') as f:
                self.image_dict = json.load(f)
            with open('imgage_info_asd_%s.json' % image_root.split('/')[5], 'r', encoding='utf-8') as f:
                self.image_info = json.load(f)


        self.size = len(self.anns_info)
        self.train = train
        self.sam_trans = sam_trans
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = sam_model
        if self.test:
            # Step 1: Convert bytes objects to UTF-8 strings
            coco_gt_dict_serializable  = convert_to_json_serializable(self.coco_gt_dict)
            with open('ground_truth_annotations.json', 'w', encoding='utf-8') as f:
                json.dump(coco_gt_dict_serializable, f)



    def __len__(self):
        return len(self.anns_info)

    def __getitem__(self, index):

        # load annotations
        anns = self.anns_info[index]# annotation for index

        # get image path and id
        img_path = self.image_dict[index]#['file_name']
        image_id = self.image_info[img_path]['image_id']

        # load the image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)#cv2.imread('/beegfs/work/shared/kitti-kins/training/image_2/000000.png')#
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        a_mask = polys_to_mask(anns["amodal_mask"], height, width)

        # lots of work
        if anns["occlusion_mask"]:
            occ_mask = polys_to_mask(anns["occlusion_mask"], height, width)
            i_mask = a_mask - occ_mask
            mask = a_mask + i_mask
        else:
            i_mask = a_mask
            mask = a_mask + i_mask


        mask = mask.astype('uint8')
        img = torch.tensor(image)
        img = img.permute((2, 0, 1))
        mask = torch.tensor(mask)
        img = F.interpolate(img[np.newaxis], size=(height, width), mode='bilinear')
        mask = F.interpolate(mask[np.newaxis, np.newaxis], size=(height, width), mode='nearest')

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(np.array(img[0].permute((1,2,0))))
        input_image_torch = torch.as_tensor(input_image)#, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        original_size = img.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])
        input_image_preprocess = (input_image_torch - self.sam.pixel_mean.cpu()) / self.sam.pixel_std.cpu()

        # Pad
        h, w = input_image_preprocess.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        input_image_preprocess = F.pad(input_image_preprocess, (0, padw, 0, padh))

        img = img.to(dtype = torch.uint8)

        visible_mask = mask[0,0].clone()
        visible_mask[visible_mask > 0] -= 1

        amodal_mask = mask[0,0] - visible_mask

        amodal_mask[amodal_mask > 0.5] = 1
        amodal_mask[amodal_mask <= 0.5] = 0

        visible_mask[visible_mask > 0.5] = 1
        visible_mask[visible_mask <= 0.5] = 0

        points = torch.where(visible_mask == 1)

        #per default currently erosion is added as results are better
        if len(points[0])>0:
            vm_numpy = visible_mask.numpy()
            import skimage
            eroded = skimage.morphology.binary_erosion(vm_numpy,footprint = skimage.morphology.square(7))
            # footprint gives neighborhood for erosion
            # default is square(3)

            # check that after erosion there is still a mask
            if eroded.any():
                points_eroded = np.where(eroded == 1)
                coord = np.random.randint(0, len(points_eroded[0]))
                point1 = [points_eroded[1][coord].item(), points_eroded[0][coord].item()]
            else:
                coord = np.random.randint(0,len(points[0]))

                point1 = [points[1][coord].item(),points[0][coord].item()]
        else:
            points = torch.where(amodal_mask == 1)
            print('image has no visible mask:', img_path, flush=True)
            print('corresponding annotation:', anns, flush=True)
            coord = np.random.randint(0, len(points[0]))
            point1 = [points[1][coord].item(), points[0][coord].item()]
        point = np.array([point1])
        point_labels = np.array([1])

        if self.pt_augmentation:
            print('unique of i_mask', np.unique(i_mask))
            if 'maxent' in self.pt_augmentation:
                entropy_point = get_entropy_points(point1, i_mask, image)
                point = np.array([entropy_point])
                point_labels = np.array([1])
            if 'saliency' in self.pt_augmentation:
                vst_random_point = get_saliency_point(image, i_mask, img_path)
                point = np.array([vst_random_point])
                point_labels = np.array([1])
            if 'maxdis' in self.pt_augmentation:
                max_dis_point = get_distance_points(point1, i_mask)
                point = np.array([max_dis_point])
                point_labels = np.array([1])
            if 'random' in self.pt_augmentation:
                random_point = get_random_point(i_mask)
                point = np.array([random_point])
                point_labels = np.array([1])

        if self.num_points>1:
            point_list = [point1]
            for point_count in range(1,self.num_points):
                new_point = get_random_point(i_mask)
                point_list.append(new_point)
            point = np.array(point_list)
            point_labels = np.array([1]*self.num_points)

        point_coords = self.transform.apply_coords(point, (height, width))
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        amodal_mask = amodal_mask[np.newaxis]
        image_size = tuple(img.shape[1:3])

        # different returns for training and test
        if self.test:
            return input_image_preprocess[0], amodal_mask, (height, width), input_size, (
            coords_torch[0], labels_torch[0]),img, image_id, visible_mask[np.newaxis], point, \
                   point_labels, anns['track_id']
        else:
            return input_image_preprocess[0], amodal_mask, (height, width), input_size, (coords_torch[0], labels_torch[0])



def get_asd_dataset(sam, sam_trans=None):
    image_root = "/beegfs/data/shared/amodal_synth_drive_new_split/train/images/front/"
    gt_root = "/beegfs/data/shared/amodal_synth_drive_new_split/train/amodal_instance_seg/front/"
    ds_train = ASD_Dataset(image_root, gt_root, sam, sam_trans=sam_trans)
    image_root = "/beegfs/data/shared/amodal_synth_drive_new_split/val/images/front/"
    gt_root = "/beegfs/data/shared/amodal_synth_drive_new_split/val/amodal_instance_seg/front/"
    ds_test = ASD_Dataset(image_root, gt_root, sam, train=False, sam_trans=sam_trans)
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

    ds_train, ds_test = get_asd_dataset(sam_trans=sam_trans)
    ds = data.DataLoader(dataset=ds_test,
                         batch_size=1,
                         shuffle=False,
                         num_workers=1)
