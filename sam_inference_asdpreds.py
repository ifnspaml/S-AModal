# this is sam_inference adapted to work with the GenVIS predictions on Amodal Synth Drive

# this script has a lot of options that need to be set
# make sure to set them before running
# this script is based on the inference script in https://github.com/talshaharabany/AutoSAM/blob/main/inference.py

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

from evaluation.ap_evaluation import customCOCOeval as COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import cvbase as cvb
import os
from utils.helper_functions import *
import json



def main(args):
    single_image = False

    device = "cuda"

    #dataset = 'kcar'
    dataset = 'asd'

    modeltype = "vit_b"
    model_checkpoint = "/beegfs/work/breitenstein/segment-anything/results/asd_1_0.00001_aw_samadpt_gpu270/net_best.pth"
    #model_checkpoint = "/beegfs/work/breitenstein/segment-anything/results/kcar_1_0.00001_aw_samadpt_gpu235/net_best.pth"


    mode = {'mode': 'samadpt'} #'mode is adapt or normal, samadpt
    pt_augmentation = None# 'saliency' # # can be: maxdis, maxent, random, saliency
    num_points = 1 #set number of points for prompting SAM
    print('settings are:', flush=True)
    print('dataset: ', dataset, flush=True)
    print('modeltype: ', modeltype, flush=True)
    print('model_checkpoint: ', model_checkpoint, flush=True)
    print('mode: ', mode, flush=True)
    print('pt_augmentation: ', pt_augmentation, flush=True)
    print('num_points: ', num_points, flush=True)
    print('savepath is: ', args['save_path'], flush=True)

    sam = sam_model_registry[modeltype](mode,checkpoint=model_checkpoint)#/beegfs/work/breitenstein/segment-anything/checkpoints/sam_vit_h_4b8939.pth")

    sam.to(device=device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.eval()
    #trainset,testset = get_kins_dataset(0,sam, sam_trans=transform, test=True)


    image_root = "/beegfs/data/shared/amodal_synth_drive_new_split/val/images/front/"
    gt_root = "/beegfs/data/shared/amodal_synth_drive_new_split/val/amodal_instance_seg/front/"
    transform_train, transform_test = get_kins_transform()

    catId = 24

    videos = sorted(os.listdir(gt_root))
    sam.eval()
    iou_list = []
    tp=0
    fp=0
    fn=0

    #asd predictions to load for inference
    labelspath = '/beegfs/work/temp/austausch_franz_jasmin/video_instance_segmentation/AmodalSynthDrive/asd_valid_genvis_1080_results.json'
    asd_preds = cvb.load(labelspath)
    #code to evaluate average precision
    # Step 1: Collect predictions and ground truth annotations
    predictions = []

    for ix, vid_ann in enumerate(asd_preds):
        video_id = vid_ann['video_id']
        video_name = videos[video_id-1]
        # load corresponding images
        frames = sorted(os.listdir(image_root + video_name))

        for frame_id, frame_ann in enumerate(vid_ann['segmentations']):
            print('frame is', frames[frame_id])
            print('loadpath is', image_root + video_name + '/' + frames[frame_id])
            frame = cv2.imread(image_root + video_name + '/' + frames[frame_id])#gt_root + video_name + '/' + frames[frame_id])
            print('frame shape', frame.shape, flush=True)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_shape = (1080, 1920)

            # load corresponding visible annotation
            if not frame_ann['counts'] == 'PPYo1':
                decoded_visible_mask = torch.tensor(mask_utils.decode(frame_ann))
                points = torch.where(decoded_visible_mask == 1)
                # print('points is', points, flush=True)
                coord = np.random.randint(0, len(points[0]))
                point1 = [points[1][coord].item(), points[0][coord].item()]
                point = np.array([point1])
                point_labels = np.array([1])


                # sam pre-processing of image and coords:
                img = torch.tensor(frame)
                img = img.permute((2, 0, 1))
                img = F.interpolate(img[np.newaxis], size=frame_shape, mode='bilinear')

                # Transform the image to the form expected by the model
                input_image = transform.apply_image(np.array(img[0].permute((1, 2, 0))))
                input_image_torch = torch.as_tensor(input_image, device=device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                original_size = img.shape[:2]

                input_size = tuple(input_image_torch.shape[-2:])
                input_image_preprocess = (input_image_torch - sam.pixel_mean) / sam.pixel_std

                # Pad
                h, w = input_image_preprocess.shape[-2:]
                padh = sam.image_encoder.img_size - h
                padw = sam.image_encoder.img_size - w
                input_image_preprocess = F.pad(input_image_preprocess, (0, padw, 0, padh))

                point_coords = transform.apply_coords(point, frame_shape)
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

                input_sam = input_image_preprocess[0]

                input_sam_point = [coords_torch[0], labels_torch[0]]
                modified_list = [tensor.unsqueeze(0) for tensor in
                                 input_sam_point]  # if len(tensor.shape) == 2 else tensor.unsqueeze(0).unsqueeze(0) for tensor in input_sam_point]


                # sam inference
                with torch.no_grad():
                    image_embedding = sam.image_encoder(input_sam[None])
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=modified_list,
                        boxes=None,
                        masks=None,
                    )
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                # upscaled_masks = sam.postprocess_masks(low_res_masks, img_sz, (375, 1242)).to(sam.device)
                upscaled_masks = sam.postprocess_masks(low_res_masks, (input_size[0], input_size[1]), frame_shape).to(
                    sam.device)


                from torch.nn.functional import threshold, normalize

                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(sam.device)

                predicted_mask = binary_mask.squeeze().detach().cpu().numpy().astype(np.uint8)
                predicted_mask = np.asfortranarray(predicted_mask)
                # plt.figure(figsize=(10, 10))
                # plt.imshow(frame.astype('uint8'))  # .transpose((1, 2, 0))
                # plt.imshow(binary_mask[0, 0].cpu().detach().numpy(), alpha=0.5)
                # # # plt.imshow(gts[0, 0].cpu().detach().numpy(), alpha=0.5)
                # show_points(point[0], point_labels[0], plt.gca())
                # # # show_points(points_small[0], input_label[0], plt.gca())
                # plt.savefig('try_asd_pred_output.png')
                # plt.close()

                pred_mask_encoded = mask_utils.encode(predicted_mask)
                pred_mask_encoded['counts'] = str(pred_mask_encoded['counts'], 'utf-8')
                image_id = 100*(video_id-1) + frame_id
                predictions.append({'image_id': image_id,
                                    'segmentation': pred_mask_encoded,
                                    'score': iou_predictions.item(),
                                    'category_id': catId,
                                    'height': frame_shape[0],
                                    'width': frame_shape[1],
                                    'track_id': ix,
                                    'original_point': point1})


    coco_gt = COCO('ground_truth_annotations.json')
    predictions_name = args['save_path'] #'predictions_asd_predbased_wgt_run1.json'


    with open(predictions_name, 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(predictions_name)

    # Step 3: Use COCO API to evaluate predictions against ground truth annotations
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Step 4: Calculate AP from the evaluation results
    # Assuming you want AP for a specific IoU threshold, e.g., 0.5
    average_precision = coco_eval.stats[0]  # AP at IoU threshold of 0.5
    print("Average Precision (AP) at IoU threshold 0.5:", average_precision)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-save_path', '--save_path', default='predictions_asd_wgt_saliency.json', help='filename to save predictions', required=False)

    #adapt mode from: https://github.com/KidsWithTokens/Medical-SAM-Adapter/tree/main
    #normal mode is standard sam
    #parser.add_argument('-optimizer', type=str, default='adam', help='current options: adam, galore, aw')
    #parser.add_argument('-dataset', type=str, default='kins', help='current options: kins,asd, kcar')
    args = vars(parser.parse_args())
    main(args)